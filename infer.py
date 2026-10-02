#!/usr/bin/env python3
"""
Inference script for deployment on the observer node.

Runs the full pipeline on a raw RTT trace and reports:
  - Per-model predictions (15-class and 11-class)
  - Timing breakdown for each pipeline stage

Usage:
  python3 infer.py deployed_models/ victim_trace.json
  python3 infer.py deployed_models/ victim_trace.json --no-cuda
"""
import os, sys, time, argparse
import numpy as np, joblib
import torch, torch.nn as nn

parser = argparse.ArgumentParser(description="RTT workload inference — observer-side inference")
parser.add_argument("model_dir", help="Directory with exported model artifacts")
parser.add_argument("trace_file", help="Raw RTT JSON file")
parser.add_argument("--no-cuda", action="store_true", help="Force CPU inference")
ARGS = parser.parse_args()

DEVICE = torch.device("cpu") if ARGS.no_cuda else torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_SEC = 60
STRIDE_SEC = 10
MAX_WINDOWS = 25

# ── Import feature extractor ──────────────────────────────────────
# Look for extract_features_v3_fixed.py in same dir as this script or model_dir
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, ARGS.model_dir)
from extract_features_v3_fixed import extract_window_features, load_trace, normalize_rtt


# ── Model definitions (must match export) ─────────────────────────
class WindowTransformer(nn.Module):
    def __init__(self, nc, fd, mw=25, ed=64, nh=4, nl=4, do=0.3):
        super().__init__()
        self.fp = nn.Sequential(nn.Linear(fd, ed), nn.GELU(), nn.Dropout(do))
        self.cls = nn.Parameter(torch.randn(1, 1, ed) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, mw + 1, ed) * 0.02)
        self.do = nn.Dropout(do)
        el = nn.TransformerEncoderLayer(d_model=ed, nhead=nh, dim_feedforward=ed*2,
            dropout=do, activation="gelu", batch_first=True)
        self.tr = nn.TransformerEncoder(el, num_layers=nl)
        self.nm = nn.LayerNorm(ed)
        self.hd = nn.Sequential(nn.Linear(ed, ed), nn.GELU(), nn.Dropout(do), nn.Linear(ed, nc))
    def forward(self, x):
        b, nw, _ = x.shape
        p = self.fp(x)
        t = torch.cat([self.cls.expand(b, -1, -1), p], dim=1)
        t = self.do(t + self.pos[:, :nw+1])
        return self.hd(self.nm(self.tr(t)[:, 0]))


class LSTMClassifier(nn.Module):
    def __init__(self, fd, hd, nc, nl=2, do=0.3):
        super().__init__()
        self.lstm = nn.LSTM(fd, hd, nl, batch_first=True, dropout=do if nl > 1 else 0, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(do), nn.Linear(hd * 2, nc))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


# ═════════════════════════════════════════════════════════════════════
# Load artifacts
# ═════════════════════════════════════════════════════════════════════
print("=" * 60)
print(f"RTT Workload Inference")
print(f"  Model dir: {ARGS.model_dir}")
print(f"  Trace:     {ARGS.trace_file}")
print(f"  Device:    {DEVICE}")
print("=" * 60)

timings = {}

t0 = time.perf_counter()
scaler_pca = joblib.load(os.path.join(ARGS.model_dir, "scaler_pca.joblib"))
pca = joblib.load(os.path.join(ARGS.model_dir, "pca_model.joblib"))
scaler_clf = joblib.load(os.path.join(ARGS.model_dir, "scaler_clf.joblib"))
le_15 = joblib.load(os.path.join(ARGS.model_dir, "label_encoder_15.joblib"))
le_11 = joblib.load(os.path.join(ARGS.model_dir, "label_encoder_11.joblib"))

rf = joblib.load(os.path.join(ARGS.model_dir, "rf_model.joblib"))
lgb_clf = joblib.load(os.path.join(ARGS.model_dir, "lgb_model.joblib"))
mlp = joblib.load(os.path.join(ARGS.model_dir, "mlp_model.joblib"))
svm = joblib.load(os.path.join(ARGS.model_dir, "svm_model.joblib"))

n_feat = pca.n_components_
labels_15 = list(le_15.classes_)
labels_11 = list(le_11.classes_)
n_classes_15 = len(labels_15)
n_classes_11 = len(labels_11)

# Load 15-class sequence models
wt_models_15 = []
for seed in [42, 123, 456, 789, 999]:
    path = os.path.join(ARGS.model_dir, f"wt_seed{seed}.pt")
    if os.path.exists(path):
        m = WindowTransformer(n_classes_15, n_feat).to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        m.eval()
        wt_models_15.append(m)

lstm_models_15 = []
for seed in [42, 123, 456]:
    path = os.path.join(ARGS.model_dir, f"lstm_seed{seed}.pt")
    if os.path.exists(path):
        m = LSTMClassifier(n_feat, 64, n_classes_15).to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        m.eval()
        lstm_models_15.append(m)

# Load 11-class models
rf_11 = joblib.load(os.path.join(ARGS.model_dir, "rf_model_11.joblib"))
lgb_11 = joblib.load(os.path.join(ARGS.model_dir, "lgb_model_11.joblib"))
mlp_11 = joblib.load(os.path.join(ARGS.model_dir, "mlp_model_11.joblib"))
svm_11 = joblib.load(os.path.join(ARGS.model_dir, "svm_model_11.joblib"))

wt_models_11 = []
for seed in [42, 123, 456, 789, 999]:
    path = os.path.join(ARGS.model_dir, f"wt_11_seed{seed}.pt")
    if os.path.exists(path):
        m = WindowTransformer(n_classes_11, n_feat).to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        m.eval()
        wt_models_11.append(m)

lstm_models_11 = []
for seed in [42, 123, 456]:
    path = os.path.join(ARGS.model_dir, f"lstm_11_seed{seed}.pt")
    if os.path.exists(path):
        m = LSTMClassifier(n_feat, 64, n_classes_11).to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        m.eval()
        lstm_models_11.append(m)

timings["load_models"] = time.perf_counter() - t0
print(f"\n  Models loaded in {timings['load_models']:.3f}s")


# ═════════════════════════════════════════════════════════════════════
# Stage 1: Load trace
# ═════════════════════════════════════════════════════════════════════
t0 = time.perf_counter()
ts, rtt = load_trace(ARGS.trace_file)
timings["load_trace"] = time.perf_counter() - t0
print(f"\n  Stage 1 — Load trace: {len(ts)} samples in {timings['load_trace']:.4f}s")


# ═════════════════════════════════════════════════════════════════════
# Stage 2: Normalize + NaN filter
# ═════════════════════════════════════════════════════════════════════
t0 = time.perf_counter()
normed, baseline = normalize_rtt(rtt)
valid_mask = ~np.isnan(normed)
n_dropped = int((~valid_mask).sum())
ts_v = ts[valid_mask]
normed_v = normed[valid_mask]

diffs = np.diff(ts_v)
diffs = diffs[diffs > 0]
sample_rate = 1.0 / np.median(diffs) if len(diffs) else 10.0
timings["normalize"] = time.perf_counter() - t0
print(f"  Stage 2 — Normalize: baseline={baseline:.2f}ms, "
      f"{n_dropped} NaN dropped, sr={sample_rate:.1f}Hz in {timings['normalize']:.4f}s")


# ═════════════════════════════════════════════════════════════════════
# Stage 3: Windowing + feature extraction
# ═════════════════════════════════════════════════════════════════════
t0 = time.perf_counter()
t_start = ts_v[0]
duration = ts_v[-1] - t_start
feature_dicts = []
win_start = 0.0
while win_start + WINDOW_SEC <= duration + 0.5:
    mask = (ts_v >= t_start + win_start) & (ts_v < t_start + win_start + WINDOW_SEC)
    window = normed_v[mask]
    if len(window) >= 20:
        feats = extract_window_features(window, sample_rate)
        if feats is not None:
            feature_dicts.append(feats)
    win_start += STRIDE_SEC

n_windows = len(feature_dicts)
timings["feature_extraction"] = time.perf_counter() - t0
per_window_feat = timings["feature_extraction"] / max(n_windows, 1)
print(f"  Stage 3 — Feature extraction: {n_windows} windows in "
      f"{timings['feature_extraction']:.4f}s ({per_window_feat*1000:.2f}ms/window)")

if n_windows == 0:
    print("\n  ERROR: No valid windows extracted. Trace too short or all NaN.")
    sys.exit(1)

# Convert to numpy array (maintain feature order from first dict)
feat_names = list(feature_dicts[0].keys())
X_104 = np.array([[d[f] for f in feat_names] for d in feature_dicts], dtype=np.float32)


# ═════════════════════════════════════════════════════════════════════
# Stage 4: PCA transform
# ═════════════════════════════════════════════════════════════════════
t0 = time.perf_counter()
X_pca_scaled = scaler_pca.transform(X_104)
X_pca = pca.transform(X_pca_scaled).astype(np.float32)
timings["pca"] = time.perf_counter() - t0
print(f"  Stage 4 — PCA: {X_104.shape[1]} → {X_pca.shape[1]} in {timings['pca']:.4f}s")


# ═════════════════════════════════════════════════════════════════════
# Stage 5: Scaler transform
# ═════════════════════════════════════════════════════════════════════
t0 = time.perf_counter()
X_final = scaler_clf.transform(X_pca)
timings["scaler"] = time.perf_counter() - t0
print(f"  Stage 5 — Scaler: {timings['scaler']:.4f}s")


# ═════════════════════════════════════════════════════════════════════
# Stage 6: Classifier inference
# ═════════════════════════════════════════════════════════════════════
def bayesian_aggregate(proba, class_labels):
    lp = np.sum(np.log(proba + 1e-10), axis=0)
    s = lp - np.max(lp); p = np.exp(s); p /= p.sum()
    return class_labels[np.argmax(p)], p

def majority_vote(preds):
    from collections import Counter
    return Counter(preds).most_common(1)[0][0]

def seq_inference(models, X_scaled, n_classes):
    if len(X_scaled) >= MAX_WINDOWS:
        seq = X_scaled[:MAX_WINDOWS]
    else:
        seq = np.vstack([X_scaled, np.zeros((MAX_WINDOWS - len(X_scaled), X_scaled.shape[1]))])
    x = torch.FloatTensor(seq).unsqueeze(0).to(DEVICE)
    posteriors = []
    with torch.no_grad():
        for m in models:
            p = torch.softmax(m(x), dim=1).cpu().numpy()[0]
            posteriors.append(p)
    return np.mean(posteriors, axis=0)

print(f"\n{'='*60}")
print("PREDICTIONS")
print(f"{'='*60}")

for n_class_label, labels, clfs, wt_models, lstm_models in [
    ("15-class", labels_15,
     {"RF": rf, "SVM": svm, "MLP": mlp, "LGB": lgb_clf},
     wt_models_15, lstm_models_15),
    ("11-class", labels_11,
     {"RF": rf_11, "SVM": svm_11, "MLP": mlp_11, "LGB": lgb_11},
     wt_models_11, lstm_models_11),
]:
    print(f"\n  --- {n_class_label} ---")
    print(f"  {'Model':<10s} {'Prediction':<20s} {'Time':>10s} {'Confidence':>10s}")

    lgb_posterior = None

    for name, clf in clfs.items():
        t0 = time.perf_counter()
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X_final)
            pred, posterior = bayesian_aggregate(proba, labels)
            conf = np.max(posterior)
        else:
            preds = [labels[p] for p in clf.predict(X_final)]
            pred = majority_vote(preds)
            conf = 0
        elapsed = time.perf_counter() - t0
        timings[f"{name}_{n_class_label}"] = elapsed
        print(f"  {name:<10s} {pred:<20s} {elapsed*1000:>9.2f}ms {conf:>9.1%}")

        if name == "LGB":
            lgb_posterior = posterior

    # LSTM
    if lstm_models:
        t0 = time.perf_counter()
        lstm_post = seq_inference(lstm_models, X_final.astype(np.float32), len(labels))
        pred = labels[np.argmax(lstm_post)]
        elapsed = time.perf_counter() - t0
        timings[f"LSTM_{n_class_label}"] = elapsed
        print(f"  {'LSTM':<10s} {pred:<20s} {elapsed*1000:>9.2f}ms {np.max(lstm_post):>9.1%}")

    # WT
    if wt_models:
        t0 = time.perf_counter()
        wt_post = seq_inference(wt_models, X_final.astype(np.float32), len(labels))
        pred = labels[np.argmax(wt_post)]
        elapsed = time.perf_counter() - t0
        timings[f"WT_{n_class_label}"] = elapsed
        print(f"  {'WT':<10s} {pred:<20s} {elapsed*1000:>9.2f}ms {np.max(wt_post):>9.1%}")

    # LGB+WT ensemble
    if lgb_posterior is not None and wt_models:
        t0 = time.perf_counter()
        best_pred, best_conf = None, 0
        for alpha in np.arange(0, 1.01, 0.05):
            combined = alpha * lgb_posterior + (1 - alpha) * wt_post
            c = np.max(combined)
            if c > best_conf:
                best_conf = c
                best_pred = labels[np.argmax(combined)]
        elapsed = time.perf_counter() - t0
        timings[f"LGB+WT_{n_class_label}"] = elapsed
        print(f"  {'LGB+WT':<10s} {best_pred:<20s} {elapsed*1000:>9.2f}ms {best_conf:>9.1%}")


# ═════════════════════════════════════════════════════════════════════
# Timing summary
# ═════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("TIMING SUMMARY")
print(f"{'='*60}")

pipeline_time = (timings["load_trace"] + timings["normalize"] +
                 timings["feature_extraction"] + timings["pca"] + timings["scaler"])

print(f"\n  Pipeline (trace → features):")
print(f"    Load trace:          {timings['load_trace']*1000:>8.2f} ms")
print(f"    Normalize + filter:  {timings['normalize']*1000:>8.2f} ms")
print(f"    Feature extraction:  {timings['feature_extraction']*1000:>8.2f} ms  "
      f"({per_window_feat*1000:.2f} ms/window × {n_windows})")
print(f"    PCA transform:       {timings['pca']*1000:>8.2f} ms")
print(f"    Scaler transform:    {timings['scaler']*1000:>8.2f} ms")
print(f"    ────────────────────────────────")
print(f"    Total pipeline:      {pipeline_time*1000:>8.2f} ms")
print(f"\n  Model loading:         {timings['load_models']*1000:>8.2f} ms  (one-time cost)")
print(f"  Windows extracted:     {n_windows}")
print(f"  Trace duration:        {duration:.1f}s")
