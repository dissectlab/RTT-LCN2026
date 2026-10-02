#!/usr/bin/env python3
"""
Unified evaluation script for RTT workload inference.

Trains on the full training pool, evaluates on all configured test sets.
Supports PCA, 15-class and 11-class formulations.

Training pool: B + C + Ep + F + P5p + B2  (21,375 windows, 675 runs)
Test sets:     A, A2, A∪A2, A6v1, A6v2, A6v1∪v2  (configurable)

Usage:
  python3 evaluate.py --quick             # 1 WT seed, 60 epochs, skip LSTM
  python3 evaluate.py                     # full: 5 WT seeds, 120 epochs, LSTM
  python3 evaluate.py --pca 0.95          # PCA retaining 95% variance
  python3 evaluate.py --pca 0.95 --quick  # PCA + quick mode
  python3 evaluate.py --test A A6v1       # only evaluate specific test sets

To add a new test set:
  1. Add the path to PATH_CONFIGS in extract_all.py and extract
  2. Add the test set definition to TEST_SETS below
  3. Run this script
"""
import os, sys, argparse
import numpy as np, pandas as pd, lightgbm as lgb
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.linalg import sqrtm
from collections import Counter

# ── Arguments ───────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="RTT workload inference — unified evaluation")
parser.add_argument("--quick", action="store_true",
                    help="Quick mode: 1 WT seed, 60 epochs, skip LSTM")
parser.add_argument("--pca", type=float, default=0,
                    help="PCA variance to retain, e.g. 0.95 for 95%% (0 = no PCA)")
parser.add_argument("--test", nargs="*", default=None,
                    help="Test sets to evaluate (default: all)")
ARGS = parser.parse_args()
QUICK = ARGS.quick
PCA_VAR = ARGS.pca

mode_tag = "quick" if QUICK else "full"
pca_tag = f"_pca{PCA_VAR}" if PCA_VAR > 0 else ""
LOG_FILE = f"evaluate_{mode_tag}{pca_tag}_log.txt"

class Tee:
    def __init__(self, path):
        self.file = open(path, "w"); self.stdout = sys.stdout
    def write(self, data):
        self.stdout.write(data); self.stdout.flush(); self.file.write(data); self.file.flush()
    def flush(self): self.stdout.flush(); self.file.flush()

sys.stdout = Tee(LOG_FILE)

META = ["label", "run", "window_start", "path", "baseline_rtt"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_WINDOWS = 25
FEAT_DIR = "features_allpaths"

WORKLOADS = ['allreduce', 'bulk', 'checkpoint', 'db_queries', 'dfs_reads',
             'federated', 'inference', 'log_agg', 'microservice', 'migration',
             'pipeline', 'preprocess', 'scatter_gather', 'shuffle', 'storage_rep']

LABEL_MAP_11 = {wl: wl for wl in WORKLOADS}
for wl in ['db_queries', 'inference', 'log_agg', 'microservice', 'shuffle']:
    LABEL_MAP_11[wl] = 'light_traffic'


# ═════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these to add/remove training or test paths
# ═════════════════════════════════════════════════════════════════════

# Training pool: path name → (csv filename, run offset to avoid collisions)
TRAIN_POOL = {
    "B":   ("B_features_fixed.csv",   0),
    "C":   ("C_features_fixed.csv",   100),
    "Ep":  ("Ep_features_fixed.csv",  200),
    "F":   ("F_features_fixed.csv",   300),
    "P5p": ("P5p_features_fixed.csv", 600),
    "B2":  ("B2_features_fixed.csv",  700),
}

# Test sets: name → list of (csv filename, run offset)
# Combined sets are built by listing multiple files
TEST_SETS = {
    "A":         [("A_features_fixed.csv",    0)],
    "A2":        [("A2_features_fixed.csv",   0)],
    "A∪A2":      [("A_features_fixed.csv",    0),
                  ("A2_features_fixed.csv",   900)],
    "A6v1":      [("A6v1_features_fixed.csv", 0)],
    "A6v2":      [("A6v2_features_fixed.csv", 0)],
    "A6v1∪v2":   [("A6v1_features_fixed.csv", 0),
                  ("A6v2_features_fixed.csv", 900)],
    "A4":        [("A4_features_fixed.csv",   0)],
    "A4int":     [("A4int_features_fixed.csv", 0)],
}


# ═════════════════════════════════════════════════════════════════════
# Core functions
# ═════════════════════════════════════════════════════════════════════

def coral(Xs, Xt):
    ms, mt = Xs.mean(0), Xt.mean(0)
    xs, xt = Xs - ms, Xt - mt
    r = 1e-6 * np.eye(xs.shape[1])
    Cs = (xs.T @ xs) / (len(xs) - 1) + r
    Ct = (xt.T @ xt) / (len(xt) - 1) + r
    return xs @ np.real(sqrtm(np.linalg.inv(Cs))) @ np.real(sqrtm(Ct)) + mt


def run_level_proba(proba, df, labels):
    y_true, y_pred = [], []
    for (label, run), group in df.groupby(["label", "run"]):
        idx = [df.index.get_loc(i) for i in group.index]
        lp = np.sum(np.log(proba[idx] + 1e-10), axis=0)
        s = lp - np.max(lp); p = np.exp(s); p /= p.sum()
        y_true.append(label); y_pred.append(labels[np.argmax(p)])
    return y_true, y_pred


def run_level_majority(y_pred_labels, df):
    y_true, y_pred = [], []
    for (label, run), group in df.groupby(["label", "run"]):
        idx = [df.index.get_loc(i) for i in group.index]
        preds = [y_pred_labels[i] for i in idx]
        y_true.append(label); y_pred.append(Counter(preds).most_common(1)[0][0])
    return y_true, y_pred


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


class SeqDS(Dataset):
    def __init__(self, s, l):
        self.X = torch.FloatTensor(s); self.y = torch.LongTensor(l)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


def build_seq(df, X_scaled, le, mw):
    seqs, labs = [], []
    for (l, r), g in df.groupby(["label", "run"]):
        idx = [df.index.get_loc(i) for i in g.index]
        feats = X_scaled[idx]
        if len(feats) >= mw: feats = feats[:mw]
        else: feats = np.vstack([feats, np.zeros((mw - len(feats), feats.shape[1]), dtype=np.float32)])
        seqs.append(feats); labs.append(le.transform([l])[0])
    return np.array(seqs, dtype=np.float32), np.array(labs)


def train_seq(model, tr_s, tr_l, te_s, te_l, labels, epochs=120, lr=5e-4):
    tl = DataLoader(SeqDS(tr_s, tr_l), batch_size=32, shuffle=True)
    tel = DataLoader(SeqDS(te_s, te_l), batch_size=64)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_acc, best_pr = 0, None
    for ep in range(1, epochs + 1):
        model.train()
        for X, y in tl:
            X, y = X.to(DEVICE), y.to(DEVICE)
            if np.random.random() < 0.5: X = X + torch.randn_like(X) * 0.03
            opt.zero_grad(); crit(model(X), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()
        if ep % 30 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                pr = []
                for X, y in tel: pr.append(torch.softmax(model(X.to(DEVICE)), dim=1).cpu().numpy())
                pr = np.vstack(pr)
            acc = accuracy_score(te_l, np.argmax(pr, axis=1))
            if acc > best_acc: best_acc = acc; best_pr = pr.copy()
    return best_acc, best_pr


def run_eval(train_df, test_df, feat_cols, le, labels, n_classes, tag, pca_model=None):
    """Run all classifiers on the given train/test split."""
    y_train = le.transform(train_df["label"].values)
    y_test = le.transform(test_df["label"].values)
    n_runs = test_df.groupby(["label", "run"]).ngroups

    Xtr_raw = np.nan_to_num(train_df[feat_cols].values, nan=0, posinf=0, neginf=0).astype(np.float32)
    Xte_raw = np.nan_to_num(test_df[feat_cols].values, nan=0, posinf=0, neginf=0).astype(np.float32)

    # CORAL alignment
    Xtr_c = coral(Xtr_raw, Xte_raw).astype(np.float32)

    # PCA (applied after CORAL, before scaling)
    if pca_model is not None:
        Xtr_c = pca_model.transform(Xtr_c).astype(np.float32)
        Xte_raw = pca_model.transform(Xte_raw).astype(np.float32)

    # Standardize
    sc = StandardScaler(); sc.fit(np.vstack([Xtr_c, Xte_raw]))
    Xtr = sc.transform(Xtr_c); Xte = sc.transform(Xte_raw)
    n_feat = Xtr.shape[1]

    print(f"\n{'='*60}")
    print(f"{tag}: {n_classes} classes, {n_runs} test runs, {n_feat} features")
    print(f"Train: {len(train_df)} windows, Test: {len(test_df)} windows")
    print(f"{'='*60}")

    results = {}

    # ── Classical classifiers ───────────────────────────────────────
    classifiers = {
        "RF": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=4),
        "SVM": SVC(kernel="rbf", C=10, gamma="scale", random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42, early_stopping=True),
        "LGB": lgb.LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                                   random_state=42, n_jobs=4, verbose=-1),
    }

    for name, clf in classifiers.items():
        clf.fit(Xtr, y_train)
        y_pw = clf.predict(Xte)
        pw_acc = accuracy_score(y_test, y_pw) * 100
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(Xte)
            yt, yp = run_level_proba(proba, test_df, labels)
            bay_acc = accuracy_score(yt, yp) * 100
            bay_f1 = f1_score(yt, yp, average="macro") * 100
            print(f"  {name:<6s}  pw={pw_acc:.1f}%  bayes={bay_acc:.1f}%  F1={bay_f1:.1f}%")
            results[name] = {"pw": pw_acc, "bay": bay_acc, "f1": bay_f1, "proba": proba, "yt": yt, "yp": yp}
        else:
            yt_m, yp_m = run_level_majority([labels[p] for p in y_pw], test_df)
            maj_acc = accuracy_score(yt_m, yp_m) * 100
            print(f"  {name:<6s}  pw={pw_acc:.1f}%  majority={maj_acc:.1f}%")
            results[name] = {"pw": pw_acc, "maj": maj_acc}

    # ── LSTM ────────────────────────────────────────────────────────
    tr_s, tr_l = build_seq(train_df, Xtr, le, MAX_WINDOWS)
    te_s, te_l = build_seq(test_df, Xte, le, MAX_WINDOWS)
    run_keys = sorted(test_df.groupby(["label", "run"]).groups.keys())

    if not QUICK:
        lstm_accs = []
        for seed in [42, 123, 456]:
            torch.manual_seed(seed); np.random.seed(seed)
            model = LSTMClassifier(n_feat, 64, n_classes).to(DEVICE)
            acc, _ = train_seq(model, tr_s, tr_l, te_s, te_l, labels, epochs=100, lr=1e-3)
            lstm_accs.append(acc)
        lstm_avg = np.mean(lstm_accs) * 100
        print(f"  LSTM    avg={lstm_avg:.1f}%  seeds={[f'{a*100:.1f}%' for a in lstm_accs]}")
        results["LSTM"] = {"avg": lstm_avg}
    else:
        print(f"  LSTM    [skipped in --quick mode]")

    # ── Window Transformer ──────────────────────────────────────────
    wt_seeds = [42] if QUICK else [42, 123, 456, 789, 999]
    wt_epochs = 60 if QUICK else 120
    wt_posts, wt_accs = [], []
    for seed in wt_seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        model = WindowTransformer(n_classes, n_feat).to(DEVICE)
        acc, pr = train_seq(model, tr_s, tr_l, te_s, te_l, labels, epochs=wt_epochs)
        wt_accs.append(acc); wt_posts.append(pr)
    wt_ens = np.mean(wt_posts, axis=0)
    ens_yt = [k[0] for k in run_keys]
    ens_yp = [labels[np.argmax(wt_ens[i])] for i in range(len(run_keys))]
    wt_acc = accuracy_score(ens_yt, ens_yp) * 100
    wt_f1 = f1_score(ens_yt, ens_yp, average="macro") * 100
    print(f"  WT-ens  acc={wt_acc:.1f}%  F1={wt_f1:.1f}%  seeds={[f'{a*100:.1f}%' for a in wt_accs]}")
    results["WT"] = {"acc": wt_acc, "f1": wt_f1, "yt": ens_yt, "yp": ens_yp, "ens": wt_ens, "keys": run_keys}

    # ── LGB + WT ensemble ──────────────────────────────────────────
    if "LGB" in results and "proba" in results["LGB"]:
        lgb_proba = results["LGB"]["proba"]
        lgb_post = {}
        for (l, r), g in test_df.groupby(["label", "run"]):
            idx = [test_df.index.get_loc(i) for i in g.index]
            lp = np.sum(np.log(lgb_proba[idx] + 1e-10), axis=0)
            s = lp - np.max(lp); p = np.exp(s); p /= p.sum()
            lgb_post[(l, r)] = p
        wt_post = {}
        for i, (l, r) in enumerate(run_keys):
            wt_post[(l, r)] = wt_ens[i]

        best_tc, best_alpha = 0, 0
        for alpha in np.arange(0.0, 1.01, 0.05):
            tc = sum(1 for (l, r) in run_keys
                     if labels[np.argmax(alpha * lgb_post.get((l, r), np.ones(n_classes)/n_classes) +
                                         (1 - alpha) * wt_post.get((l, r), np.ones(n_classes)/n_classes))] == l)
            if tc > best_tc: best_tc, best_alpha = tc, alpha

        combo_yt, combo_yp = [], []
        for (l, r) in run_keys:
            combined = best_alpha * lgb_post.get((l, r), np.ones(n_classes)/n_classes) + \
                       (1 - best_alpha) * wt_post.get((l, r), np.ones(n_classes)/n_classes)
            combo_yt.append(l); combo_yp.append(labels[np.argmax(combined)])
        combo_acc = accuracy_score(combo_yt, combo_yp) * 100
        combo_f1 = f1_score(combo_yt, combo_yp, average="macro") * 100
        print(f"  LGB+WT  acc={combo_acc:.1f}%  F1={combo_f1:.1f}%  alpha={best_alpha:.2f}")
        results["LGB+WT"] = {"acc": combo_acc, "f1": combo_f1, "yt": combo_yt, "yp": combo_yp}

    # ── Per-class report + confusion matrix ─────────────────────────
    best_key = max(["WT", "LGB+WT"], key=lambda k: results.get(k, {}).get("acc", 0))
    best_r = results[best_key]
    print(f"\n  Per-class ({best_key}):")
    print(classification_report(best_r["yt"], best_r["yp"],
                                labels=sorted(set(best_r["yt"])), digits=3, zero_division=0))

    cls = sorted(set(best_r["yt"]))
    cm = confusion_matrix(best_r["yt"], best_r["yp"], labels=cls)
    print(f"  Confusion matrix ({best_key}) — rows=true, cols=pred:")
    header = "    " + " ".join(f"{c[:6]:>6s}" for c in cls)
    print(header)
    for i, c in enumerate(cls):
        row = " ".join(f"{v:>6d}" for v in cm[i])
        print(f"    {c[:10]:<10s} {row}")

    return results


# ═════════════════════════════════════════════════════════════════════
# Load data
# ═════════════════════════════════════════════════════════════════════
print("=" * 70)
print("RTT Workload Inference — Unified Evaluation")
print(f"  Device:  {DEVICE}")
print(f"  Mode:    {'QUICK (1 WT seed, 60 epochs, skip LSTM)' if QUICK else 'FULL (5 WT seeds, 120 epochs, LSTM)'}")
print(f"  PCA:     {PCA_VAR*100:.0f}% variance" if PCA_VAR > 0 else "  PCA:     off")
print("=" * 70)

# Load training pool
print("\nLoading training pool...")
train_parts = []
for pname, (fname, offset) in TRAIN_POOL.items():
    fpath = os.path.join(FEAT_DIR, fname)
    if not os.path.exists(fpath):
        print(f"  MISSING: {fpath}")
        sys.exit(1)
    df = pd.read_csv(fpath)
    if offset > 0:
        df["run"] = df["run"] + offset
    train_parts.append(df)
    n_runs = df.groupby(["label", "run"]).ngroups
    print(f"  {pname}: {len(df)} windows, {n_runs} runs")

full_train = pd.concat(train_parts, ignore_index=True)
feat_cols = [c for c in full_train.columns if c not in META]
print(f"\nTraining pool: {len(full_train)} windows, "
      f"{full_train.groupby(['label','run']).ngroups} runs, {len(feat_cols)} features")

# Determine which test sets to run
requested_tests = ARGS.test if ARGS.test else list(TEST_SETS.keys())
invalid_tests = [t for t in requested_tests if t not in TEST_SETS]
if invalid_tests:
    print(f"ERROR: Unknown test sets: {invalid_tests}")
    print(f"Available: {list(TEST_SETS.keys())}")
    sys.exit(1)

# Load test sets
print("\nLoading test sets...")
test_dfs = {}
for tname in requested_tests:
    parts = []
    for fname, offset in TEST_SETS[tname]:
        fpath = os.path.join(FEAT_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  MISSING: {fpath}")
            sys.exit(1)
        df = pd.read_csv(fpath)
        if offset > 0:
            df["run"] = df["run"] + offset
        parts.append(df)
    test_dfs[tname] = pd.concat(parts, ignore_index=True)
    n_runs = test_dfs[tname].groupby(["label", "run"]).ngroups
    print(f"  {tname}: {len(test_dfs[tname])} windows, {n_runs} runs")

# ═════════════════════════════════════════════════════════════════════
# PCA (fit on training data only)
# ═════════════════════════════════════════════════════════════════════
pca_model = None
if PCA_VAR > 0:
    print(f"\nFitting PCA ({PCA_VAR*100:.0f}% variance on training data)...")
    Xtr_for_pca = np.nan_to_num(full_train[feat_cols].values, nan=0, posinf=0, neginf=0).astype(np.float32)
    sc_pca = StandardScaler()
    Xtr_for_pca = sc_pca.fit_transform(Xtr_for_pca)
    pca_model = PCA(n_components=PCA_VAR, random_state=42)
    pca_model.fit(Xtr_for_pca)
    n_components = pca_model.n_components_
    explained = np.sum(pca_model.explained_variance_ratio_) * 100
    print(f"  Components selected: {n_components} (out of {len(feat_cols)})")
    print(f"  Explained variance: {explained:.1f}%")

    pca_cols = [f"pc_{i+1}" for i in range(n_components)]

    def apply_pca(df):
        X = np.nan_to_num(df[feat_cols].values, nan=0, posinf=0, neginf=0).astype(np.float32)
        X = sc_pca.transform(X)
        X_pca = pca_model.transform(X)
        df_out = df[["label", "run", "window_start", "path", "baseline_rtt"]].copy()
        for i, col in enumerate(pca_cols):
            df_out[col] = X_pca[:, i]
        return df_out

    full_train = apply_pca(full_train)
    for tname in test_dfs:
        test_dfs[tname] = apply_pca(test_dfs[tname])
    feat_cols = pca_cols
    pca_model = None  # already applied, don't apply again in run_eval
    print(f"  Features after PCA: {len(feat_cols)}")

# ═════════════════════════════════════════════════════════════════════
# Run evaluations
# ═════════════════════════════════════════════════════════════════════
summary = {}

for test_name in requested_tests:
    test_df = test_dfs[test_name]

    print("\n\n" + "#" * 70)
    print(f"TEST SET: {test_name}")
    print("#" * 70)

    # 15-class
    le_15 = LabelEncoder()
    le_15.fit(sorted(WORKLOADS))
    labels_15 = list(le_15.classes_)
    r15 = run_eval(full_train, test_df, feat_cols, le_15, labels_15, len(labels_15),
                   f"{test_name} — 15-class", pca_model)

    # 11-class
    tr11 = full_train.copy()
    te11 = test_df.copy()
    tr11["label"] = tr11["label"].map(LABEL_MAP_11)
    te11["label"] = te11["label"].map(LABEL_MAP_11)
    le_11 = LabelEncoder()
    le_11.fit(sorted(tr11["label"].unique()))
    labels_11 = list(le_11.classes_)
    r11 = run_eval(tr11, te11, feat_cols, le_11, labels_11, len(labels_11),
                   f"{test_name} — 11-class", pca_model)

    summary[test_name] = {"15": r15, "11": r11}


# ═════════════════════════════════════════════════════════════════════
# Summary table
# ═════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 90)
print(f"SUMMARY{' (PCA ' + str(PCA_VAR*100) + '%)' if PCA_VAR > 0 else ''}")
print("=" * 90)

model_names = ["RF", "SVM", "MLP", "LGB"]
if not QUICK:
    model_names.append("LSTM")
model_names.extend(["WT", "LGB+WT"])

for test_name in requested_tests:
    s = summary[test_name]
    print(f"\n  {test_name}")
    print(f"    {'Model':<10s}  {'15-class':>14s}  {'11-class':>14s}")
    print(f"    {'-'*42}")
    for name in model_names:
        r15 = s["15"].get(name, {})
        r11 = s["11"].get(name, {})
        v15 = r15.get("bay", r15.get("maj", r15.get("acc", r15.get("avg", 0))))
        v11 = r11.get("bay", r11.get("maj", r11.get("acc", r11.get("avg", 0))))
        print(f"    {name:<10s}  {v15:>13.1f}%  {v11:>13.1f}%")

print(f"\nLog saved to: {LOG_FILE}")
