#!/usr/bin/env python3
"""
Train all models on the full training pool and export artifacts for deployment.

Output: deployed_models/
  - scaler_pca.joblib, pca_model.joblib, scaler_clf.joblib
  - rf_model.joblib, lgb_model.joblib, mlp_model.joblib, svm_model.joblib
  - wt_seed{42,123,456,789,999}.pt
  - lstm_seed{42,123,456}.pt
  - label_encoder_15.joblib, label_encoder_11.joblib

Usage:
  python3 export_models.py
"""
import os, sys
import numpy as np, pandas as pd, joblib, lightgbm as lgb
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler

LOG_FILE = "export_models_log.txt"

class Tee:
    def __init__(self, path):
        self.file = open(path, "w"); self.stdout = sys.stdout
    def write(self, data):
        self.stdout.write(data); self.stdout.flush(); self.file.write(data); self.file.flush()
    def flush(self): self.stdout.flush(); self.file.flush()

sys.stdout = Tee(LOG_FILE)

META = ["label", "run", "window_start", "path", "baseline_rtt"]
FEAT_DIR = "features_allpaths"
OUT_DIR = "deployed_models"
os.makedirs(OUT_DIR, exist_ok=True)

WORKLOADS = ['allreduce', 'bulk', 'checkpoint', 'db_queries', 'dfs_reads',
             'federated', 'inference', 'log_agg', 'microservice', 'migration',
             'pipeline', 'preprocess', 'scatter_gather', 'shuffle', 'storage_rep']

LABEL_MAP_11 = {wl: wl for wl in WORKLOADS}
for wl in ['db_queries', 'inference', 'log_agg', 'microservice', 'shuffle']:
    LABEL_MAP_11[wl] = 'light_traffic'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_WINDOWS = 25

TRAIN_POOL = {
    "B":   ("B_features_fixed.csv",   0),
    "C":   ("C_features_fixed.csv",   100),
    "Ep":  ("Ep_features_fixed.csv",  200),
    "F":   ("F_features_fixed.csv",   300),
    "P5p": ("P5p_features_fixed.csv", 600),
    "B2":  ("B2_features_fixed.csv",  700),
}


# ── Models ─────────────────────────────────────────────────────────
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


def train_seq_model(model, tr_s, tr_l, epochs=120, lr=5e-4):
    tl = DataLoader(SeqDS(tr_s, tr_l), batch_size=32, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    for ep in range(1, epochs + 1):
        model.train()
        for X, y in tl:
            X, y = X.to(DEVICE), y.to(DEVICE)
            if np.random.random() < 0.5: X = X + torch.randn_like(X) * 0.03
            opt.zero_grad(); crit(model(X), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()
    return model


# ═════════════════════════════════════════════════════════════════════
# Load data
# ═════════════════════════════════════════════════════════════════════
print("=" * 70)
print("Model Export — Training on full pool, saving all artifacts")
print(f"  Device: {DEVICE}")
print("=" * 70)

print("\nLoading training pool...")
train_parts = []
for pname, (fname, offset) in TRAIN_POOL.items():
    df = pd.read_csv(os.path.join(FEAT_DIR, fname))
    if offset > 0:
        df["run"] = df["run"] + offset
    train_parts.append(df)
    print(f"  {pname}: {len(df)} windows")

full_train = pd.concat(train_parts, ignore_index=True)
feat_cols = [c for c in full_train.columns if c not in META]
print(f"Training pool: {len(full_train)} windows, {len(feat_cols)} features")

X_raw = np.nan_to_num(full_train[feat_cols].values, nan=0, posinf=0, neginf=0).astype(np.float32)


# ═════════════════════════════════════════════════════════════════════
# PCA
# ═════════════════════════════════════════════════════════════════════
print("\nFitting PCA...")
scaler_pca = StandardScaler()
X_scaled_for_pca = scaler_pca.fit_transform(X_raw)

pca = PCA(n_components=0.95, random_state=42)
pca.fit(X_scaled_for_pca)
X_pca = pca.transform(X_scaled_for_pca)
n_comp = pca.n_components_
print(f"  PCA: {len(feat_cols)} → {n_comp} components ({np.sum(pca.explained_variance_ratio_)*100:.1f}% variance)")

joblib.dump(scaler_pca, os.path.join(OUT_DIR, "scaler_pca.joblib"))
joblib.dump(pca, os.path.join(OUT_DIR, "pca_model.joblib"))
print("  Saved: scaler_pca.joblib, pca_model.joblib")


# ═════════════════════════════════════════════════════════════════════
# Classifier scaler (no CORAL during export — fit on training only)
# ═════════════════════════════════════════════════════════════════════
scaler_clf = StandardScaler()
X_final = scaler_clf.fit_transform(X_pca)
joblib.dump(scaler_clf, os.path.join(OUT_DIR, "scaler_clf.joblib"))
print("  Saved: scaler_clf.joblib")


# ═════════════════════════════════════════════════════════════════════
# Label encoders
# ═════════════════════════════════════════════════════════════════════
le_15 = LabelEncoder(); le_15.fit(sorted(WORKLOADS))
le_11 = LabelEncoder(); le_11.fit(sorted(set(LABEL_MAP_11.values())))
joblib.dump(le_15, os.path.join(OUT_DIR, "label_encoder_15.joblib"))
joblib.dump(le_11, os.path.join(OUT_DIR, "label_encoder_11.joblib"))
print("  Saved: label_encoder_15.joblib, label_encoder_11.joblib")

y_15 = le_15.transform(full_train["label"].values)

# Also prepare 11-class
train_11 = full_train.copy()
train_11["label"] = train_11["label"].map(LABEL_MAP_11)
y_11 = le_11.transform(train_11["label"].values)


# ═════════════════════════════════════════════════════════════════════
# Classical classifiers (15-class)
# ═════════════════════════════════════════════════════════════════════
print("\n--- Classical classifiers (15-class) ---")

print("  Training RF...")
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=4)
rf.fit(X_final, y_15)
joblib.dump(rf, os.path.join(OUT_DIR, "rf_model.joblib"))
print("  Saved: rf_model.joblib")

print("  Training SVM...")
svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
svm.fit(X_final, y_15)
joblib.dump(svm, os.path.join(OUT_DIR, "svm_model.joblib"))
print("  Saved: svm_model.joblib")

print("  Training MLP...")
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42, early_stopping=True)
mlp.fit(X_final, y_15)
joblib.dump(mlp, os.path.join(OUT_DIR, "mlp_model.joblib"))
print("  Saved: mlp_model.joblib")

print("  Training LGB...")
lgb_clf = lgb.LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                               random_state=42, n_jobs=4, verbose=-1)
lgb_clf.fit(X_final, y_15)
joblib.dump(lgb_clf, os.path.join(OUT_DIR, "lgb_model.joblib"))
print("  Saved: lgb_model.joblib")


# ═════════════════════════════════════════════════════════════════════
# Classical classifiers (11-class)
# ═════════════════════════════════════════════════════════════════════
print("\n--- Classical classifiers (11-class) ---")

print("  Training RF 11-class...")
rf_11 = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=4)
rf_11.fit(X_final, y_11)
joblib.dump(rf_11, os.path.join(OUT_DIR, "rf_model_11.joblib"))

print("  Training SVM 11-class...")
svm_11 = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
svm_11.fit(X_final, y_11)
joblib.dump(svm_11, os.path.join(OUT_DIR, "svm_model_11.joblib"))

print("  Training MLP 11-class...")
mlp_11 = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42, early_stopping=True)
mlp_11.fit(X_final, y_11)
joblib.dump(mlp_11, os.path.join(OUT_DIR, "mlp_model_11.joblib"))

print("  Training LGB 11-class...")
lgb_11 = lgb.LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                               random_state=42, n_jobs=4, verbose=-1)
lgb_11.fit(X_final, y_11)
joblib.dump(lgb_11, os.path.join(OUT_DIR, "lgb_model_11.joblib"))
print("  Saved: rf/svm/mlp/lgb_model_11.joblib")


# ═════════════════════════════════════════════════════════════════════
# Sequence models (15-class)
# ═════════════════════════════════════════════════════════════════════
print("\n--- Sequence models (15-class) ---")

tr_s, tr_l = build_seq(full_train, X_final, le_15, MAX_WINDOWS)
n_classes_15 = len(le_15.classes_)
n_feat = X_final.shape[1]
print(f"  Sequences: {tr_s.shape} (runs, windows, features)")

# Window Transformer — 5 seeds
for seed in [42, 123, 456, 789, 999]:
    print(f"  Training WT seed={seed}...")
    torch.manual_seed(seed); np.random.seed(seed)
    wt = WindowTransformer(n_classes_15, n_feat).to(DEVICE)
    wt = train_seq_model(wt, tr_s, tr_l, epochs=120, lr=5e-4)
    wt.eval()
    torch.save(wt.state_dict(), os.path.join(OUT_DIR, f"wt_seed{seed}.pt"))
    print(f"  Saved: wt_seed{seed}.pt")

# LSTM — 3 seeds
for seed in [42, 123, 456]:
    print(f"  Training LSTM seed={seed}...")
    torch.manual_seed(seed); np.random.seed(seed)
    lstm = LSTMClassifier(n_feat, 64, n_classes_15).to(DEVICE)
    lstm = train_seq_model(lstm, tr_s, tr_l, epochs=100, lr=1e-3)
    lstm.eval()
    torch.save(lstm.state_dict(), os.path.join(OUT_DIR, f"lstm_seed{seed}.pt"))
    print(f"  Saved: lstm_seed{seed}.pt")


# ═════════════════════════════════════════════════════════════════════
# Sequence models (11-class)
# ═════════════════════════════════════════════════════════════════════
print("\n--- Sequence models (11-class) ---")

tr_s_11, tr_l_11 = build_seq(train_11, X_final, le_11, MAX_WINDOWS)
n_classes_11 = len(le_11.classes_)

for seed in [42, 123, 456, 789, 999]:
    print(f"  Training WT-11 seed={seed}...")
    torch.manual_seed(seed); np.random.seed(seed)
    wt = WindowTransformer(n_classes_11, n_feat).to(DEVICE)
    wt = train_seq_model(wt, tr_s_11, tr_l_11, epochs=120, lr=5e-4)
    wt.eval()
    torch.save(wt.state_dict(), os.path.join(OUT_DIR, f"wt_11_seed{seed}.pt"))

for seed in [42, 123, 456]:
    print(f"  Training LSTM-11 seed={seed}...")
    torch.manual_seed(seed); np.random.seed(seed)
    lstm = LSTMClassifier(n_feat, 64, n_classes_11).to(DEVICE)
    lstm = train_seq_model(lstm, tr_s_11, tr_l_11, epochs=100, lr=1e-3)
    lstm.eval()
    torch.save(lstm.state_dict(), os.path.join(OUT_DIR, f"lstm_11_seed{seed}.pt"))

print("  Saved: wt_11_seed*.pt, lstm_11_seed*.pt")


# ═════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXPORT COMPLETE")
print("=" * 70)

files = sorted(os.listdir(OUT_DIR))
total_size = 0
for f in files:
    size = os.path.getsize(os.path.join(OUT_DIR, f))
    total_size += size
    print(f"  {f:<35s} {size/1024:>10.1f} KB")

print(f"\n  Total: {len(files)} files, {total_size/1024/1024:.1f} MB")
print(f"  Output dir: {OUT_DIR}/")
print(f"\n  Copy to observer node:")
print(f"    scp -r {OUT_DIR}/ extract_features_v3_fixed.py observer1:~/inference/")

print(f"\nLog saved to: {LOG_FILE}")
