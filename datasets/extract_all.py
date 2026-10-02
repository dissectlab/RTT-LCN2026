#!/usr/bin/env python3
"""
Unified feature extraction — runs the extractor over every path.

Extracts 104 percentile-based features (60s windows, 10s stride) from
all training and test paths in the dataset.

Paths:
  Phase 1:  A (victim, 20 runs), B (profile, 20 runs)
  Phase 2:  A2 (victim, 5 runs), B2 (profile, 5 runs)
  Phase 3:  C, Ep, F (5 runs each)
  Phase 5:  P5p (profile only, 5 runs)
  Phase 6:  A6v1, A6v2 (victim, 3 runs each, parameter variations)

Output: features_allpaths/{path}_features_fixed.csv
Log:    extract_all_log.txt

Usage:
  python3 extract_all.py              # extract all paths
  python3 extract_all.py A6v1 A6v2    # extract specific paths only
"""
import os, sys
import numpy as np, pandas as pd

LOG_FILE = "extract_all_log.txt"

class Tee:
    def __init__(self, path):
        self.file = open(path, "w"); self.stdout = sys.stdout
    def write(self, data):
        self.stdout.write(data); self.stdout.flush(); self.file.write(data); self.file.flush()
    def flush(self): self.stdout.flush(); self.file.flush()

sys.stdout = Tee(LOG_FILE)

WINDOW_SEC = 60
STRIDE_SEC = 10

from extract_features_v3_fixed import extract_window_features, load_trace, normalize_rtt

WORKLOADS = ['allreduce', 'bulk', 'checkpoint', 'db_queries', 'dfs_reads',
             'federated', 'inference', 'log_agg', 'microservice', 'migration',
             'pipeline', 'preprocess', 'scatter_gather', 'shuffle', 'storage_rep']

# ── All path configurations ─────────────────────────────────────────
PATH_CONFIGS = {
    # Phase 1
    "A":    ("raw_rtt_phase1", "victim_{wl}_run{run}_rtt.json",       range(1, 21)),
    "B":    ("raw_rtt_phase1", "profile_{wl}_run{run}_rtt.json",      range(1, 21)),
    # Phase 2
    "A2":   ("raw_rtt_phase2", "victim_{wl}_run{run}_p2_rtt.json",    range(1, 6)),
    "B2":   ("raw_rtt_phase2", "profile_{wl}_run{run}_p2_rtt.json",   range(1, 6)),
    # Phase 3
    "C":    ("raw_rtt_phase3", "pathC_{wl}_run{run}_rtt.json",        range(1, 6)),
    "Ep":   ("raw_rtt_phase3", "pathEp_{wl}_run{run}_rtt.json",       range(1, 6)),
    "F":    ("raw_rtt_phase3", "pathF_{wl}_run{run}_rtt.json",        range(1, 6)),
    # Phase 5 (profile only — victim dropped)
    "P5p":  ("raw_rtt_phase5", "profile_{wl}_run{run}_p5_rtt.json",   range(1, 6)),
    # Phase 6 (parameter variations, victim only)
    "A6v1": ("raw_rtt_phase6", "victim_{wl}_v1_run{run}_p6_rtt.json", range(1, 4)),
    "A6v2": ("raw_rtt_phase6", "victim_{wl}_v2_run{run}_p6_rtt.json", range(1, 4)),
}


def process_trace(filepath, path_label, workload, run):
    ts, rtt = load_trace(filepath)
    normed, baseline = normalize_rtt(rtt)
    # Drop NaN samples (probe timeouts: rtt_ms = -1 → NaN after normalize)
    valid_mask = ~np.isnan(normed)
    n_dropped = int((~valid_mask).sum())
    ts_v = ts[valid_mask]
    normed_v = normed[valid_mask]
    if len(normed_v) < 20:
        return [], n_dropped

    diffs = np.diff(ts_v)
    diffs = diffs[diffs > 0]
    sample_rate = 1.0 / np.median(diffs) if len(diffs) else 10.0

    t0 = ts_v[0]
    duration = ts_v[-1] - t0
    rows = []
    win_start = 0.0
    while win_start + WINDOW_SEC <= duration + 0.5:
        mask = (ts_v >= t0 + win_start) & (ts_v < t0 + win_start + WINDOW_SEC)
        window_normed = normed_v[mask]
        if len(window_normed) < 20:
            win_start += STRIDE_SEC
            continue
        feats = extract_window_features(window_normed, sample_rate)
        if feats is None:
            win_start += STRIDE_SEC
            continue
        feats["label"] = workload
        feats["run"] = run
        feats["window_start"] = win_start
        feats["path"] = path_label
        feats["baseline_rtt"] = baseline
        rows.append(feats)
        win_start += STRIDE_SEC
    return rows, n_dropped


# ── Determine which paths to extract ────────────────────────────────
requested = sys.argv[1:] if len(sys.argv) > 1 else list(PATH_CONFIGS.keys())
# Filter out the --quick/etc flags if accidentally passed
requested = [p for p in requested if not p.startswith("-")]
invalid = [p for p in requested if p not in PATH_CONFIGS]
if invalid:
    print(f"ERROR: Unknown paths: {invalid}")
    print(f"Available: {list(PATH_CONFIGS.keys())}")
    sys.exit(1)

print("=" * 70)
print("Unified Feature Extraction (104 features)")
print(f"  Paths: {', '.join(requested)}")
print("=" * 70)

OUT_DIR = "features_allpaths"
os.makedirs(OUT_DIR, exist_ok=True)

for path_name in requested:
    raw_dir, fname_template, runs = PATH_CONFIGS[path_name]
    print(f"\n--- Path {path_name} ---")

    rows = []
    missing = 0
    total_dropped = 0
    for wl in WORKLOADS:
        wl_count = 0
        wl_dropped = 0
        for run in runs:
            fname = fname_template.format(wl=wl, run=run)
            fpath = os.path.join(raw_dir, fname)
            if not os.path.exists(fpath):
                missing += 1
                continue
            r, dropped = process_trace(fpath, path_name, wl, run)
            rows.extend(r)
            wl_count += len(r)
            wl_dropped += dropped
        total_dropped += wl_dropped
        tag = f" [{wl_dropped} NaN samples dropped]" if wl_dropped > 0 else ""
        print(f"  {wl:<16s}: {wl_count} windows{tag}")

    df = pd.DataFrame(rows)
    meta = ["label", "run", "window_start", "path", "baseline_rtt"]
    feat_cols = [c for c in df.columns if c not in meta]
    n_runs = df.groupby(["label", "run"]).ngroups

    out_file = os.path.join(OUT_DIR, f"{path_name}_features_fixed.csv")
    df.to_csv(out_file, index=False)

    print(f"  -> {len(df)} windows total, {n_runs} runs, {len(feat_cols)} features")
    print(f"  -> {total_dropped} total NaN samples dropped, {missing} missing files")
    print(f"  -> {out_file}")

print(f"\nLog saved to: {LOG_FILE}")
