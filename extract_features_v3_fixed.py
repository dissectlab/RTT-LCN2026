#!/usr/bin/env python3
"""
v3-fixed: Path-independent feature extraction.

Fixes two path-dependent issues in v3:
1. Run-length thresholds: replaced max(hardcoded, k*std) with percentile-based
2. Spike detection: removed hardcoded 0.005 floor

Also fixes np.math deprecation warning.

All 104 features remain identical in name and count — only threshold
computation changes for 28 features (24 run-length + 4 spike/elevation).
"""

import json
import os
import math
import numpy as np
import pandas as pd
from scipy import signal, stats
from glob import glob

WINDOW_SEC = 60
STRIDE_SEC = 10
RAW_DIR = "raw_rtt"


def load_trace(filepath):
    with open(filepath) as f:
        data = json.load(f)
    ts = np.array([d["send_ts"] for d in data])
    rtt = np.array([d["rtt_ms"] for d in data])
    return ts, rtt


def normalize_rtt(rtt):
    valid = rtt[rtt > 0]
    if len(valid) < 10:
        return np.zeros_like(rtt, dtype=float), 0
    baseline = np.median(valid)
    normed = np.where(rtt > 0, (rtt - baseline) / baseline, np.nan)
    return normed, baseline


def approx_entropy(x, m=2, r_frac=0.2):
    N = len(x)
    if N < 30:
        return 0
    r = r_frac * np.std(x)
    if r == 0:
        return 0
    x = x[:min(N, 300)]
    N = len(x)

    def phi(m):
        patterns = np.array([x[i:i+m] for i in range(N - m + 1)])
        counts = np.zeros(len(patterns))
        for i in range(len(patterns)):
            counts[i] = np.sum(np.max(np.abs(patterns - patterns[i]), axis=1) <= r)
        return np.mean(np.log(counts / (N - m + 1)))

    return abs(phi(m) - phi(m + 1))


def sample_entropy(x, m=2, r_frac=0.2):
    N = len(x)
    if N < 30:
        return 0
    r = r_frac * np.std(x)
    if r == 0:
        return 0
    x = x[:min(N, 300)]
    N = len(x)

    def count_matches(m):
        patterns = np.array([x[i:i+m] for i in range(N - m)])
        count = 0
        for i in range(len(patterns)):
            dists = np.max(np.abs(patterns[i+1:] - patterns[i]), axis=1)
            count += np.sum(dists <= r)
        return count

    A = count_matches(m + 1)
    B = count_matches(m)
    if B == 0:
        return 0
    return -np.log(A / B) if A > 0 else 0


def permutation_entropy(x, order=3, delay=1):
    N = len(x)
    if N < order * delay + 1:
        return 1.0

    n_patterns = N - (order - 1) * delay
    perms = []
    for i in range(n_patterns):
        indices = [i + j * delay for j in range(order)]
        vals = [x[idx] for idx in indices]
        perms.append(tuple(np.argsort(vals)))

    from collections import Counter
    counts = Counter(perms)
    total = sum(counts.values())
    probs = np.array(list(counts.values())) / total
    ent = -np.sum(probs * np.log2(probs))
    max_ent = np.log2(math.factorial(order))  # fixed: was np.math.factorial
    return ent / max_ent if max_ent > 0 else 1.0


def hurst_exponent(x):
    N = len(x)
    if N < 20:
        return 0.5
    max_k = min(int(np.log2(N)), 7)
    rs_list, ns = [], []
    for k in range(2, max_k + 1):
        size = N // (2 ** k)
        if size < 4:
            break
        rs_vals = []
        for start in range(0, N - size + 1, size):
            chunk = x[start:start + size]
            mean_c = np.mean(chunk)
            cumdev = np.cumsum(chunk - mean_c)
            R = np.max(cumdev) - np.min(cumdev)
            S = np.std(chunk)
            if S > 1e-15:
                rs_vals.append(R / S)
        if rs_vals:
            rs_list.append(np.log(np.mean(rs_vals)))
            ns.append(np.log(size))
    if len(ns) >= 2:
        return np.polyfit(ns, rs_list, 1)[0]
    return 0.5


def wavelet_features(x, max_level=4):
    features = {}
    signal_in = x.copy()
    total_energy = np.sum(x ** 2) + 1e-15

    for level in range(1, max_level + 1):
        n = len(signal_in)
        if n < 4:
            break
        n = n - (n % 2)
        signal_in = signal_in[:n]
        approx = (signal_in[0::2] + signal_in[1::2]) / np.sqrt(2)
        detail = (signal_in[0::2] - signal_in[1::2]) / np.sqrt(2)
        features[f"wavelet_energy_d{level}"] = np.sum(detail ** 2) / total_energy
        features[f"wavelet_std_d{level}"] = np.std(detail)
        features[f"wavelet_max_d{level}"] = np.max(np.abs(detail))
        signal_in = approx

    if len(signal_in) > 1:
        features["wavelet_energy_approx"] = np.sum(signal_in ** 2) / total_energy
        features["wavelet_std_approx"] = np.std(signal_in)

    return features


def run_length_features(x, thresholds):
    features = {}
    for thresh_name, thresh_val in thresholds:
        above = (x > thresh_val).astype(int)
        transitions = np.diff(above)

        starts = np.where(transitions == 1)[0]
        ends = np.where(transitions == -1)[0]
        if len(starts) > 0 and len(ends) > 0:
            if ends[0] < starts[0]:
                ends = ends[1:]
            min_len = min(len(starts), len(ends))
            if min_len > 0:
                run_lens = ends[:min_len] - starts[:min_len]
                features[f"rl_{thresh_name}_max"] = np.max(run_lens)
                features[f"rl_{thresh_name}_mean"] = np.mean(run_lens)
                features[f"rl_{thresh_name}_count"] = len(run_lens)
                if len(run_lens) > 1:
                    features[f"rl_{thresh_name}_cv"] = np.std(run_lens) / (np.mean(run_lens) + 1e-10)
                else:
                    features[f"rl_{thresh_name}_cv"] = 0
            else:
                for suffix in ["max", "mean", "count", "cv"]:
                    features[f"rl_{thresh_name}_{suffix}"] = 0
        else:
            for suffix in ["max", "mean", "count", "cv"]:
                features[f"rl_{thresh_name}_{suffix}"] = 0

        below = 1 - above
        trans_b = np.diff(below)
        starts_b = np.where(trans_b == 1)[0]
        ends_b = np.where(trans_b == -1)[0]
        if len(starts_b) > 0 and len(ends_b) > 0:
            if ends_b[0] < starts_b[0]:
                ends_b = ends_b[1:]
            min_len = min(len(starts_b), len(ends_b))
            if min_len > 0:
                quiet_lens = ends_b[:min_len] - starts_b[:min_len]
                features[f"rl_{thresh_name}_quiet_max"] = np.max(quiet_lens)
                features[f"rl_{thresh_name}_quiet_mean"] = np.mean(quiet_lens)
            else:
                features[f"rl_{thresh_name}_quiet_max"] = 0
                features[f"rl_{thresh_name}_quiet_mean"] = 0
        else:
            features[f"rl_{thresh_name}_quiet_max"] = len(x) if np.all(below) else 0
            features[f"rl_{thresh_name}_quiet_mean"] = len(x) if np.all(below) else 0

    return features


def extract_window_features(normed_rtt, sample_rate):
    """Extract all features from a normalized RTT window.

    v3-fixed: all thresholds are percentile-based or std-relative.
    No hardcoded absolute values that could create path dependence.
    """
    valid = normed_rtt[~np.isnan(normed_rtt)]
    n = len(valid)
    n_total = len(normed_rtt)

    if n < 20:
        return None

    features = {}

    # Timeout
    features["timeout_frac"] = 1.0 - n / n_total if n_total > 0 else 0

    # Basic shape
    features["cv"] = np.std(valid) / (np.mean(np.abs(valid)) + 1e-10)
    features["iqr"] = np.percentile(valid, 75) - np.percentile(valid, 25)
    features["skew"] = stats.skew(valid)
    features["kurtosis"] = stats.kurtosis(valid)
    features["range"] = np.ptp(valid)
    features["p95_p50_ratio"] = np.percentile(valid, 95) - np.percentile(valid, 50)
    features["p99_p50_ratio"] = np.percentile(valid, 99) - np.percentile(valid, 50)

    # =========================================================================
    # FIX 1: Spike/elevation — removed hardcoded 0.005 floor
    # Was:    elevated = valid > max(spike_thresh, 0.005)
    # Now:    elevated = valid > spike_thresh  (purely percentile-based)
    # =========================================================================
    spike_thresh = np.percentile(valid, 90)
    elevated = valid > spike_thresh
    features["elevated_frac"] = np.mean(elevated)
    features["mean_spike_amp"] = np.mean(valid[elevated]) if np.any(elevated) else 0
    features["max_spike_amp"] = np.max(valid)

    # Burst analysis (uses the fixed elevated mask)
    elevated_int = elevated.astype(int)
    transitions = np.diff(elevated_int)
    burst_starts = np.where(transitions == 1)[0]
    burst_ends = np.where(transitions == -1)[0]

    if len(burst_starts) > 0 and len(burst_ends) > 0:
        if burst_ends[0] < burst_starts[0]:
            burst_ends = burst_ends[1:]
        min_len = min(len(burst_starts), len(burst_ends))
        burst_starts = burst_starts[:min_len]
        burst_ends = burst_ends[:min_len]
        burst_lengths = burst_ends - burst_starts

        if len(burst_lengths) > 1:
            features["burst_count"] = len(burst_lengths)
            features["mean_burst_len"] = np.mean(burst_lengths) / sample_rate
            features["std_burst_len"] = np.std(burst_lengths) / sample_rate
            features["cv_burst_len"] = np.std(burst_lengths) / (np.mean(burst_lengths) + 1e-10)
            if len(burst_starts) > 1:
                ibi = np.diff(burst_starts) / sample_rate
                features["mean_ibi"] = np.mean(ibi)
                features["cv_ibi"] = np.std(ibi) / (np.mean(ibi) + 1e-10)
            else:
                features["mean_ibi"] = WINDOW_SEC
                features["cv_ibi"] = 0
        else:
            features["burst_count"] = 1 if len(burst_lengths) == 1 else 0
            features["mean_burst_len"] = burst_lengths[0] / sample_rate if len(burst_lengths) == 1 else 0
            features["std_burst_len"] = 0
            features["cv_burst_len"] = 0
            features["mean_ibi"] = WINDOW_SEC
            features["cv_ibi"] = 0
    else:
        for k in ["burst_count", "mean_burst_len", "std_burst_len", "cv_burst_len",
                   "mean_ibi", "cv_ibi"]:
            features[k] = 0

    features["burst_rate"] = features["burst_count"] / WINDOW_SEC

    # Multi-scale variability
    for scale_name, n_chunks in [("fine", 20), ("mid", 6), ("coarse", 2)]:
        chunk_size = max(1, n // n_chunks)
        chunks = [valid[i:i+chunk_size] for i in range(0, n - chunk_size + 1, chunk_size)]
        if len(chunks) >= 2:
            chunk_means = [np.mean(c) for c in chunks]
            chunk_stds = [np.std(c) for c in chunks]
            features[f"inter_chunk_cv_{scale_name}"] = np.std(chunk_means) / (np.mean(np.abs(chunk_means)) + 1e-10)
            features[f"mean_intra_chunk_cv_{scale_name}"] = np.mean(chunk_stds) / (np.std(valid) + 1e-10)
        else:
            features[f"inter_chunk_cv_{scale_name}"] = 0
            features[f"mean_intra_chunk_cv_{scale_name}"] = 0

    # Temporal dynamics
    diffs = np.diff(valid)
    features["mean_abs_diff"] = np.mean(np.abs(diffs))
    features["diff_skew"] = stats.skew(diffs)
    features["diff_kurtosis"] = stats.kurtosis(diffs)

    # Autocorrelation
    for lag_sec in [0.5, 1, 2, 5, 10, 20, 30]:
        lag = int(lag_sec * sample_rate)
        if lag < n - 1:
            ac = np.corrcoef(valid[:-lag], valid[lag:])[0, 1]
            features[f"ac_lag_{lag_sec}s"] = ac if np.isfinite(ac) else 0
        else:
            features[f"ac_lag_{lag_sec}s"] = 0

    # Frequency domain
    if n >= 32:
        detrended = signal.detrend(valid)
        windowed = detrended * signal.windows.hann(len(detrended))
        fft_vals = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(len(windowed), d=1.0 / sample_rate)
        fft_vals = fft_vals[1:]
        freqs = freqs[1:]

        if len(fft_vals) > 0 and np.sum(fft_vals) > 0:
            fft_norm = fft_vals / np.sum(fft_vals)
            fft_norm_pos = fft_norm[fft_norm > 0]
            features["spectral_entropy"] = -np.sum(fft_norm_pos * np.log2(fft_norm_pos)) / np.log2(len(fft_norm))

            peak_idx = np.argmax(fft_vals)
            features["dominant_freq"] = freqs[peak_idx]
            features["dominant_freq_power"] = fft_norm[peak_idx]

            top3_idx = np.argsort(fft_vals)[-3:]
            features["top3_freq_power"] = np.sum(fft_norm[top3_idx])

            for band_name, (f_low, f_high) in [
                ("very_low", (0.001, 0.05)),
                ("low", (0.05, 0.2)),
                ("mid", (0.2, 1.0)),
                ("high", (1.0, 3.0)),
            ]:
                band_mask = (freqs >= f_low) & (freqs < f_high)
                features[f"band_power_{band_name}"] = np.sum(fft_norm[band_mask])

            features["spectral_centroid"] = np.sum(freqs * fft_norm) / (np.sum(fft_norm) + 1e-10)
            cumsum = np.cumsum(fft_norm)
            rolloff_idx = np.searchsorted(cumsum, 0.85)
            features["spectral_rolloff"] = freqs[min(rolloff_idx, len(freqs) - 1)]
        else:
            for key in ["spectral_entropy", "dominant_freq", "dominant_freq_power",
                        "top3_freq_power", "band_power_very_low", "band_power_low",
                        "band_power_mid", "band_power_high", "spectral_centroid",
                        "spectral_rolloff"]:
                features[key] = 0

    # Energy asymmetry
    half = n // 2
    e1, e2 = np.sum(valid[:half] ** 2), np.sum(valid[half:] ** 2)
    features["energy_asymmetry"] = (e2 - e1) / (e1 + e2 + 1e-15)

    # Entropy / Complexity
    features["approx_entropy"] = approx_entropy(valid)
    features["sample_entropy"] = sample_entropy(valid)
    features["perm_entropy_d1"] = permutation_entropy(valid, order=3, delay=1)
    features["perm_entropy_d2"] = permutation_entropy(valid, order=3, delay=2)
    features["perm_entropy_d5"] = permutation_entropy(valid, order=3, delay=5)

    # Hurst exponent
    features["hurst"] = hurst_exponent(valid)

    # Wavelet features
    features.update(wavelet_features(valid, max_level=4))

    # =========================================================================
    # FIX 2: Run-length thresholds — percentile-based instead of hardcoded
    # Was:    ("tiny",  max(0.0005, 0.5 * std_val))
    #         ("small", max(0.001,  1.0 * std_val))
    #         ("med",   max(0.005,  2.0 * std_val))
    #         ("large", max(0.01,   3.0 * std_val))
    # Now:    percentile-based on |valid|, always relative to signal distribution
    # =========================================================================
    abs_valid = np.abs(valid)
    thresholds = [
        ("tiny",  np.percentile(abs_valid, 60)),   # ~0.5 sigma equivalent
        ("small", np.percentile(abs_valid, 75)),   # ~1.0 sigma equivalent
        ("med",   np.percentile(abs_valid, 90)),   # ~2.0 sigma equivalent
        ("large", np.percentile(abs_valid, 97)),   # ~3.0 sigma equivalent
    ]
    features.update(run_length_features(valid, thresholds))

    # Distribution shape (histogram-based) — already path-independent (z-scored)
    std_val = np.std(valid)
    if std_val > 1e-10:
        z = valid / std_val
    else:
        z = valid * 0
    for bin_name, (lo, hi) in [
        ("near_zero", (-0.5, 0.5)),
        ("moderate_pos", (0.5, 2.0)),
        ("moderate_neg", (-2.0, -0.5)),
        ("large_pos", (2.0, 5.0)),
        ("extreme_pos", (5.0, np.inf)),
    ]:
        features[f"bin_{bin_name}"] = np.mean((z >= lo) & (z < hi))

    # Ratio features between scales
    fine_chunk = max(1, n // 20)
    coarse_chunk = max(1, n // 4)
    fine_vars = [np.var(valid[i:i+fine_chunk]) for i in range(0, n - fine_chunk + 1, fine_chunk)]
    coarse_vars = [np.var(valid[i:i+coarse_chunk]) for i in range(0, n - coarse_chunk + 1, coarse_chunk)]
    if len(fine_vars) > 1 and len(coarse_vars) > 1:
        features["var_ratio_fine_coarse"] = np.mean(fine_vars) / (np.mean(coarse_vars) + 1e-15)
        features["var_cv_fine"] = np.std(fine_vars) / (np.mean(fine_vars) + 1e-15)
        features["var_cv_coarse"] = np.std(coarse_vars) / (np.mean(coarse_vars) + 1e-15)
    else:
        features["var_ratio_fine_coarse"] = 1.0
        features["var_cv_fine"] = 0
        features["var_cv_coarse"] = 0

    # Trend
    t = np.arange(n, dtype=float) / n
    slope = np.polyfit(t, valid, 1)[0]
    features["trend_slope"] = slope / (std_val + 1e-15)

    # Energy quarters
    q = n // 4
    for i, qname in enumerate(["q1", "q2", "q3", "q4"]):
        features[f"energy_{qname}"] = np.sum(valid[i*q:(i+1)*q] ** 2) / (e1 + e2 + 1e-15)

    # Autocorrelation decay
    ac_vals = []
    for lag in range(1, min(n // 2, 50)):
        ac = np.corrcoef(valid[:-lag], valid[lag:])[0, 1]
        if np.isfinite(ac):
            ac_vals.append(ac)
        else:
            ac_vals.append(0)
    if len(ac_vals) >= 5:
        ac_arr = np.array(ac_vals)
        features["ac_decay_rate"] = ac_arr[0] - ac_arr[min(4, len(ac_arr)-1)]
        half_ac = ac_arr[0] * 0.5
        below_half = np.where(ac_arr < half_ac)[0]
        features["ac_half_life"] = below_half[0] / sample_rate if len(below_half) > 0 else len(ac_arr) / sample_rate
    else:
        features["ac_decay_rate"] = 0
        features["ac_half_life"] = 0

    return features
