# RTT-LCN2026

Code and datasets accompanying the paper:

> **Round Trip Time: A Benign Signal or an Indirect Window into Datacenter Workloads?**
> Sourya Saha, Md Nurul Absur, Saptarshi Debroy.
> *2026 IEEE 51st Conference on Local Computer Networks (LCN)*, pp. 1–9.
> DOI: [10.1109/LCN67947.2026.11660789](https://doi.org/10.1109/LCN67947.2026.11660789) · arXiv: [2607.26239](https://arxiv.org/abs/2607.26239)

This repository lets anyone reproduce the workload-inference results from raw
RTT probe traces: the full chain of **raw RTT → per-window features → model
training/evaluation → real-time inference**. All datasets used in the paper are
included.

---

## What the paper studies

In a shared leaf-spine datacenter fabric, an **observer** that can only measure
round-trip time (RTT) to a co-located endpoint can infer *which workload* a
neighboring tenant is running, purely from congestion-induced latency variation.
We treat this as a side-channel *vulnerability* and characterize when it is — and
is not — exploitable. Evaluation is cross-path: models are trained on profiling
paths and tested on held-out victim paths.

Headline results (cross-path, run-level accuracy): **73.7%** across the full
15-workload set, rising to **97.3%** over 11 classes once the five workloads whose
traffic falls below the observable congestion floor are grouped into a single
low-signal class.

---

## Repository layout

```
RTT-LCN2026/
├── README.md                     ← this file
├── LICENSE                       ← MIT (code)
├── requirements.txt
├── datasets/                     ← all RTT datasets + feature pipeline (see datasets/README.md)
│   ├── README.md                 ← full data + feature documentation
│   ├── LICENSE                   ← CC-BY-4.0 (data)
│   ├── raw_rtt_phase1..6/        ← raw RTT probe traces (JSON)
│   ├── features_allpaths/        ← extracted per-window feature CSVs
│   ├── extract_features_v3_fixed.py
│   └── extract_all.py
├── extract_features_v3_fixed.py  ← v3-fixed feature extractor (104 features, path-invariant)
├── extract_all.py                ← batch-extract features for all paths
├── evaluate.py                   ← train + cross-path evaluation (reproduces the accuracy results)
├── export_models.py              ← train on full pool and export deployable model artifacts
└── infer.py                      ← real-time observer-side inference from a raw RTT trace
```

The `datasets/` folder is self-contained and has its own detailed
[`README.md`](datasets/README.md) documenting the raw-trace format, path/workload
naming, per-phase contents, and the 104-feature catalogue.

---

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10+. A CUDA GPU is optional (used for the LSTM / Window
Transformer sequence models); everything runs on CPU otherwise.

---

## Reproducing the results

**1. (Optional) Rebuild the feature CSVs from raw RTT traces.**
The CSVs in `datasets/features_allpaths/` are already provided, but can be
regenerated from the raw JSON:

```bash
python3 extract_all.py          # rebuilds every CSV in datasets/features_allpaths/
```

**2. Cross-path evaluation (the paper's accuracy numbers).**

```bash
python3 evaluate.py --pca 0.95              # full evaluation, PCA at 95% variance
python3 evaluate.py --pca 0.95 --test A2 B2 # evaluate on specific held-out paths
python3 evaluate.py --quick                 # fast subset for a smoke test
```

`--pca 0.95` applies the production domain-adaptation pipeline (PCA to 95%
variance + CORAL covariance alignment) described in the paper.

**3. (Optional) Export models and run real-time inference (Table I).**
Trained model blobs are **not** shipped (they are large and fully regenerable).
Produce them locally, then run the observer-side inference path:

```bash
python3 export_models.py                             # writes deployed_models/
python3 infer.py deployed_models/ path/to/trace.json # classify a single raw RTT trace
```

---

## Datasets

See [`datasets/README.md`](datasets/README.md) for the complete description:
6 measurement phases on the FABRIC testbed (17 nodes, 4-spine/4-leaf Clos
topology), raw UDP-echo RTT probe traces (~10 Hz), 15 workload classes, and the
path/run mapping. The feature pipeline produces 104 path-invariant features over
60 s windows with a 10 s stride.

---

## Citation

If you use this code or the datasets, please cite the paper:

```bibtex
@INPROCEEDINGS{11660789,
  author={Saha, Sourya and Absur, Md Nurul and Debroy, Saptarshi},
  booktitle={2026 IEEE 51st Conference on Local Computer Networks (LCN)},
  title={Round Trip Time: A Benign Signal or an Indirect Window into Datacenter Workloads?},
  year={2026},
  pages={1-9},
  doi={10.1109/LCN67947.2026.11660789}}
```

Use of the datasets additionally requires attribution under CC-BY-4.0 (see
`datasets/LICENSE`).

---

## License

- **Code** (`*.py`): MIT — see [`LICENSE`](LICENSE).
- **Datasets** (`datasets/`): Creative Commons Attribution 4.0 (CC-BY-4.0) — see
  [`datasets/LICENSE`](datasets/LICENSE).
