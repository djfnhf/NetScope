# Benchmark Reproducibility Guide

This folder contains dataset preprocessing and explanation scripts for traffic benchmarks.

## Goals for GitHub release

- No machine-specific absolute paths in runnable scripts
- English-only comments and user-facing messages
- Deterministic behavior where random sampling is used
- Clear expected input/output directory layout

## Expected project layout

From repository root:

- `datasets/<dataset_name>/...`
- `outputs_tsv/...` and `outputs_png/...` (generated)
- `exp/...` (generated artifacts)
- model repositories as sibling folders when required:
  - `ET-BERT`
  - `TrafficFormer`
  - `NetGPT`
  - `YaTC`

## Script notes

- `pcap2tsv.py`, `pcap2png.py`, `explain_tsv.py`, `explain_png.py`
  - Resolve default paths from project root.
  - Edit only the config block at top when changing dataset/model targets.
- `gen.py`
  - `TRAIN_ROOTS` supports relative paths from project root.
- `robust.py`
  - `test_root` and `out_root` support relative paths from project root.
- `build/`
  - End-to-end dataset builder with flow/packet stages.

## Reproducibility checklist

1. Keep `global_seed`/`seed` fixed in configs.
2. Keep input file ordering deterministic (already sorted in scripts).
3. Record exact command line and config values for each run.
4. Pin Python package versions in your release environment (recommended).

## Minimal pre-release validation

Run from repo root:

```bash
python3 -m py_compile benchmark/*.py benchmark/build/**/*.py
bash -n inference_benchmark/run_*.sh
```

Then run one smoke test per major path with a tiny subset:

- `pcap2tsv.py`
- `pcap2png.py`
- `explain_tsv.py` or `explain_png.py`
- `build/main.py` (optional if you use the build pipeline)

## Known external dependencies

- Python 3.8+
- `scapy`, `numpy`, `Pillow`, `matplotlib`, `torch`, `torchvision`, `tqdm`
- `SplitCap.exe` (for `build/` pipeline; Linux users typically run via Mono)
