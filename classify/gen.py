#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_train_subsets_multi_v2_localout.py
=======================================
Features:
  - Build subsets for train only, ignore valid/test.
  - Support multiple train roots (multiple datasets).
  - Treat first-level subdirectories as classes.
  - Support multiple sampling ratios and random seeds.
  - Use symlink by default, optional file copy mode.
  - Write outputs under each dataset root (parent of train).
"""


TRAIN_ROOTS = [
    "datasets/ISCX-VPN_app_flow/train"
]

PROPORTIONS = [1, 5, 10, 20, 50, 75]
SEEDS = [2]
COPY_FILES = False

# ========== Implementation ==========

import os, shutil, random
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_train_root(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def collect_class_files(train_root: Path) -> Dict[str, List[Path]]:
    cls_map = {}
    if not train_root.exists():
        raise SystemExit(f"[ERROR] Train root not found: {train_root}")
    for cls_dir in sorted(train_root.iterdir()):
        if cls_dir.is_dir():
            files = [f for f in cls_dir.iterdir() if f.is_file()]
            if files:
                cls_map[cls_dir.name] = sorted(files)
    if not cls_map:
        raise SystemExit(f"[ERROR] No valid classes found in {train_root}")
    return cls_map

def link_or_copy(src: Path, dst: Path, do_copy=False):
    ensure_dir(dst.parent)
    if dst.exists():
        return
    try:
        if do_copy:
            shutil.copy2(src, dst)
        else:
            rel = os.path.relpath(src, start=dst.parent)
            dst.symlink_to(rel)
    except Exception:
        shutil.copy2(src, dst)

def sample_one_dataset(train_classes: Dict[str, List[Path]],
                       out_base: Path, seed: int, proportions, do_copy=False):
    random.seed(seed)
    seed_dir = out_base / f"seed_{seed}"
    ensure_dir(seed_dir)

    for pct in proportions:
        prop_dir = seed_dir / f"prop_{int(pct)}_train"
        ensure_dir(prop_dir)

        total_count = 0
        per_class_counts = {}

        for cls, files in train_classes.items():
            cls_out = prop_dir / cls
            ensure_dir(cls_out)
            n_total = len(files)
            n_sample = int(round(n_total * pct / 100.0)) if pct > 0 else 0
            if pct > 0 and n_sample == 0:
                n_sample = 1
            sample_files = random.sample(files, n_sample) if n_sample > 0 else []
            for f in sample_files:
                dst = cls_out / f.name
                link_or_copy(f, dst, do_copy=do_copy)
            per_class_counts[cls] = len(sample_files)
            total_count += len(sample_files)

        with (prop_dir / "stats.txt").open("w", encoding="utf-8") as sf:
            sf.write(f"seed: {seed}\nproportion_pct: {pct}\ntrain_count: {total_count}\n")
            for cls in sorted(per_class_counts.keys()):
                sf.write(f"{cls}: {per_class_counts[cls]}\n")

        print(f"[INFO] {out_base.name} | seed {seed} | {pct}% -> {total_count} samples")

def main():
    for train_root in [_resolve_train_root(p) for p in TRAIN_ROOTS]:
        if not train_root.exists():
            print(f"[WARN] Train root not found: {train_root}")
            continue

        ds_dir = train_root.parent 
        train_classes = collect_class_files(train_root)

        print(f"\n[INFO] ===== Dataset: {ds_dir.name} =====")
        for c, fs in train_classes.items():
            print(f"  {c}: {len(fs)} files")

        for seed in SEEDS:
            print(f"[INFO] ---- Seed {seed} ----")
            sample_one_dataset(train_classes, ds_dir, seed, PROPORTIONS, do_copy=COPY_FILES)

    print("\n[DONE] All train subsets generated in respective dataset directories.")

if __name__ == "__main__":
    main()
