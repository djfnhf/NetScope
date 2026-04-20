#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""PNG saliency explainer for YaTC using token_byte_index.jsonl mapping."""

import os
import sys
import json
import random
import traceback
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from PIL import Image
from torchvision import datasets, transforms

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Main configuration.
CONFIG = {
    # Data and mapping.
    "png_root": os.path.join(PROJECT_ROOT, "outputs_png", "yatc", "ISCX-VPN_service_flow"),
    "split_subdir": "test",
    "jsonl_name": "token_byte_index.jsonl",

    # Model.
    "repo_root": os.path.join(PROJECT_ROOT, "YaTC"),
    "model_name": "TraFormer_YaTC",
    "resume_ckpt": os.path.join(PROJECT_ROOT, "exp", "yatc", "ISCX-VPN_service_flow", "train", "best_acc.pth"),
    "nb_classes": 11,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # Saliency settings.
    "target_class_mode": "pred",              # "pred" | "gold"
    "scale_grad_by_inv_std": False,

    # Shape constants for plotting.
    "packets": 5,
    "header_bytes": 80,
    "payload_bytes": 240,
    "matrix_rows": 40,
    "matrix_cols": 40,

    # Plot and statistics.
    "BYTE_WINDOW": 320,
    "PLOT_MAX_PACKETS": 5,
    "draw_global_heatmap": True,
    "draw_sample_count": 6,
    "random_seed": 42,
    "dpi": 180,

    # Output directory.
    "base_out_dir": os.path.join(PROJECT_ROOT, "exp", "explain_png"),
}

# Utility helpers.
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True); return p

def _vmax_from_grid(grid: np.ndarray) -> Optional[float]:
    if grid.size == 0: return None
    vals = grid.ravel()
    return float(np.quantile(vals, 0.99)) if np.any(vals > 0) else None

def grid_from_packets_dict(per_packet_real: Dict[int, Dict[int, float]], byte_window: int, packets_max: int):
    P = packets_max
    all_real = []
    for d in per_packet_real.values():
        if d: all_real.extend(d.keys())
    if not all_real:
        return np.zeros((byte_window, P), dtype=np.float32), 0
    ymin = min(all_real)
    grid = np.zeros((byte_window, P), dtype=np.float32)
    for k in range(1, P+1):
        d = per_packet_real.get(k, {})
        for rb, v in d.items():
            if ymin <= rb < ymin + byte_window:
                grid[rb - ymin, k-1] += float(v)
    return grid, ymin

def grid_from_real_dict(real_dict: Dict[int, float], byte_window: int):
    if not real_dict:
        return np.zeros((byte_window, 1), dtype=np.float32), 0
    ymin = min(real_dict.keys())
    grid = np.zeros((byte_window, 1), dtype=np.float32)
    for rb, v in real_dict.items():
        if ymin <= rb < ymin + byte_window:
            grid[rb - ymin, 0] += float(v)
    return grid, ymin

# Minimal heatmap plotting.

def _save_minimal_heatmap(grid: np.ndarray, out_png: str, y_start: int):
    hb = int(CONFIG["header_bytes"])
    pb = int(CONFIG["payload_bytes"])
    total = hb + pb

    vmax = _vmax_from_grid(grid)
    fig_w = 2.0 if grid.shape[1] == 1 else (1.8 + 0.4 * grid.shape[1])
    fig = plt.figure(figsize=(fig_w, 6.0))
    ax = plt.gca()
    im = ax.imshow(
        grid, aspect="auto", origin="lower", interpolation="nearest",
        vmin=0.0, vmax=(None if vmax is None else float(vmax))
    )

    B = grid.shape[0]
    y_end = y_start + B - 1
    divider_in_view = (y_start <= hb <= y_end)
    y_div = hb - y_start

    # Segment-aware y-axis ticks.
    step = max(1, B // 10)
    ticks = list(range(0, B, step))
    if (B - 1) not in ticks:
        ticks.append(B - 1)

    def _fmt(y, _pos=None):
        y_abs = int(round(y_start + y))
        if y_abs <= hb:
            return f"{max(0, y_abs)}"
        else:
            return f"{y_abs - hb}"
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(_fmt))
    ax.tick_params(axis="y", labelsize=9, width=0.8, length=4)

    if divider_in_view:
        ax.axhline(y_div, color="red", lw=1.2, linestyle="--")

        if len(ticks) > 0:
            _thr = max(1.0, step / 2.0)
            ticks = [t for t in ticks if abs(t - y_div) >= _thr]
            ax.set_yticks(ticks)
            ax.yaxis.set_major_formatter(mtick.FuncFormatter(_fmt))
            ax.tick_params(axis="y", which="major", labelsize=9, width=0.8, length=4)

        ax.set_yticks([y_div], minor=True)
        ax.set_yticklabels([f"{int(CONFIG['header_bytes'])}"], minor=True)
        ax.tick_params(axis="y", which="minor",
                       length=8, width=1.2, colors="red", labelsize=9)



    ax.set_xticks([]); ax.set_xticklabels([])
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(False)

    # Keep only the score colorbar.
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
    cbar.set_label("Score", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ensure_dir(os.path.dirname(out_png))
    plt.tight_layout(pad=0.2)
    plt.savefig(out_png, dpi=CONFIG["dpi"], bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

def draw_multicol_packet_heatmap(per_packet_real: Dict[int, Dict[int, float]], out_png: str, title: Optional[str] = None):
    P = CONFIG["PLOT_MAX_PACKETS"]; B = CONFIG["BYTE_WINDOW"]
    grid, ymin = grid_from_packets_dict(per_packet_real, B, P)
    _save_minimal_heatmap(grid, out_png, y_start=ymin)

def draw_singlecol_heatmap(real_dict: Dict[int, float], out_png: str, title: Optional[str] = None):
    B = CONFIG["BYTE_WINDOW"]
    grid, ymin = grid_from_real_dict(real_dict, B)
    _save_minimal_heatmap(grid, out_png, y_start=ymin)

# Model loading.
def load_yatc_model():
    sys.path.insert(0, CONFIG["repo_root"])
    import models_YaTC  # noqa: F401

    model = getattr(models_YaTC, CONFIG["model_name"])(
        num_classes=CONFIG["nb_classes"], drop_path_rate=0.1,
    )
    ckpt = torch.load(CONFIG["resume_ckpt"], map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    msg = model.load_state_dict(state, strict=False)
    print("[load] checkpoint:", CONFIG["resume_ckpt"])
    print("[load] missing/unexpected:", msg)
    model = model.to(CONFIG["device"])
    model.eval()
    return model

class PNGExplainer:
    """Backprop target logit to input and use absolute gradient."""
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model; self.device = device

    def pixel_importance(self, input_tensor: torch.Tensor, target_class: Optional[int]) -> Tuple[np.ndarray, int]:
        x = input_tensor.clone().detach().to(self.device)
        x.requires_grad_(True)
        logits = self.model(x)
        pred = int(torch.argmax(logits, dim=1).item())
        t = pred if (target_class is None) else int(target_class)
        logits[0, t].backward()
        grad = x.grad.detach().abs()[0, 0]  # [40,40]
        if CONFIG["scale_grad_by_inv_std"]:
            grad = grad * 2.0  # std=0.5
        return grad.flatten().cpu().numpy(), pred

# JSONL mapping loading.
def load_jsonl_index(split_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(split_dir, CONFIG["jsonl_name"])
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing JSONL mapping: {path}")
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    print(f"[index] loaded {len(items)} records from {path}")
    return items

def build_relpng_to_record(index_items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    m = {}
    for rec in index_items:
        rel_png = rec.get("rel_png") or ""
        m[rel_png] = rec
    return m

# Main entry.
def main():
    try:
        png_root = CONFIG["png_root"]
        split_dir = os.path.join(png_root, CONFIG["split_subdir"])
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"split_dir not found: {split_dir}")

        out_root = os.path.join(CONFIG["base_out_dir"],
                                os.path.basename(png_root),
                                CONFIG["split_subdir"])
        ensure_dir(out_root)

        # Build dataset for file paths and gold labels.
        mean = [0.5]; std = [0.5]
        to_tensor = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = datasets.ImageFolder(split_dir, transform=to_tensor)
        classes = dataset.classes
        print(f"[data] classes={len(classes)} samples={len(dataset)} split={split_dir}")

        # JSONL mapping must exist.
        index_items = load_jsonl_index(split_dir)
        relpng2rec = build_relpng_to_record(index_items)

        # Randomly sample instances for visualization.
        total = len(dataset)
        draw_ids = list(range(total))
        if CONFIG["draw_sample_count"] > 0:
            random.seed(CONFIG["random_seed"])
            draw_ids = random.sample(draw_ids, min(CONFIG["draw_sample_count"], total))
        draw_ids = sorted(draw_ids)

        # Build model and explainer.
        model = load_yatc_model()
        explainer = PNGExplainer(model, CONFIG["device"])

        # Accumulators for merged (single-column) views.
        global_accum_sum: Dict[int, float] = {}
        global_count = 0
        class_accum_sum: Dict[int, Dict[int, float]] = {}
        class_count: Dict[int, int] = {}

        # Accumulators for per-packet (multi-column) views.
        global_perpkt_accum: Dict[int, Dict[int, float]] = {}               # pkt_idx -> {real_byte: sum}
        class_perpkt_accum: Dict[int, Dict[int, Dict[int, float]]] = {}     # cls -> pkt_idx -> {real_byte: sum}

        # Main loop.
        for i in range(total):
            abs_png, gold = dataset.samples[i]  # (path, class_id)
            # Build relative PNG path under split_dir.
            rel_png = os.path.relpath(abs_png, split_dir)
            alt_rel_png = rel_png.replace("\\", "/")
            rec = relpng2rec.get(rel_png) or relpng2rec.get(alt_rel_png)

            if rec is None:
                print(f"[warn] no index for: {rel_png}, skip")
                continue

            # Prepare input tensor.
            img = Image.open(abs_png).convert("L")
            x = to_tensor(img).unsqueeze(0)  # [1,1,40,40]

            # Choose target class.
            target = None if CONFIG["target_class_mode"] == "pred" else int(gold)

            # Compute per-pixel saliency (length 1600).
            pix_imp, pred = explainer.pixel_importance(x, target)

            # Map pixel saliency to real-byte space using JSONL mapping.
            per_packet_real: Dict[int, Dict[int, float]] = {}
            pixels = rec.get("pixels", [])
            if len(pixels) != CONFIG["matrix_rows"] * CONFIG["matrix_cols"]:
                print(f"[warn] bad pixels length for: {rel_png}, expect 1600 got {len(pixels)}; skip")
                continue

            for g, pmap in enumerate(pixels):
                if pmap.get("global_offset", g) != g:
                    g = int(pmap["global_offset"])
                pkt = int(pmap["packet_index"])          # 0..4
                region = pmap["region"]                   # "header"|"payload"
                roff = int(pmap["region_offset"])        # 0..(hb-1) or 0..(pb-1)
                score = float(pix_imp[g])

                pkt_ordinal = pkt + 1                    # 1..5
                # Convert to real byte index within packet.
                real_byte_in_packet = roff if region == "header" else (CONFIG["header_bytes"] + roff)

                bucket = per_packet_real.setdefault(pkt_ordinal, {})
                bucket[real_byte_in_packet] = bucket.get(real_byte_in_packet, 0.0) + score

            # Accumulate merged single-column statistics.
            sample_sum: Dict[int, float] = {}
            for _, d in per_packet_real.items():
                for rb, v in d.items():
                    sample_sum[rb] = sample_sum.get(rb, 0.0) + float(v)

            for rb, v in sample_sum.items():
                global_accum_sum[rb] = global_accum_sum.get(rb, 0.0) + float(v)
            global_count += 1

            cls = int(gold) if gold is not None else int(pred)
            dst = class_accum_sum.setdefault(cls, {})
            for rb, v in sample_sum.items():
                dst[rb] = dst.get(rb, 0.0) + float(v)
            class_count[cls] = class_count.get(cls, 0) + 1

            # Accumulate per-packet multi-column statistics.
            for pkt_idx, d in per_packet_real.items():
                gp = global_perpkt_accum.setdefault(pkt_idx, {})
                for rb, v in d.items():
                    gp[rb] = gp.get(rb, 0.0) + float(v)

            cp = class_perpkt_accum.setdefault(cls, {})
            for pkt_idx, d in per_packet_real.items():
                pdst = cp.setdefault(pkt_idx, {})
                for rb, v in d.items():
                    pdst[rb] = pdst.get(rb, 0.0) + float(v)

            # Draw sampled multi-column heatmaps.
            if i in draw_ids:
                out_png = os.path.join(out_root, f"sample_{i:06d}.png")
                draw_multicol_packet_heatmap(per_packet_real, out_png, title=None)

            if (i + 1) % 100 == 0 or i == total - 1:
                print(f"[{i+1}/{total}] done")

        # Draw global/class mean heatmaps (single-column).
        if CONFIG["draw_global_heatmap"] and global_count > 0:
            global_mean = {rb: v / max(1, global_count) for rb, v in global_accum_sum.items()}
            draw_singlecol_heatmap(
                global_mean,
                os.path.join(out_root, "global_mean.png"),
                title=None
            )

        for cls, acc in class_accum_sum.items():
            cnt = max(1, class_count.get(cls, 1))
            cls_mean = {rb: v / cnt for rb, v in acc.items()}
            draw_singlecol_heatmap(
                cls_mean,
                os.path.join(out_root, f"class_{cls}_mean.png"),
                title=None
            )

        # Draw global/class mean heatmaps (per-packet multi-column).
        if global_count > 0:
            global_perpkt_mean: Dict[int, Dict[int, float]] = {}
            for pkt_idx, d in global_perpkt_accum.items():
                global_perpkt_mean[pkt_idx] = {rb: v / max(1, global_count) for rb, v in d.items()}
            draw_multicol_packet_heatmap(
                global_perpkt_mean,
                os.path.join(out_root, "global_mean_packets.png"),
                title=None
            )

        for cls, cp_acc in class_perpkt_accum.items():
            cnt = max(1, class_count.get(cls, 1))
            cls_perpkt_mean: Dict[int, Dict[int, float]] = {}
            for pkt_idx, d in cp_acc.items():
                cls_perpkt_mean[pkt_idx] = {rb: v / cnt for rb, v in d.items()}
            draw_multicol_packet_heatmap(
                cls_perpkt_mean,
                os.path.join(out_root, f"class_{cls}_mean_packets.png"),
                title=None
            )

        print(f"Done. Output directory: {out_root}")

    except Exception as e:
        print("\n[EXCEPTION] Execution failed:")
        print(str(e)); traceback.print_exc(); sys.exit(1)

if __name__ == "__main__":
    main()
