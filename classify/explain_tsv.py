#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Minimal TSV token-to-byte saliency heatmap explainer."""

import os, sys, json, math, random, traceback
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Main configuration.
CONFIG = {
    # Select model kind: "etbert" | "trafficformer".
    "model_kind": "trafficformer",

    # Repo roots and classifier entry paths.
    "repos": {
        "etbert": {
            "repo_root": os.path.join(PROJECT_ROOT, "ET-BERT"),
            "classifier_path": os.path.join(PROJECT_ROOT, "ET-BERT", "fine_tuning", "run_classifier.py"),
            "load_model_path": os.path.join(PROJECT_ROOT, "exp", "etbert_14", "ISCX-VPN_service_flow", "train", "model.bin"),
            "vocab_path": os.path.join(PROJECT_ROOT, "ET-BERT", "models", "encryptd_vocab_all.txt"),
        },
        "trafficformer": {
            "repo_root": os.path.join(PROJECT_ROOT, "TrafficFormer"),
            "classifier_path": os.path.join(PROJECT_ROOT, "TrafficFormer", "fine-tuning", "run_classifier.py"),
            "load_model_path": os.path.join(PROJECT_ROOT, "exp", "trafficformer_14", "ISCX-VPN_service_flow", "train", "model.bin"),
            "vocab_path": os.path.join(PROJECT_ROOT, "TrafficFormer", "models", "encryptd_vocab.txt"),
        },
    },

    # Dataset path (test split contains dataset.tsv and token_byte_index.jsonl).
    "dataset_root": os.path.join(PROJECT_ROOT, "outputs_tsv", "model", "ISCX-VPN_service_flow"),
    "split_subdir": "test",

    # Core model hyperparameters.
    "seq_length": 512,
    "pooling": "first",
    "labels_num": 11,
    "tokenizer": "bert",
    "batch_size": 64,

    # Explanation target class: "pred" | "gold".
    "target_class_mode": "pred",

    # Byte aggregation mode: "sum" | "max".
    "byte_reduce": "sum",

    # Heatmap y-axis window size in bytes.
    "BYTE_WINDOW": 64,

    # Fixed start offset in bytes.
    "START_OFFSET_BYTES": 14,

    # Maximum packet columns shown for sample figures.
    "PLOT_MAX_PACKETS": 5,

    # Sampling options.
    "draw_global_heatmap": True,
    "draw_sample_ratio": 0.0,
    "draw_sample_count": 3,
    "random_seed": 42,

    # Base output directory.
    "base_out_prefix": os.path.join(PROJECT_ROOT, "exp", "explanations"),

    # Alignment mode: "by_index" | "by_filename".
    "align_mode": "by_index",

    # Save per-sample token scores as .npy.
    "save_token_scores": False,

    # Plot options.
    "dpi": 180,
}

CLS_TOKEN = "[CLS]"
PAD_TOKEN = "[PAD]"
SEP_TOKEN = "[SEP]"

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p

# Load classifier from selected repository.
def load_uer_and_classifier():
    import importlib.util

    mkind = CONFIG["model_kind"]
    repo_root = os.path.abspath(CONFIG["repos"][mkind]["repo_root"])
    cls_abs   = os.path.abspath(CONFIG["repos"][mkind]["classifier_path"])

    if not os.path.isfile(cls_abs):
        raise FileNotFoundError(f"classifier_path does not exist: {cls_abs}")

    cls_dir = os.path.dirname(cls_abs)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    if cls_dir not in sys.path:
        sys.path.insert(0, cls_dir)

    spec = importlib.util.spec_from_file_location("user_defined_classifier", cls_abs)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "Classifier"):
        raise ImportError(f"Classifier class was not found in {cls_abs}.")

    from uer.utils import str2tokenizer
    return mod.Classifier, str2tokenizer

# TSV and mapping readers.
def read_tsv(tsv_path: str) -> Dict[str, Any]:
    rows, columns = [], {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            if i == 0:
                for j, name in enumerate(parts): columns[name] = j
                continue
            obj = {
                "text_a": parts[columns["text_a"]] if "text_a" in columns else "",
                "label": int(parts[columns["label"]]) if "label" in columns else None,
                "filename": parts[columns["filename"]] if "filename" in columns else None,
            }
            rows.append(obj)
    return {"rows": rows, "has_label": "label" in columns, "has_filename": "filename" in columns}

def read_mappings(jsonl_path: str) -> List[Dict[str, Any]]:
    maps = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f: maps.append(json.loads(line))
    return maps

# ---------------------------
# ---------------------------
def _vmax_from_grid(grid: np.ndarray) -> Optional[float]:
    if grid.size == 0: return None
    vals = grid.ravel()
    vmax = float(np.quantile(vals, 0.99)) if np.any(vals > 0) else None
    return vmax

# Aggregate real-byte saliency by packet.
def preprocess_mapping_offsets(mapping: Dict[str, Any]) -> None:
    offset = 0
    for p in mapping.get("packets", []):
        p["_flow_token_offset"] = offset
        offset += int(p.get("token_count", 0))

def aggregate_per_packet_realbytes(
    token_scores: np.ndarray,
    mapping: Dict[str, Any],
    reduce_mode: str = "sum",
    seq_offset: int = 1
) -> Dict[int, Dict[int, float]]:
    start_index_bytes = int(CONFIG.get("START_OFFSET_BYTES", 0))
    reduce = (lambda a, b: a + b) if reduce_mode == "sum" else (lambda a, b: max(a, b))
    per_packet: Dict[int, Dict[int, float]] = {}

    for p in mapping.get("packets", []):
        ordn = int(p["packet_ordinal"])
        byt_off = int(p.get("byte_offset", 0))
        flow_off = int(p.get("_flow_token_offset", 0))
        bucket = per_packet.setdefault(ordn, {})
        for tmap in p.get("mapping", []):
            tok_idx = int(tmap["token_index_in_packet"])
            glb_tok = seq_offset + flow_off + tok_idx
            if glb_tok < 0 or glb_tok >= len(token_scores):
                continue
            score = float(token_scores[glb_tok])

            bs_rel = int(tmap["byte_start_in_packet"])
            be_rel = int(tmap["byte_end_in_packet"])
            if be_rel <= bs_rel:
                continue

            bs_real_in_packet = start_index_bytes + bs_rel
            be_real_in_packet = start_index_bytes + be_rel

            for rb in range(byt_off + bs_real_in_packet, byt_off + be_real_in_packet):
                bucket[rb] = (reduce(bucket[rb], score) if rb in bucket else score)
    return per_packet

def aggregate_sample_sum_realbytes_from_packets(
    per_packet_real: Dict[int, Dict[int, float]]
) -> Dict[int, float]:
    sample_sum: Dict[int, float] = {}
    for _, d in per_packet_real.items():
        for rb, v in d.items():
            sample_sum[rb] = sample_sum.get(rb, 0.0) + float(v)
    return sample_sum

# Plotting helpers (only byte ticks and score colorbar).
def _grid_from_real_dict(real_dict: Dict[int, float], byte_window: int):
    if not real_dict:
        return np.zeros((byte_window, 1), dtype=np.float32), 0
    ymin = min(real_dict.keys())
    ymax = ymin + byte_window - 1
    grid = np.zeros((byte_window, 1), dtype=np.float32)
    for rb, v in real_dict.items():
        if ymin <= rb <= ymax:
            grid[rb - ymin, 0] += float(v)
    return grid, ymin

def _grid_from_packets_dict(
    per_packet_real: Dict[int, Dict[int, float]],
    byte_window: int,
    packets_max: int
):
    P = packets_max
    all_real = []
    for _, d in per_packet_real.items():
        if d: all_real.extend(d.keys())
    if not all_real:
        return np.zeros((byte_window, P), dtype=np.float32), 0
    ymin = min(all_real)
    ymax = ymin + byte_window - 1

    grid = np.zeros((byte_window, P), dtype=np.float32)
    for k in range(1, P+1):
        d = per_packet_real.get(k, {})
        for rb, v in d.items():
            if ymin <= rb <= ymax:
                grid[rb - ymin, k-1] += float(v)
    return grid, ymin

def _save_minimal_heatmap(grid: np.ndarray, out_png: str, y_start: int, show_colorbar: bool = True):
    vmax = _vmax_from_grid(grid)
    # Adapt width to number of columns.
    fig_w = 2.0 if grid.shape[1] == 1 else (1.8 + 0.4 * grid.shape[1])
    plt.figure(figsize=(fig_w, 6.0))
    im = plt.imshow(
        grid, aspect="auto", origin="lower", interpolation="nearest",
        vmin=0.0, vmax=(None if vmax is None else float(vmax))
    )

    # Keep only numeric y-axis ticks.
    B = grid.shape[0]
    ticks = list(range(0, B, max(1, B // 10)))
    labels = [str(y_start + t) for t in ticks]
    plt.yticks(ticks, labels, fontsize=9)

    # Remove x ticks and all spines.
    plt.xticks([], [])
    ax = plt.gca()
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(False)

    # Keep score colorbar.
    if show_colorbar:
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label("Score", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    ensure_dir(os.path.dirname(out_png))
    plt.tight_layout(pad=0.1)
    plt.savefig(out_png, dpi=CONFIG["dpi"], bbox_inches="tight", pad_inches=0.05)
    plt.close()

def draw_singlecol_heatmap_minimal(real_dict: Dict[int, float], out_png: str):
    B = CONFIG["BYTE_WINDOW"]
    grid, ymin = _grid_from_real_dict(real_dict, B)
    _save_minimal_heatmap(grid, out_png, y_start=ymin, show_colorbar=True)

def draw_multicol_packet_heatmap_minimal(per_packet_real: Dict[int, Dict[int, float]], out_png: str):
    P = CONFIG["PLOT_MAX_PACKETS"]
    B = CONFIG["BYTE_WINDOW"]
    grid, ymin = _grid_from_packets_dict(per_packet_real, B, P)
    _save_minimal_heatmap(grid, out_png, y_start=ymin, show_colorbar=True)

# Gradient-based explanation.
class Explainer:
    """Use embedding-output gradients (L2 norm) as token saliency."""
    def __init__(self, device: torch.device, model: nn.Module, tokenizer, labels_num: int):
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.labels_num = labels_num
        self.model.eval()
        self._emb_out = None
        self._emb_grad = None
        emb_module = getattr(self.model, "embedding", None)
        if emb_module is None:
            raise RuntimeError("Embedding module was not found on the model.")
        self._register_hook_on_embedding(emb_module)

    def _register_hook_on_embedding(self, emb_module: nn.Module):
        def _forward_hook(module, inputs, output):
            if isinstance(output, torch.Tensor):
                output = output.requires_grad_()
                self._emb_out = output
                def _capture_grad(grad): 
                    self._emb_grad = grad
                output.register_hook(_capture_grad)
            else:
                self._emb_out = output
        emb_module.register_forward_hook(_forward_hook)

    @torch.no_grad()
    def forward_logits(self, src_ids: torch.Tensor, seg_ids: torch.Tensor) -> torch.Tensor:
        _, logits = self.model(src_ids, None, seg_ids)
        return logits

    def token_importance(self, src_ids: torch.Tensor, seg_ids: torch.Tensor, target_class: Optional[int]) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        self._emb_out = None
        self._emb_grad = None
        _, logits = self.model(src_ids, None, seg_ids)
        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())
        logits[0, target_class].backward(retain_graph=False)
        if self._emb_grad is None:
            raise RuntimeError("Failed to capture embedding-output gradients.")
        grad = self._emb_grad.detach()  # [1, L, H]
        token_scores = torch.norm(grad[0], p=2, dim=-1)  # (L,)
        return token_scores.cpu().numpy()

# Build tokenizer and model.
def set_uer_default_flags(args):
    args.max_seq_length = getattr(args, "max_seq_length", args.seq_length)
    args.max_position_embeddings = getattr(args, "max_position_embeddings", args.seq_length)
    # Embedding defaults.
    args.word_embedding = getattr(args, "word_embedding", True)
    args.pos_embedding = getattr(args, "pos_embedding", True)
    args.seg_embedding = getattr(args, "seg_embedding", True)
    args.remove_embedding_layernorm = getattr(args, "remove_embedding_layernorm", False)
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    # LayerNorm
    args.layernorm = getattr(args, "layernorm", "normal")
    args.layernorm_positioning = getattr(args, "layernorm_positioning", "post")
    # Transformer defaults.
    args.hidden_size = getattr(args, "hidden_size", 768)
    args.emb_size = getattr(args, "emb_size", args.hidden_size)
    args.layers_num = getattr(args, "layers_num", 12)
    args.attention_heads_num = getattr(args, "attention_heads_num", 12)
    args.heads_num = getattr(args, "heads_num", args.attention_heads_num)
    args.dropout = getattr(args, "dropout", 0.1)
    args.bidirectional = getattr(args, "bidirectional", True)
    # FFN and activation defaults.
    args.feed_forward = getattr(args, "feed_forward", "dense")
    args.feedforward_size = getattr(args, "feedforward_size", 3072)
    args.hidden_act = getattr(args, "hidden_act", "gelu")
    # Relative position defaults.
    args.relative_position_embedding = getattr(args, "relative_position_embedding", False)
    args.relative_attention_buckets_num = getattr(args, "relative_attention_buckets_num", 32)
    args.remove_attention_scale = getattr(args, "remove_attention_scale", False)
    args.remove_transformer_bias = getattr(args, "remove_transformer_bias", False)
    # Compatibility defaults.
    args.parameter_sharing = getattr(args, "parameter_sharing", False)
    args.factorized_embedding_parameterization = getattr(args, "factorized_embedding_parameterization", False)
    args.is_moe = getattr(args, "is_moe", False)
    args.moebert_expert_dim = getattr(args, "moebert_expert_dim", 3072)
    args.moebert_expert_num = getattr(args, "moebert_expert_num", 1)
    args.moebert_route_method = getattr(args, "moebert_route_method", "hash-random")
    args.moebert_route_hash_list = getattr(args, "moebert_route_hash_list", None)
    args.moebert_load_balance = getattr(args, "moebert_load_balance", 0.0)
    # Tokenizer defaults.
    args.spm_model_path = getattr(args, "spm_model_path", "")
    args.cls_token = getattr(args, "cls_token", "[CLS]")
    args.sep_token = getattr(args, "sep_token", "[SEP]")
    args.pad_token = getattr(args, "pad_token", "[PAD]")
    args.pooling = getattr(args, "pooling", "first")
    args.soft_targets = getattr(args, "soft_targets", False)
    args.soft_alpha = getattr(args, "soft_alpha", False)

def build_tokenizer_and_model():
    Classifier, str2tokenizer = load_uer_and_classifier()

    class Args: ...
    args = Args()
    # Required fields.
    args.vocab_path = CONFIG["repos"][CONFIG["model_kind"]]["vocab_path"]
    args.tokenizer = CONFIG["tokenizer"]
    args.seq_length = CONFIG["seq_length"]
    args.pooling = CONFIG["pooling"]
    args.labels_num = CONFIG["labels_num"]
    args.embedding = "word_pos_seg"
    args.encoder = "transformer"
    args.mask = "fully_visible"
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.soft_targets = False
    args.soft_alpha = False

    set_uer_default_flags(args)

    from uer.utils import str2tokenizer as _s2t
    args.tokenizer = _s2t[CONFIG["tokenizer"]](args)

    model = Classifier(args)
    ckpt_path = CONFIG["repos"][CONFIG["model_kind"]]["load_model_path"]
    state = torch.load(ckpt_path, map_location="cpu")
    if "output_layer_2.weight" in state:
        args.labels_num = int(state["output_layer_2.weight"].shape[0])
    model.load_state_dict(state, strict=False)
    model = model.to(args.device)
    return args, model

# Token to id conversion.
def tokenize_text_a(tokenizer, text_a: str, seq_length: int) -> Tuple[List[int], List[int]]:
    ids = tokenizer.convert_tokens_to_ids([CLS_TOKEN] + tokenizer.tokenize(text_a))
    seg = [1] * len(ids)
    if len(ids) > seq_length:
        ids, seg = ids[:seq_length], seg[:seq_length]
    else:
        PAD_ID = tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0]
        while len(ids) < seq_length:
            ids.append(PAD_ID); seg.append(0)
    return ids, seg

# Main flow.
def main():
    try:
        mkind = CONFIG["model_kind"]
        ds_dir = os.path.join(CONFIG["dataset_root"], CONFIG["split_subdir"])
        tsv_path = os.path.join(ds_dir, "dataset.tsv")
        map_path = os.path.join(ds_dir, "token_byte_index.jsonl")

        if not os.path.isfile(tsv_path): raise FileNotFoundError(f"dataset.tsv not found: {tsv_path}")
        if not os.path.isfile(map_path): raise FileNotFoundError(f"token_byte_index.jsonl not found: {map_path}")

        # Build output path dynamically.
        out_root = os.path.join(
            CONFIG["base_out_prefix"],
            CONFIG["model_kind"],
            os.path.basename(CONFIG["dataset_root"]),
            CONFIG["split_subdir"],
        )
        ensure_dir(out_root)

        # Build model and tokenizer.
        args, model = build_tokenizer_and_model()
        device, tokenizer, labels_num = args.device, args.tokenizer, args.labels_num
        explainer = Explainer(device, model, tokenizer, labels_num)

        tsv = read_tsv(tsv_path)
        rows = tsv["rows"]
        mappings = read_mappings(map_path)

        # Align TSV rows with mapping entries.
        if CONFIG["align_mode"] == "by_index":
            N = min(len(rows), len(mappings))
            if len(rows) != len(mappings):
                print(f"[WARN] TSV({len(rows)}) != JSONL({len(mappings)}), aligned by min length {N}")
            rows, mappings = rows[:N], mappings[:N]
        else:
            if not tsv["has_filename"]:
                raise RuntimeError("align_mode=by_filename requires filename column in TSV.")
            name2idx = {m.get("file", ""): i for i, m in enumerate(mappings)}
            keep_rows, keep_maps = [], []
            for r in rows:
                idx = name2idx.get(r.get("filename", ""), None)
                if idx is not None: keep_rows.append(r); keep_maps.append(mappings[idx])
            rows, mappings = keep_rows, keep_maps

        total = len(rows)

        # Build sampled index set for sample visualizations.
        draw_ids = list(range(total))
        if CONFIG["draw_sample_count"] > 0:
            random.seed(CONFIG["random_seed"])
            draw_ids = random.sample(draw_ids, min(CONFIG["draw_sample_count"], total))
        elif CONFIG["draw_sample_ratio"] > 0:
            k = max(1, int(math.ceil(total * CONFIG["draw_sample_ratio"])))
            random.seed(CONFIG["random_seed"])
            draw_ids = random.sample(draw_ids, k)
        draw_ids = sorted(draw_ids)

        # Aggregation containers.
        P = CONFIG["PLOT_MAX_PACKETS"]

        # Global and class accumulators (merged single-column).
        global_accum_sum: Dict[int, float] = {}
        global_count = 0
        class_accum_sum: Dict[int, Dict[int, float]] = {}
        class_count: Dict[int, int] = {}

        # Global and class accumulators (per-packet multi-column).
        global_perpkt_accum: Dict[int, Dict[int, float]] = {}
        class_perpkt_accum: Dict[int, Dict[int, Dict[int, float]]] = {}

        print(f"== Start explaining: total {total} samples (sample {len(draw_ids)}) ==")

        # Main loop over samples.
        for i in range(total):
            row, mapping = rows[i], mappings[i]
            preprocess_mapping_offsets(mapping)

            text_a = row["text_a"]
            gold = row["label"] if tsv["has_label"] else None

            ids, seg = tokenize_text_a(tokenizer, text_a, CONFIG["seq_length"])
            src = torch.LongTensor(ids).unsqueeze(0).to(device)
            ssg = torch.LongTensor(seg).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = explainer.forward_logits(src, ssg)
                pred = int(torch.argmax(logits, dim=1).item())
            target = pred if (CONFIG["target_class_mode"] == "pred" or gold is None) else int(gold)

            token_scores = explainer.token_importance(src, ssg, target)
            if CONFIG["save_token_scores"]:
                np.save(os.path.join(out_root, f"token_scores_{i:06d}.npy"), token_scores)

            # Build per-packet real-byte saliency and merged view per sample.
            per_packet_real = aggregate_per_packet_realbytes(
                token_scores, mapping, reduce_mode=CONFIG["byte_reduce"], seq_offset=1
            )
            sample_sum = aggregate_sample_sum_realbytes_from_packets(per_packet_real)

            # Accumulate merged single-column statistics.
            for rb, v in sample_sum.items():
                global_accum_sum[rb] = global_accum_sum.get(rb, 0.0) + float(v)
            global_count += 1

            cls = gold if gold is not None else pred
            dst = class_accum_sum.setdefault(cls, {})
            for rb, v in sample_sum.items():
                dst[rb] = dst.get(rb, 0.0) + float(v)
            class_count[cls] = class_count.get(cls, 0) + 1

            # Accumulate per-packet statistics for multi-column plots.
            for pkt_idx, d in per_packet_real.items():
                gp = global_perpkt_accum.setdefault(pkt_idx, {})
                for rb, v in d.items():
                    gp[rb] = gp.get(rb, 0.0) + float(v)

            cp = class_perpkt_accum.setdefault(cls, {})
            for pkt_idx, d in per_packet_real.items():
                pdst = cp.setdefault(pkt_idx, {})
                for rb, v in d.items():
                    pdst[rb] = pdst.get(rb, 0.0) + float(v)

            # Render sampled multi-column packet heatmaps.
            if i in draw_ids:
                out_png = os.path.join(out_root, f"sample_{i:06d}.png")
                draw_multicol_packet_heatmap_minimal(per_packet_real, out_png)

            if (i + 1) % 100 == 0 or i == total - 1:
                print(f"[{i+1}/{total}] done")

        # Export global mean heatmaps.
        if CONFIG["draw_global_heatmap"] and global_count > 0:
            global_mean = {rb: v / max(1, global_count) for rb, v in global_accum_sum.items()}
            draw_singlecol_heatmap_minimal(
                global_mean,
                os.path.join(out_root, "global_mean.png"),
            )

            # Export per-packet global mean view.
            global_perpkt_mean: Dict[int, Dict[int, float]] = {}
            for pkt_idx, d in global_perpkt_accum.items():
                global_perpkt_mean[pkt_idx] = {rb: v / max(1, global_count) for rb, v in d.items()}
            draw_multicol_packet_heatmap_minimal(
                global_perpkt_mean,
                os.path.join(out_root, "global_mean_packets.png"),
            )

        # Export per-class mean heatmaps.
        for cls, acc in class_accum_sum.items():
            cnt = max(1, class_count.get(cls, 1))
            cls_mean = {rb: v / cnt for rb, v in acc.items()}
            draw_singlecol_heatmap_minimal(
                cls_mean,
                os.path.join(out_root, f"class_{cls}_mean.png"),
            )

            cp_acc = class_perpkt_accum.get(cls, {})
            cls_perpkt_mean: Dict[int, Dict[int, float]] = {}
            for pkt_idx, d in cp_acc.items():
                cls_perpkt_mean[pkt_idx] = {rb: v / cnt for rb, v in d.items()}
            draw_multicol_packet_heatmap_minimal(
                cls_perpkt_mean,
                os.path.join(out_root, f"class_{cls}_mean_packets.png"),
            )

        print(f"Done. Output directory: {out_root}")

    except Exception as e:
        print("\n[EXCEPTION] Execution failed:")
        print(str(e))
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
