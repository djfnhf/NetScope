#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pcap2tsv.py — Generic traffic TSV generator (per-packet sliding window)
-----------------------------------------------------------------------
Features:
1) Supports sequential generation for train/valid/test directories.
2) Flexible packet selection: firstN / randomN / all.
3) Optional padding behaviors: pad_bytes (intra-packet) / pad_packets (inter-packet).
4) Auto build / reuse label_map.json.
5) Multiprocessing.
6) Sliding window is performed per packet; tokens never cross packet boundaries.
7) Optional token→byte mapping file (token_byte_index.jsonl) with per-packet offsets.
"""

import os, json, random, multiprocessing as mp, hashlib
from scapy.all import rdpcap
from tqdm import tqdm
from utils import (
    truncate_bytes,
    safe_hex,
    randomize_sensitive_fields,
    zeroize_sensitive_fields,
    sliding_bigram_generation,
    build_label_map,
    write_dataset_tsv
)

# ==========================================================
# 0. Configuration
# ==========================================================

CONFIG = {
    "start_index": 76,          # Number of bytes to skip from the beginning of each packet payload (e.g., 76)
    "payload_len": 64,          # Number of bytes to keep per packet after start_index
    "window_size": 4,           # Sliding window width in HEX CHARACTERS (4 hex chars == 2 bytes)
    "stride": 2,                # Sliding stride in HEX CHARACTERS (2 hex chars == 1 byte)
    "token_limit": 320,         # Max tokens per flow (after concatenating per-packet tokens)
    "packet_strategy": "firstN",# firstN / randomN / all
    "packet_count": 5,          # N for firstN/randomN
    "pad_bytes": False,         # If True, pad per-packet bytes when shorter than payload_len
    "pad_packets": False,       # (Reserved) If True, pad missing packets up to N with empty ones
    "randomize": True,          # Randomize sensitive header fields
    "zeroize": False,           # Zeroize sensitive header fields
    "num_workers": 16,          # Number of worker processes

    # Mapping file switch
    "enable_mapping": True,     # Set False to skip building & writing the mapping file

    # New: unified global seed for full reproducibility <<<
    "global_seed": 0,     
}

# Paths and run-level options (edit here).
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL = "etbert"
DATASET_ROOT = os.path.join(PROJECT_ROOT, "datasets", "ISCX-VPN_service_flow")
INPUT_DIRS = [
    os.path.join(DATASET_ROOT, "train"),
    os.path.join(DATASET_ROOT, "valid"),
    os.path.join(DATASET_ROOT, "test"),
]
BASE_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "outputs_tsv_76")

# ---- helpers for deterministic seeding ----
def _stable_int_from_str(s: str) -> int:
    """Stable non-cryptographic int from string (independent of Python's hash randomization)."""
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)  # 32-bit slice is enough

# ==========================================================
# 1. Packet selection strategies
# ==========================================================

def select_packets(pkts, strategy="firstN", n=5):
    """Select packets according to the given strategy."""
    total = len(pkts)
    if total == 0:
        return []

    s = strategy.lower()
    if s.startswith("first"):
        return pkts[:n]
    elif s.startswith("random"):
        if total <= n:
            return pkts
        # random.sample() uses current RNG state; worker already seeded deterministically
        return random.sample(pkts, n)
    elif s == "all":
        return pkts
    else:
        # Fallback: firstN
        return pkts[:n]

# ==========================================================
# 2. Per-file worker task
# ==========================================================

def process_one_pcap_task(args):
    """
    Worker for one pcap:
    - Perform per-packet truncation and per-packet sliding window.
    - Concatenate tokens from selected packets in order (tokens never cross packets).
    - If enable_mapping=True, produce a per-packet mapping with byte offsets
      relative to the truncated payload (i.e., after start_index).
    """
    pcap_path, label, cfg = args

    # === NEW: unified deterministic random seed per file ===
    base_seed = int(cfg.get("global_seed", 0)) & 0xffffffff
    file_seed = _stable_int_from_str(os.path.basename(pcap_path))
    # Mix global_seed, file name, and label into one deterministic seed.
    final_seed = (base_seed * 1000003) ^ file_seed ^ (int(label) & 0xffffffff)
    random.seed(final_seed)
    # =======================================================

    try:
        pkts = rdpcap(pcap_path)
    except Exception:
        return None, None

    selected_pkts = select_packets(pkts, cfg["packet_strategy"], cfg["packet_count"])

    if cfg.get("randomize", False):
        selected_pkts = [randomize_sensitive_fields(p) for p in selected_pkts]
    if cfg.get("zeroize", False):
        selected_pkts = [zeroize_sensitive_fields(p) for p in selected_pkts]

    all_tokens = []
    enable_mapping = cfg.get("enable_mapping", True)
    per_packet_blocks = []  # only filled when enable_mapping=True

    for pkt_idx, pkt in enumerate(selected_pkts):
        try:
            raw_bytes = bytes(pkt)
            sub_bytes = truncate_bytes(
                raw_bytes,
                start=cfg["start_index"],
                length=cfg["payload_len"],
                pad_bytes=cfg.get("pad_bytes", True),
            )
            hex_str = safe_hex(sub_bytes)

            # NOTE:
            # - window_size / stride are in HEX CHARACTERS (2 hex chars == 1 byte).
            # - sliding_bigram_generation should respect token_limit at the flow level,
            #   but we still enforce a global limit right after concatenation.
            pkt_tokens = sliding_bigram_generation(
                hex_str,
                window_size=cfg["window_size"],
                stride=cfg["stride"],
                token_limit=cfg["token_limit"],
            )
        except Exception:
            continue

        # Build per-packet mapping (relative to the truncated payload)
        if enable_mapping:
            packet_mapping = []
            for j, tok in enumerate(pkt_tokens):
                # Convert hex indices to byte indices:
                #   byte_start_in_packet = (j * stride_hex) // 2
                #   byte_end_in_packet   = (j * stride_hex + window_size_hex) // 2
                byte_start_in_packet = (j * cfg["stride"]) // 2
                byte_end_in_packet   = (j * cfg["stride"] + cfg["window_size"]) // 2

                packet_mapping.append({
                    # Machine-friendly (0-based)
                    "packet_index": pkt_idx,
                    "token_index_in_packet": j,
                    "byte_start_in_packet": byte_start_in_packet,  # inclusive
                    "byte_end_in_packet": byte_end_in_packet,      # exclusive
                    "token": tok,

                    # Human-friendly (1-based, optional)
                    "packet_ordinal": pkt_idx + 1,
                    "byte_start_in_packet_1b": byte_start_in_packet + 1,
                    "byte_end_in_packet_1b": byte_end_in_packet
                })

            per_packet_blocks.append({
                "packet_index": pkt_idx,
                "packet_ordinal": pkt_idx + 1,
                "token_count": len(pkt_tokens),
                "mapping": packet_mapping,
                "meta": {
                    "start_index_bytes": cfg["start_index"],
                    "payload_len_bytes": cfg["payload_len"],
                    "window_size_hex": cfg["window_size"],
                    "stride_hex": cfg["stride"],
                    "note": "byte_*_in_packet are relative to the truncated payload (after start_index)."
                }
            })

        # Append to the flow-level token sequence
        all_tokens.extend(pkt_tokens)

        # Enforce global token limit at flow level
        if len(all_tokens) >= cfg["token_limit"]:
            all_tokens = all_tokens[:cfg["token_limit"]]
            break

    # Prepare mapping info (one line per file)
    mapping_info = None
    if enable_mapping:
        mapping_info = {
            "file": os.path.basename(pcap_path),
            "label": label,
            "total_tokens": len(all_tokens),
            "packets": per_packet_blocks
        }

    return {"label": label, "token_sequence": all_tokens}, mapping_info

# ==========================================================
# 3. Directory-level generation (multiprocessing)
# ==========================================================

def generate_tsv_from_dir(input_dir, label_map, cfg, output_dir):
    samples, all_index = [], []
    os.makedirs(output_dir, exist_ok=True)

    dataset_tsv = os.path.join(output_dir, "dataset.tsv")
    token_index_path = os.path.join(output_dir, "token_byte_index.jsonl")

    tsv_exists = os.path.exists(dataset_tsv)
    map_exists = os.path.exists(token_index_path)
    enable_mapping = cfg.get("enable_mapping", True)

    if tsv_exists and enable_mapping and not map_exists:
        print(f"[•] {output_dir}: dataset.tsv exists but no mapping. Rebuilding mapping only...")

        tasks = []
        # Sort to keep deterministic order.
        for label_name, label_id in sorted(label_map.items(), key=lambda x: x[0]):
            class_dir = os.path.join(input_dir, label_name)
            if not os.path.exists(class_dir):
                continue
            for f in sorted(os.listdir(class_dir)):
                if f.endswith(".pcap"):
                    tasks.append((os.path.join(class_dir, f), label_id, cfg))

        with mp.get_context("spawn").Pool(processes=cfg.get("num_workers", 8)) as pool:
            # Use imap to preserve task order.
            for _, mapping_info in tqdm(
                pool.imap(process_one_pcap_task, tasks),
                total=len(tasks),
                desc=f"Re-mapping {os.path.basename(input_dir)}"
            ):
                if mapping_info:
                    all_index.append(mapping_info)

        # Sort by file name before writing for stable ordering.
        all_index.sort(key=lambda x: x["file"])
        with open(token_index_path, "w", encoding="utf-8") as f:
            for item in all_index:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"[✓] Mapping rebuilt → {token_index_path}")
        return

    if tsv_exists and map_exists:
        print(f"[•] Skip {output_dir}: dataset.tsv and mapping already exist.")
        return

    print(f"[+] Found {len(os.listdir(input_dir))} classes in {input_dir}")

    tasks = []
    for label_name, label_id in sorted(label_map.items(), key=lambda x: x[0]):
        class_dir = os.path.join(input_dir, label_name)
        if not os.path.exists(class_dir):
            continue
        for f in sorted(os.listdir(class_dir)):
            if f.endswith(".pcap"):
                tasks.append((os.path.join(class_dir, f), label_id, cfg))

    print(f"[+] Found {len(tasks)} pcap files in {input_dir}")

    with mp.get_context("spawn").Pool(processes=cfg.get("num_workers", 8)) as pool:
        for s, mapping_info in tqdm(
            pool.imap(process_one_pcap_task, tasks),   # Keep result order aligned with tasks.
            total=len(tasks),
            desc=f"Processing {os.path.basename(input_dir)}"
        ):
            if s:
                samples.append(s)
                if enable_mapping and mapping_info:
                    all_index.append(mapping_info)

    write_dataset_tsv(samples, dataset_tsv)
    print(f"[✓] Generated {dataset_tsv}  (Total: {len(samples)} samples)")

    # Write mapping file.
    if enable_mapping:
        # Sort by file name before writing for stable ordering.
        all_index.sort(key=lambda x: x["file"])
        with open(token_index_path, "w", encoding="utf-8") as f:
            for item in all_index:
                if item is not None:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[✓] Token-byte index saved to {token_index_path}")
    else:
        print("[•] Mapping disabled → token_byte_index.jsonl not generated.")

# ==========================================================
# 4. Main: auto path wiring + label_map build/reuse
# ==========================================================
if __name__ == "__main__":
    # Set global seed in main process.
    random.seed(CONFIG.get("global_seed", 0))
    DATASET_NAME = os.path.basename(DATASET_ROOT.rstrip("/"))
    MODEL_OUTPUT_ROOT = os.path.join(BASE_OUTPUT_ROOT, MODEL, DATASET_NAME)
    os.makedirs(MODEL_OUTPUT_ROOT, exist_ok=True)

    label_map_path = os.path.join(MODEL_OUTPUT_ROOT, "label_map.json")

    for input_dir in INPUT_DIRS:
        split_name = os.path.basename(input_dir)
        output_dir = os.path.join(MODEL_OUTPUT_ROOT, split_name)
        os.makedirs(output_dir, exist_ok=True)

        # Build or reuse label_map.
        if os.path.exists(label_map_path):
            with open(label_map_path, "r") as f:
                label_map = json.load(f)
            print(f"[•] Reusing label_map.json from {label_map_path}")
        else:
            label_map = build_label_map(input_dir)
            with open(label_map_path, "w") as f:
                json.dump(label_map, f, indent=2, ensure_ascii=False)
            print(f"[+] Built new label_map.json → {label_map_path}")

        print(f"\n[+] Processing split: {split_name.upper()} → {output_dir}")
        generate_tsv_from_dir(input_dir, label_map, CONFIG, output_dir)
