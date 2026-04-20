#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import json
import time
import traceback
import random
import hashlib
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
from PIL import Image
from tqdm import tqdm

# ========= Global Seed =========
# Read GLOBAL_SEED from environment, default to 2025.
GLOBAL_SEED = int(os.getenv("GLOBAL_SEED", "2025"))

def _set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# Set global seed once in main process.
_set_global_seed(GLOBAL_SEED)

def _seed_from_key(key: str):
    """Generate deterministic sub-seed from (GLOBAL_SEED, key)."""
    h = hashlib.sha256(f"{GLOBAL_SEED}|{key}".encode("utf-8")).hexdigest()
    # Use first 16 hex chars as stable 32-bit seed.
    sub = int(h[:16], 16) & 0xFFFFFFFF
    random.seed(sub)
    np.random.seed(sub)
# =================================

# --------- config (edit here) ---------
MODEL = "yatc"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_ROOT = os.path.join(PROJECT_ROOT, "datasets", "cstnet_120_flow")

INPUT_DIRS = [
    os.path.join(DATASET_ROOT, "train"),
    os.path.join(DATASET_ROOT, "valid"),
    os.path.join(DATASET_ROOT, "test"),
]

BASE_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "outputs_png")

PACKETS = 5
HEADER_BYTES = 80
PAYLOAD_BYTES = 240
BYTES_PER_ROW = 40
ROWS_PER_PACKET = 8
MATRIX_ROWS = PACKETS * ROWS_PER_PACKET  # 40
MATRIX_COLS = BYTES_PER_ROW              # 40
BYTES_PER_PACKET = HEADER_BYTES + PAYLOAD_BYTES  # 320
PNG_DTYPE = np.uint8
NUM_WORKERS = max(1, cpu_count() - 1)
# --------------------------------------

EXTERNAL_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
if not os.path.isdir(EXTERNAL_UTILS_DIR):
    raise RuntimeError(f"External utils directory not found: {EXTERNAL_UTILS_DIR}")
sys.path.append(EXTERNAL_UTILS_DIR)
try:
    import utils as ext_utils
except Exception as e:
    raise RuntimeError(f"Cannot import {EXTERNAL_UTILS_DIR}/utils.py: {e}")
if not hasattr(ext_utils, "randomize_sensitive_fields"):
    raise RuntimeError("utils.py must define randomize_sensitive_fields().")

from scapy.all import Raw
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6
from scapy.packet import Packet
from scapy.utils import PcapReader


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def sanitize_headers_random(pkt):
    if IP not in pkt and IPv6 not in pkt:
        return None
    # randomize_sensitive_fields may rely on random/np state.
    p = ext_utils.randomize_sensitive_fields(pkt)
    if not isinstance(p, Packet):
        raise TypeError("randomize_sensitive_fields() must return a Scapy Packet")
    p = p.copy()
    if Raw in p:
        del p[Raw]
    if IP in p:
        p[IP].chksum = None
    if TCP in p:
        p[TCP].chksum = None
    if UDP in p:
        p[UDP].chksum = None
    return bytes(p)


def raw_payload_bytes(pkt):
    return bytes(pkt[Raw]) if Raw in pkt else b""


def packet_to_320B(pkt):
    hdr = sanitize_headers_random(pkt)
    if hdr is None:
        return None
    pay = raw_payload_bytes(pkt)
    hdr = (hdr[:HEADER_BYTES]).ljust(HEADER_BYTES, b"\x00")
    pay = (pay[:PAYLOAD_BYTES]).ljust(PAYLOAD_BYTES, b"\x00")
    return hdr + pay  # 80 + 240


def pcap_to_matrix_40x40(pcap_path: str):
    chunks = []
    try:
        with PcapReader(pcap_path) as pr:
            for pkt in pr:
                if (IP not in pkt) and (IPv6 not in pkt):
                    continue
                buf = packet_to_320B(pkt)
                if buf is None:
                    continue
                chunks.append(buf)
                if len(chunks) >= PACKETS:
                    break
    except Exception:
        return None

    if not chunks:
        return None
    while len(chunks) < PACKETS:
        chunks.append(b"\x00" * (HEADER_BYTES + PAYLOAD_BYTES))

    blob = b"".join(chunks)  # 1600 bytes
    return np.frombuffer(blob, dtype=PNG_DTYPE).reshape(MATRIX_ROWS, MATRIX_COLS)


def build_label_map(split_root: str, out_json_path: str):
    classes = sorted([
        d for d in os.listdir(split_root)
        if os.path.isdir(os.path.join(split_root, d))
    ])
    ensure_dir(os.path.dirname(out_json_path))
    label_map = {cls: idx for idx, cls in enumerate(classes)}
    with open(out_json_path, "w") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)
    print(f"[label_map] -> {out_json_path} : {label_map}")
    return label_map


def list_pcaps_under_split(split_root: str):
    files = []
    files += glob.glob(os.path.join(split_root, "**", "*.pcap"), recursive=True)
    files += glob.glob(os.path.join(split_root, "**", "*.pcapng"), recursive=True)
    return files


def list_pngs_under_output(out_split_dir: str):
    files = []
    files += glob.glob(os.path.join(out_split_dir, "**", "*.png"), recursive=True)
    return files


def _build_pixel_mapping(mat: np.ndarray):
    """
    Return a pixels list (length 1600), each item:
      {
        "row", "col", "global_offset",
        "packet_index", "region", "region_offset", "value",
        "row_in_packet", "col_in_row", "region_label"
      }
    """
    if mat.shape != (MATRIX_ROWS, MATRIX_COLS):
        raise ValueError("Matrix must be 40x40.")
    pixels = []
    for r in range(MATRIX_ROWS):
        for c in range(MATRIX_COLS):
            global_offset = r * MATRIX_COLS + c            # 0..1599
            packet_index = global_offset // BYTES_PER_PACKET  # 0..4
            inner = global_offset % BYTES_PER_PACKET       # 0..319
            row_in_packet = r % ROWS_PER_PACKET
            col_in_row = c

            if inner < HEADER_BYTES:
                region = "header"
                region_offset = inner                       # 0..79
                region_label = "Header 0–80"
            else:
                region = "payload"
                region_offset = inner - HEADER_BYTES        # 0..239
                region_label = "Payload 0–240"

            val = int(mat[r, c])
            pixels.append({
                "row": r,
                "col": c,
                "global_offset": global_offset,
                "packet_index": packet_index,
                "row_in_packet": row_in_packet,
                "col_in_row": col_in_row,
                "region": region,
                "region_offset": region_offset,
                "region_label": region_label,
                "value": val
            })
    return pixels


def _make_record(rel_pcap: str, rel_png: str, label_id: int, class_name: str, mat: np.ndarray, out_split_dir: str):
    """Build one JSON record for one file."""
    file_name = os.path.basename(rel_pcap) if rel_pcap else os.path.basename(rel_png)
    record = {
        "file": file_name,
        "rel_pcap": rel_pcap,
        "rel_png": rel_png,
        "label": label_id,
        "class_name": class_name,
        "total_pixels": MATRIX_ROWS * MATRIX_COLS,
        "meta": {
            "packets": PACKETS,
            "header_bytes": HEADER_BYTES,
            "payload_bytes": PAYLOAD_BYTES,
            "matrix_rows": MATRIX_ROWS,
            "matrix_cols": MATRIX_COLS,
            # Explicit region definition for visualization/explanation.
            "region_def": {
                "header": {"label": "Header 0–80", "bytes": [0, HEADER_BYTES]},
                "payload": {"label": "Payload 0–240", "bytes": [0, PAYLOAD_BYTES]}
            }
        },
        "pixels": _build_pixel_mapping(mat)
    }
    return record


def _worker_from_pcap(pcap_path: str, out_split_dir: str, split_root: str, label_map: dict):
    """
    Generate PNG from pcap and return one JSON record.
    Return: (ok, json_record)
    """
    try:
        # Seed by input path for deterministic parallel runs.
        _seed_from_key(os.path.relpath(pcap_path, split_root))

        mat = pcap_to_matrix_40x40(pcap_path)
        if mat is None:
            return False, None

        rel = os.path.relpath(pcap_path, split_root)  # e.g., class/host/file.pcap
        out_path = os.path.join(out_split_dir, rel)
        out_path = out_path.replace(".pcapng", ".png").replace(".pcap", ".png")
        ensure_dir(os.path.dirname(out_path))
        Image.fromarray(mat).save(out_path)

        parts = rel.split(os.sep)
        class_name = parts[0] if parts else ""
        label_id = label_map.get(class_name, -1)

        rec = _make_record(
            rel_pcap=rel,
            rel_png=os.path.relpath(out_path, out_split_dir),
            label_id=label_id,
            class_name=class_name,
            mat=mat,
            out_split_dir=out_split_dir
        )
        return True, rec

    except Exception as e:
        print(f"[ERR-PCAP] {pcap_path}: {e}", flush=True)
        traceback.print_exc()
        return False, None


def _worker_from_png(png_path: str, out_split_dir: str, split_root: str, label_map: dict):
    """Rebuild index using existing PNG files only."""
    try:
        # Seed by PNG relative path for deterministic indexing.
        _seed_from_key(os.path.relpath(png_path, out_split_dir))

        img = Image.open(png_path).convert("L")
        mat = np.array(img, dtype=PNG_DTYPE)
        if mat.shape != (MATRIX_ROWS, MATRIX_COLS):
            raise ValueError(f"PNG shape must be {MATRIX_ROWS}x{MATRIX_COLS}, got {mat.shape}")

        rel_png = os.path.relpath(png_path, out_split_dir)  # e.g., class/host/file.png
        parts = rel_png.split(os.sep)
        class_name = parts[0] if parts else ""
        label_id = label_map.get(class_name, -1)

        # Try to infer source pcap relative path.
        base_no_ext = os.path.splitext(rel_png)[0]  # class/host/file
        candidate_pcap = base_no_ext + ".pcap"
        candidate_pcapng = base_no_ext + ".pcapng"
        abs_candidate_pcap = os.path.join(split_root, candidate_pcap)
        abs_candidate_pcapng = os.path.join(split_root, candidate_pcapng)
        rel_pcap = None
        if os.path.exists(abs_candidate_pcap):
            rel_pcap = candidate_pcap
        elif os.path.exists(abs_candidate_pcapng):
            rel_pcap = candidate_pcapng

        rec = _make_record(
            rel_pcap=rel_pcap,
            rel_png=rel_png,
            label_id=label_id,
            class_name=class_name,
            mat=mat,
            out_split_dir=out_split_dir
        )
        return True, rec

    except Exception as e:
        print(f"[ERR-PNG] {png_path}: {e}", flush=True)
        traceback.print_exc()
        return False, None


def _append_jsonl(path: str, objs):
    """Write JSON objects to JSONL (overwrite mode)."""
    with open(path, "w", encoding="utf-8") as f:
        for o in objs:
            if o is not None:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")


def process_one_split(split_root: str, model_out_root: str, label_map_path: str):
    split_name = os.path.basename(split_root.rstrip("/"))
    out_split_dir = os.path.join(model_out_root, split_name)
    ensure_dir(out_split_dir)

    # 1) Build or reuse label_map.
    if os.path.exists(label_map_path):
        with open(label_map_path, "r") as f:
            label_map = json.load(f)
        print(f"[reuse] label_map.json: {label_map_path}")
    else:
        label_map = build_label_map(split_root, label_map_path)

    # 2) Target index path.
    jsonl_path = os.path.join(out_split_dir, "token_byte_index.jsonl")

    # 3) Check whether PNG files already exist.
    existing_pngs = list_pngs_under_output(out_split_dir)
    png_exists = len(existing_pngs) > 0

    # 4) Rebuild index only when PNG exists but JSONL is missing.
    if png_exists and (not os.path.exists(jsonl_path)):
        print(f"[reindex] PNGs detected in {out_split_dir}, but no JSONL. Rebuilding index only...")
        json_objs = []
        worker = partial(_worker_from_png, out_split_dir=out_split_dir, split_root=split_root, label_map=label_map)
        # No initializer needed; each worker self-seeds deterministically.
        with Pool(processes=NUM_WORKERS) as pool:
            for ok, rec in tqdm(pool.imap_unordered(worker, existing_pngs, chunksize=16),
                                total=len(existing_pngs), desc=f"{split_name}-reindex"):
                if ok and rec:
                    json_objs.append(rec)
        _append_jsonl(jsonl_path, json_objs)
        print(f"[done-reindex] {split_name}: JSONL -> {jsonl_path}")
        return

    # 5) Regular flow: generate PNG and index from pcap files.
    pcaps = list_pcaps_under_split(split_root)
    print(f"[scan] {split_root}")
    print(f"[scan] pcaps={len(pcaps)} -> output={out_split_dir}")
    if not pcaps:
        # No pcap: skip when PNG and JSONL are already present.
        if png_exists and os.path.exists(jsonl_path):
            print(f"[skip] No pcaps; PNG & JSONL already present.")
            return
        # No pcap and no PNG: nothing to process.
        print(f"[warn] No pcaps found and no PNGs to index. Skipped.")
        return

    worker = partial(_worker_from_pcap, out_split_dir=out_split_dir, split_root=split_root, label_map=label_map)

    ok = 0
    json_objs = []
    with Pool(processes=NUM_WORKERS) as pool:
        for res_ok, rec in tqdm(pool.imap_unordered(worker, pcaps, chunksize=8),
                                total=len(pcaps), desc=split_name):
            if res_ok:
                ok += 1
                if rec is not None:
                    json_objs.append(rec)

    _append_jsonl(jsonl_path, json_objs)
    print(f"[done] {split_name}: {ok}/{len(pcaps)} pngs -> {out_split_dir}")
    print(f"[map ] JSONL -> {jsonl_path}")


def main():
    dataset_name = os.path.basename(DATASET_ROOT.rstrip("/"))
    model_out_root = os.path.join(BASE_OUTPUT_ROOT, MODEL, dataset_name)
    ensure_dir(model_out_root)
    label_map_path = os.path.join(model_out_root, "label_map.json")

    t0 = time.time()
    for split_root in INPUT_DIRS:
        if not os.path.isdir(split_root):
            print(f"[skip] not found: {split_root}")
            continue
        process_one_split(split_root, model_out_root, label_map_path)
    print(f"[all done] output: {model_out_root} | elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
