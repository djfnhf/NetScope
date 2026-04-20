#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

# Ensure current directory is searched first.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import argparse
import json
import multiprocessing as mp

# ========== Load CONFIG ==========
from config import CONFIG

# ========== Flow & Packet pipelines ==========
from flow_stage.flow_pipeline import run_flow_stage
from packet_stage.packet_pipeline import run_packet_stage


# ============================================================
# CLI parsing
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser("TrafficBench Dataset Builder")

    ap.add_argument(
        "--in-root",
        type=str,
        help="One or more input roots. Example: '/data/cstnet,/data/CICIoT2023'"
    )
    ap.add_argument("--workers", type=int)
    ap.add_argument("--seed", type=int)

    # flow thresholds
    ap.add_argument("--flow-min-bytes", type=int)
    ap.add_argument("--flow-min-pkts", type=int)
    ap.add_argument("--flow-class-min", type=int)
    ap.add_argument("--flow-class-cap", type=int)

    # packet thresholds
    ap.add_argument("--tcp-min", type=int)
    ap.add_argument("--udp-min", type=int)
    ap.add_argument("--pkt-class-min", type=int)
    ap.add_argument("--pkt-class-cap", type=int)

    # toggles
    ap.add_argument("--no-flow", action="store_true")
    ap.add_argument("--no-packet", action="store_true")

    return ap.parse_args()


# ============================================================
# Apply CLI overrides
# ============================================================

def apply_overrides(args):
    if args.in_root:
        CONFIG["IN_ROOT_BASE"] = [x.strip() for x in args.in_root.split(",")]

    if args.workers:
        CONFIG["WORKERS"] = args.workers
    if args.seed:
        CONFIG["SEED"] = args.seed

    # Flow parameters
    if args.flow_min_bytes:
        CONFIG["FLOW_MIN_BYTES"] = args.flow_min_bytes
    if args.flow_min_pkts:
        CONFIG["FLOW_MIN_PKTS"] = args.flow_min_pkts
    if args.flow_class_min:
        CONFIG["FLOW_CLASS_MIN"] = args.flow_class_min
    if args.flow_class_cap:
        CONFIG["FLOW_CLASS_CAP"] = args.flow_class_cap

    # Packet parameters
    if args.tcp_min:
        CONFIG["TCP_MIN_L3"] = args.tcp_min
    if args.udp_min:
        CONFIG["UDP_MIN_L3"] = args.udp_min
    if args.pkt_class_min:
        CONFIG["PKT_CLASS_MIN"] = args.pkt_class_min
    if args.pkt_class_cap:
        CONFIG["PKT_CLASS_CAP"] = args.pkt_class_cap

    # toggles
    if args.no_flow:
        CONFIG["DO_FLOW_STAGE"] = False
    if args.no_packet:
        CONFIG["DO_PACKET_FROM_FLOW"] = False


# ============================================================
# Main pipeline
# ============================================================

def main():
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass

    args = parse_args()
    apply_overrides(args)

    # Support IN_ROOT_BASE as either str or list.
    roots = CONFIG["IN_ROOT_BASE"]
    if isinstance(roots, str):
        roots = [roots]

    for root in roots:
        CONFIG["IN_ROOT_BASE"] = root
        base = Path(root).resolve()

        # Auto-generate output paths.
        CONFIG["OUT_FLOW"] = str(base.with_name(base.name + "_flow"))
        CONFIG["OUT_PACKET"] = str(base.with_name(base.name + "_packet"))

        print("\n" + "=" * 70)
        print(f" Processing dataset root: {root}")
        print("=" * 70 + "\n")

        print("===== Effective CONFIG =====")
        print(json.dumps(CONFIG, indent=2, ensure_ascii=False))
        print("============================\n")

        # Flow stage
        if CONFIG["DO_FLOW_STAGE"]:
            run_flow_stage(CONFIG)

        # Packet stage
        if CONFIG["DO_PACKET_FROM_FLOW"]:
            run_packet_stage(CONFIG)

        print(f"\n Finished dataset: {root}\n")

    print("\nAll datasets finished.\n")


if __name__ == "__main__":
    main()
