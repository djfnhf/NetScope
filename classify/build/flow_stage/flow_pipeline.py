# flow_stage/flow_pipeline.py

from pathlib import Path
from collections import defaultdict

from .flow_split import perform_session_split
from .flow_filter import gather_flow_candidates, count_all_packets, filter_valid_flows
from .flow_sampling import apply_per_class_sampling
from .flow_split_assign import stratified_split_flow
from .flow_materialize import materialize_flow_dataset, write_flow_stats_lists

from utils.io_utils import ensure_dir


def run_flow_stage(config):
    """
    Top-level Flow Pipeline.
    """

    seed = config["SEED"]
    exe = config["SPLITCAP_EXE"]

    # Use IN_ROOT_BASE instead of deprecated IN_ROOT_FLOW_SRC.
    in_root = Path(config["IN_ROOT_BASE"]).resolve()

    out_root = Path(config["OUT_FLOW"]).resolve()
    ensure_dir(out_root)

    # 1) Session split
    flow_split_root = out_root / "split" / "flow"
    ensure_dir(flow_split_root)

    class_dirs = [d for d in in_root.iterdir() if d.is_dir()]
    perform_session_split(
        class_dirs=class_dirs,
        out_root=flow_split_root,
        exe=exe,
        workers=config["FLOW_SPLIT_WORKERS"],
    )

    # 2) gather & count packets
    selected_by_src, all_flow_files = gather_flow_candidates(
        split_dir=flow_split_root,
        flow_cap=config["FLOW_CLASS_CAP"],
        seed=seed
    )

    if not all_flow_files:
        print("[Flow] No candidate flows.")
        return

    pkt_cnt_map = count_all_packets(all_flow_files, workers=config["WORKERS"])

    # 3) filter flows
    valid_by_class = filter_valid_flows(
        selected_by_src,
        pkt_cnt_map,
        min_bytes=config["FLOW_MIN_BYTES"],
        min_pkts=config["FLOW_MIN_PKTS"],
    )

    # 4) per-class sampling
    kept, overflow = apply_per_class_sampling(
        valid_by_class,
        flow_class_min=config["FLOW_CLASS_MIN"],
        flow_class_cap=config["FLOW_CLASS_CAP"],
        seed=seed
    )

    # 5) manifest
    manifest = []
    for c, files in kept.items():
        for p in files:
            manifest.append({
                "level": "flow",
                "path": str(p.relative_to(out_root)),
                "class": c,
                "bytes": p.stat().st_size,
                "pkts": pkt_cnt_map.get(str(p), 0),
                "source_tag": p.parent.name
            })
    
    # 6) split
    train, valid, test = stratified_split_flow(
        manifest=manifest,
        ratios=config["FLOW_SPLIT_RATIOS"],
        overflow_pool=overflow,
        out_root=out_root,
        pkt_cnt_map=pkt_cnt_map,
        cap=config["FLOW_CLASS_CAP"],
        seed=seed
    )

    # 7) materialize
    materialize_flow_dataset(out_root, train, valid, test)
    write_flow_stats_lists(out_root, train, valid, test, seed)

    # 8) cleanup
    import shutil
    print("[Flow] Cleanup split/flow ...")
    shutil.rmtree(flow_split_root, ignore_errors=True)
