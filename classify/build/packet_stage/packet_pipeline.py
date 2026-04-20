# packet_stage/packet_pipeline.py

from pathlib import Path
import shutil
import random

from .pkt_from_flow import split_packets_by_flow
from .pkt_target_calc import compute_target
from .pkt_sampling import sample_packets_by_target
from .pkt_expand_test import expand_test_if_all_cap
from .pkt_materialize import materialize_packet_dataset, write_packet_stats

from utils.io_utils import ensure_dir


def run_packet_stage(config):
    seed = config["SEED"]
    random.seed(seed)

    flow_root = Path(config["OUT_FLOW"]).resolve()
    packet_root = Path(config["OUT_PACKET"]).resolve()
    ensure_dir(packet_root)

    tmp_pkt_root = packet_root / "tmp_pkt_split"
    ensure_dir(tmp_pkt_root)

    # 1) split packets from flow files
    per_class_split_pkts = split_packets_by_flow(
        flow_root=flow_root,
        tmp_pkt_root=tmp_pkt_root,
        exe=config["SPLITCAP_EXE"],
        tcp_min=config["TCP_MIN_L3"],
        udp_min=config["UDP_MIN_L3"],
        workers=config["PACKET_SPLIT_WORKERS"]
    )

    ratios = config["PKT_SPLIT_RATIOS"]
    pkt_min = config["PKT_CLASS_MIN"]
    pkt_cap = config["PKT_CLASS_CAP"]

    final_selected = {}
    overflow_dict  = {}

    # 2) For each class, compute Target and sample
    for cls, buckets in per_class_split_pkts.items():

        P_train = len(buckets.get("train", []))
        P_valid = len(buckets.get("valid", []))
        P_test  = len(buckets.get("test", []))

        print(f"[Packet] class {cls}: P_train={P_train} P_valid={P_valid} P_test={P_test}")

        Target = compute_target(P_train, P_valid, P_test,
                                ratios, pkt_min, pkt_cap)

        if Target == 0:
            print(f"  -> dropped (Target=0)")
            continue

        selected, overflow = sample_packets_by_target(
            cls_name=cls,
            buckets=buckets,
            Target=Target,
            ratios=ratios,
            seed=seed
        )

        if selected is None:
            print(f"  -> dropped (cannot meet Target)")
            continue

        final_selected[cls] = selected
        overflow_dict[cls] = overflow

        print(f"  -> selected_total={Target} train={len(selected['train'])} "
              f"valid={len(selected['valid'])} test={len(selected['test'])}")

    # 3) expand test if all classes reach cap
    final_selected = expand_test_if_all_cap(final_selected, overflow_dict, pkt_cap, seed)

    # 4) materialize
    manifests = materialize_packet_dataset(packet_root, final_selected)

    # 5) write stats
    write_packet_stats(packet_root, manifests, seed)

    # 6) cleanup tmp
    print("[Packet] Cleaning up tmp_pkt_split ...")
    shutil.rmtree(tmp_pkt_root, ignore_errors=True)
