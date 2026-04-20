# packet_stage/pkt_from_flow.py

import math
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm

from utils.splitcap_utils import splitcap_packets
from utils.pcap_utils import rdpcap_first_pkt_info
from utils.io_utils import ensure_dir

def worker_split_and_filter(args):
    """
    Args:
        (split_name, cls_name, flow_path, tmp_root, tcp_min, udp_min, exe)
    """
    (split_name, cls_name, flow_str, tmp_root, tcp_min, udp_min, exe) = args

    flow_pcap = Path(flow_str)
    flow_id = flow_pcap.stem
    out_dir = Path(tmp_root) / split_name / cls_name / flow_id

    ensure_dir(out_dir)

    # split packets
    if not out_dir.exists() or not any(out_dir.iterdir()):
        splitcap_packets(exe, flow_pcap, out_dir)

    kept = []
    for pkt_pcap in out_dir.glob("*.pcap"):
        if pkt_pcap.stat().st_size == 0:
            continue
        proto, L, cnt = rdpcap_first_pkt_info(pkt_pcap)
        if cnt == 0:
            continue
        if proto not in ("tcp", "udp"):
            continue
        if proto == "tcp" and L < tcp_min:
            continue
        if proto == "udp" and L < udp_min:
            continue
        kept.append(pkt_pcap)

    return split_name, cls_name, kept


def split_packets_by_flow(flow_root, tmp_pkt_root, exe, tcp_min, udp_min, workers):
    """
    Return:
    per_class_split_packets[class_name][split] = [pkt_pcap...]
    """
    result = {}
    split_names = ["train", "valid", "test"]

    # gather task list
    tasks = []
    for split_name in split_names:
        split_dir = flow_root / split_name
        if not split_dir.exists():
            continue

        for cls_dir in sorted([d for d in split_dir.iterdir() if d.is_dir()]):
            cls = cls_dir.name
            for flow_pcap in sorted(cls_dir.glob("*.pcap")):
                tasks.append(
                    (split_name, cls, str(flow_pcap),
                     str(tmp_pkt_root), tcp_min, udp_min, exe)
                )

    if not tasks:
        print("[Packet] No flow pcaps found. Did flow stage run?")
        return {}

    cs = max(1, math.ceil(len(tasks) / workers))
    print(f"[Packet] Split per-flow packets: tasks={len(tasks)} workers={workers} chunksize={cs}")

    result = {}
    with Pool(workers) as pool, tqdm(total=len(tasks),
                                     desc="SplitCap packets (from flow)",
                                     ncols=100, leave=False) as pbar:
        for sp, cls, kept_list in pool.imap_unordered(worker_split_and_filter, tasks, chunksize=cs):
            result.setdefault(cls, {}).setdefault(sp, []).extend(kept_list)
            pbar.update(1)

    return result
