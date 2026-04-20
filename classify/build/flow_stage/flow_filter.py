# flow_stage/flow_filter.py

import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import math
from multiprocessing import Pool

from utils.io_utils import bytes_of_file
from utils.pcap_utils import fast_count_packets

def gather_flow_candidates(split_dir, flow_cap, seed):

    selected_by_src = {}
    all_flows = []

    for cls in sorted([d for d in split_dir.iterdir() if d.is_dir()]):
        for src_dir in sorted([d for d in cls.iterdir() if d.is_dir()]):
            flows = sorted(src_dir.glob("*.pcap"))
            if not flows:
                continue

            soft_cap = flow_cap * 2
            if len(flows) > soft_cap:
                rnd = random.Random(seed)
                rnd.shuffle(flows)
                flows = flows[:soft_cap]

            selected_by_src.setdefault(cls.name, {})[src_dir.name] = flows
            all_flows.extend(flows)

    return selected_by_src, all_flows


def count_all_packets(flow_paths, workers):
    tasks = [str(p) for p in flow_paths]
    cs = max(1, math.ceil(len(tasks) / workers))

    pkt_cnt_map = {}
    with Pool(workers) as pool:
        for p, cnt in tqdm(
                pool.imap_unordered(fast_count_packets, tasks, chunksize=cs),
                total=len(tasks),
                desc="count_pkts (all classes)",
                ncols=100,
                leave=False):
            pkt_cnt_map[p] = cnt

    return pkt_cnt_map


def filter_valid_flows(selected_by_src, pkt_cnt_map, min_bytes, min_pkts):
    """
    Return:
       valid_by_class[class] = [Path...]
    """
    valid_by_class = defaultdict(list)

    for cls_name, src_map in selected_by_src.items():
        for src_name, flow_list in src_map.items():
            for p in flow_list:
                p_str = str(p)
                if bytes_of_file(p) < min_bytes:
                    continue
                if pkt_cnt_map.get(p_str, 0) < min_pkts:
                    continue
                valid_by_class[cls_name].append(p)

    return valid_by_class
