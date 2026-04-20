# utils/sampling.py

import random
from collections import defaultdict

# ------------------ Flow: max-min fair sampling ------------------

def maxmin_fair_sample(index_map, cap, seed=42):
    """
    index_map: {src_tag: [idx1, idx2, ...]}
    cap: max total
    """
    rng = random.Random(seed)

    sizes = {src: len(v) for src, v in index_map.items()}
    total = sum(sizes.values())

    if total <= cap:
        return [i for lst in index_map.values() for i in lst]

    alloc = {src: 0 for src in sizes}
    remaining = cap

    # initial: give one to each source if possible
    active = [s for s in sizes if sizes[s] > 0]
    if remaining >= len(active):
        for s in active:
            alloc[s] = 1
        remaining -= len(active)
    else:
        rng.shuffle(active)
        for s in active[:remaining]:
            alloc[s] = 1
        remaining = 0

    active = [s for s in sizes if alloc[s] < sizes[s]]

    # water filling
    while remaining > 0 and active:
        k = len(active)
        per = remaining // k
        if per > 0:
            for s in list(active):
                take = min(per, sizes[s] - alloc[s])
                alloc[s] += take
                remaining -= take
                if alloc[s] >= sizes[s]:
                    active.remove(s)
        else:
            rng.shuffle(active)
            for s in active:
                if remaining == 0:
                    break
                if alloc[s] < sizes[s]:
                    alloc[s] += 1
                    remaining -= 1
            active = [x for x in active if alloc[x] < sizes[x]]

    # build selected
    selected = []
    for src, idxs in index_map.items():
        k = alloc[src]
        if k > 0:
            selected.extend(random.Random(seed).sample(idxs, k))
    return selected


# ------------------ Packet: split (train/valid/test) ------------------

def packet_split_random(pkts, ratios, seed=42):
    """
    pkts: list
    ratios: (0.8,0.1,0.1)
    """
    rnd = random.Random(seed)
    rnd.shuffle(pkts)

    n = len(pkts)
    r_train, r_valid, r_test = ratios
    nt = int(round(n * r_train))
    nv = int(round(n * r_valid))
    return pkts[:nt], pkts[nt:nt+nv], pkts[nt+nv:]


# ------------------ Packet: overflow test expansion ------------------

def expand_test_split(final_map, overflow_map, cap, seed=42):
    """
    final_map[class] = {"train":[...], "valid":[...], "test":[...]}
    overflow_map[class] = list of extra pkts
    cap = PKT_CLASS_CAP
    """
    # condition: all classes reached cap
    all_cap = all(len(final_map[c]["train"]) +
                  len(final_map[c]["valid"]) +
                  len(final_map[c]["test"]) >= cap
                  for c in final_map)

    if not all_cap:
        return final_map

    # compute per-class remaining overflow
    remain_counts = [len(overflow_map[c])
                     for c in final_map
                     if len(overflow_map[c]) > 0]

    if not remain_counts:
        return final_map

    min_overflow = min(remain_counts)

    print(f"[Packet] Expand test: add {min_overflow} per class")

    rnd = random.Random(seed)

    for c in final_map:
        pool = overflow_map[c]
        if len(pool) == 0:
            continue
        extra = rnd.sample(pool, min_overflow)
        final_map[c]["test"].extend(extra)

    return final_map
