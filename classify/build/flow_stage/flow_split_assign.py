import random
from collections import defaultdict
from utils.io_utils import bytes_of_file

def stratified_split_flow(manifest, ratios, overflow_pool, out_root, pkt_cnt_map,
                          cap, seed, max_expand_ratio=10.0):
    """
    Flow-level stratified split with optional test expansion from overflow.
    Test expansion is capped by train_count * max_expand_ratio.
    """

    rng = random.Random(seed)

    # group by class index
    by_cls = defaultdict(list)
    for i, m in enumerate(manifest):
        by_cls[m["class"]].append(i)

    kept = {}

    train_r, valid_r, test_r = ratios

    # 1. normal split
    for c, idxs in by_cls.items():
        rng.shuffle(idxs)

        n = len(idxs)
        n_train = int(round(n * train_r))
        n_valid = int(round(n * valid_r))
        n_test  = n - n_train - n_valid

        kept[c] = {
            "train": idxs[:n_train],
            "valid": idxs[n_train:n_train + n_valid],
            "test":  idxs[n_train + n_valid:]
        }

    # 2. expand only if all classes reach the HARD cap
    all_reach_cap = all(len(by_cls[c]) >= cap for c in kept)

    if all_reach_cap:
        valid_overflows = [len(overflow_pool[c]) for c in kept if len(overflow_pool[c]) > 0]
        if valid_overflows:
            m = min(valid_overflows)
            print(f"[Split] All classes reach cap={cap}, initial expand by {m}.")

            for c in kept:
                pool = overflow_pool.get(c, [])
                if len(pool) == 0:
                    continue

                train_len = len(kept[c]["train"])
                test_len = len(kept[c]["test"])
                max_allow = int(train_len * max_expand_ratio)
                remain_space = max(0, max_allow - test_len)

                if remain_space == 0:
                    print(f"  [Limit] class {c}: test already {test_len}, skip (max={max_allow})")
                    continue

                take_n = min(m, remain_space, len(pool))
                if take_n <= 0:
                    continue

                extra = rng.sample(pool, take_n)
                for p in extra:
                    manifest.append({
                        "level": "flow",
                        "path": str(p.relative_to(out_root)),
                        "class": c,
                        "bytes": bytes_of_file(p),
                        "pkts": pkt_cnt_map.get(str(p), 0),
                        "source_tag": p.parent.name
                    })
                    kept[c]["test"].append(len(manifest) - 1)

                print(f"  [Expand] class {c}: +{take_n} extra test flows (new total={len(kept[c]['test'])}, max={max_allow})")

    train = [manifest[i] for c in kept for i in kept[c]["train"]]
    valid = [manifest[i] for c in kept for i in kept[c]["valid"]]
    test  = [manifest[i] for c in kept for i in kept[c]["test"]]

    return train, valid, test
