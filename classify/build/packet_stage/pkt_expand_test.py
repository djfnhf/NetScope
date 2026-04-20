import random

def expand_test_if_all_cap(final_selected, overflow_dict, pkt_cap, seed=42, max_expand_ratio=10.0):
    """
    final_selected[class] = {"train":[], "valid":[], "test":[]}
    overflow_dict[class] = [pkts...]

    Expand test split from overflow only when all classes reach cap.
    Test expansion is capped by train_count * max_expand_ratio.
    """

    for c in final_selected:
        total = (len(final_selected[c]["train"])
                 + len(final_selected[c]["valid"])
                 + len(final_selected[c]["test"]))
        if total != pkt_cap:
            return final_selected

    # Count available overflow items.
    counts = [len(overflow_dict[c]) for c in final_selected if len(overflow_dict[c]) > 0]
    if not counts:
        return final_selected

    min_overflow = min(counts)
    print(f"[Packet] All classes reach cap={pkt_cap}, initial expand by {min_overflow}")

    rnd = random.Random(seed)

    for c in final_selected:
        pool = overflow_dict.get(c, [])
        if len(pool) == 0:
            continue

        train_len = len(final_selected[c]["train"])
        test_len = len(final_selected[c]["test"])
        max_allow = int(train_len * max_expand_ratio)

        # Remaining expansion capacity.
        remain_space = max(0, max_allow - test_len)
        if remain_space == 0:
            print(f"  [Limit] class {c}: test already {test_len}, skip (max={max_allow})")
            continue

        take_n = min(min_overflow, remain_space, len(pool))
        if take_n <= 0:
            continue

        extra = rnd.sample(pool, take_n)
        final_selected[c]["test"].extend(extra)

        print(f"  [Expand] class {c}: +{take_n} extra test packets (new total={len(final_selected[c]['test'])}, max={max_allow})")

    return final_selected
