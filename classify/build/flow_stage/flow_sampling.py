# flow_stage/flow_sampling.py

from collections import defaultdict
from utils.sampling import maxmin_fair_sample

def apply_per_class_sampling(valid_by_class, flow_class_min, flow_class_cap, seed):
    kept = {}
    overflow = {}

    for c, files in valid_by_class.items():
        n = len(files)
        print(f"[Flow] class {c}: valid_flows={n}")

        if n < flow_class_min:
            print(f"  -> dropped (< FLOW_CLASS_MIN={flow_class_min})")
            continue

        if n > flow_class_cap:
            # build src index
            src_map = defaultdict(list)
            for i, p in enumerate(files):
                src_map[p.parent.name].append(i)

            sel_idx = maxmin_fair_sample(src_map, flow_class_cap, seed=seed)
            kept[c] = [files[i] for i in sel_idx]

            overflow_idx = sorted(set(range(n)) - set(sel_idx))
            overflow[c] = [files[i] for i in overflow_idx]

        else:
            print("  -> within cap, keep all.")
            kept[c] = files
            overflow[c] = []

    return kept, overflow
