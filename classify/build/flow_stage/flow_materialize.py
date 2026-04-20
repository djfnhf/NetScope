# flow_stage/flow_materialize.py

from pathlib import Path
from collections import Counter
from tqdm import tqdm
import json

from utils.io_utils import ensure_dir, materialize_file
from utils.hashing import short_hash


def materialize_flow_dataset(root, train, valid, test):
    """
    Copy/link flow pcaps into:
        OUT_FLOW/{train,valid,test}/<class>/...
    Then update manifest["path"] to the materialized relative path.
    """

    for split, arr in [("train", train), ("valid", valid), ("test", test)]:
        split_dir = root / split
        ensure_dir(split_dir)

        for m in tqdm(arr, desc=f"Materialize {split}", ncols=100, leave=False):
            cls = m["class"]

            # src inside OUT_FLOW/split/flow tree
            src_rel = m["path"]
            src = (root / src_rel).resolve()

            # ---- Safety Check ----
            if not src.exists():
                raise FileNotFoundError(
                    f"[Flow-Materialize] Missing file: {src} "
                    f"(manifest path = {src_rel})"
                )

            dst_dir = split_dir / cls
            ensure_dir(dst_dir)

            fname = src.name
            dst = dst_dir / fname

            # Avoid filename conflict: use hash of RELATIVE PATH
            if dst.exists():
                stem = Path(fname).stem
                suf = "".join(Path(fname).suffixes)
                # Use relative path for reproducible hashing
                hashed = short_hash(src_rel)
                dst = dst_dir / f"{stem}_h{hashed}{suf}"

            if not dst.exists():
                materialize_file(src, dst)

            # Update manifest path to materialized relative path
            m["path"] = str(dst.relative_to(root))


def write_flow_stats_lists(root, train, valid, test, seed):
    """
    Write:
        lists/train.txt, lists/valid.txt, lists/test.txt
        stats.json
        manifest.jsonl
    """

    lists_dir = root / "lists"
    ensure_dir(lists_dir)

    # Write split list files
    def write_list(name, arr):
        with open(lists_dir / f"{name}.txt", "w") as f:
            for m in arr:
                f.write(m["path"] + "\n")

    write_list("train", train)
    write_list("valid", valid)
    write_list("test",  test)

    # Stats
    label_train = Counter([m["class"] for m in train])
    label_valid = Counter([m["class"] for m in valid])
    label_test  = Counter([m["class"] for m in test])

    # total per class (for debugging)
    class_total = {}
    for c in set(label_train) | set(label_valid) | set(label_test):
        class_total[c] = (
            label_train.get(c, 0)
            + label_valid.get(c, 0)
            + label_test.get(c, 0)
        )

    stats = {
        "seed": seed,
        "counts": {
            "train": len(train),
            "valid": len(valid),
            "test":  len(test),
            "total": len(train) + len(valid) + len(test),
        },
        "label_dist": {
            "train": label_train,
            "valid": label_valid,
            "test":  label_test,
            "class_total": class_total
        }
    }

    with open(root / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # manifest.jsonl
    with open(root / "manifest.jsonl", "w") as f:
        for tag, arr in [("train", train), ("valid", valid), ("test", test)]:
            for m in arr:
                mm = dict(m)
                mm["split"] = tag
                f.write(json.dumps(mm) + "\n")
