# packet_stage/pkt_materialize.py

from pathlib import Path
from collections import Counter
from tqdm import tqdm
from utils.io_utils import ensure_dir, bytes_of_file, materialize_file

def materialize_packet_dataset(packet_root, final_selected):
    """
    Materialize OUT_PACKET/train|valid|test/<class>/xxx.pcap
    and return manifest structure.
    """

    manifests = {"train": [], "valid": [], "test": []}

    for c, split_map in final_selected.items():
        for split_name, pkts in split_map.items():
            out_dir = packet_root / split_name / c
            ensure_dir(out_dir)

            for p in tqdm(pkts, desc=f"Materialize {split_name}-{c}",
                          ncols=100, leave=False):
                fname = p.name
                dst = out_dir / fname

                if not dst.exists():
                    materialize_file(p, dst)

                manifests[split_name].append({
                    "level": "packet",
                    "class": c,
                    "path": str(dst.relative_to(packet_root)),
                    "bytes": bytes_of_file(dst)
                })

    return manifests


def write_packet_stats(packet_root, manifests, seed):
    lists_dir = packet_root / "lists"
    ensure_dir(lists_dir)

    def write_list(name, arr):
        with open(lists_dir / f"{name}.txt", "w") as f:
            for m in arr:
                f.write(m["path"] + "\n")

    write_list("train", manifests["train"])
    write_list("valid", manifests["valid"])
    write_list("test", manifests["test"])

    stats = {
        "seed": seed,
        "counts": {
            "train": len(manifests["train"]),
            "valid": len(manifests["valid"]),
            "test":  len(manifests["test"]),
            "total": (len(manifests["train"]) +
                      len(manifests["valid"]) +
                      len(manifests["test"]))
        },
        "label_dist": {
            "train": Counter([m["class"] for m in manifests["train"]]),
            "valid": Counter([m["class"] for m in manifests["valid"]]),
            "test":  Counter([m["class"] for m in manifests["test"]])
        }
    }

    import json
    with open(packet_root / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    with open(packet_root / "manifest.jsonl", "w") as f:
        for split in ["train", "valid", "test"]:
            for m in manifests[split]:
                mm = dict(m)
                mm["split"] = split
                f.write(json.dumps(mm) + "\n")
