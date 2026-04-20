#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robust Shadowset Builder (combo-enabled + label noise + scope control)
- Input: test_root/class/*.pcap
- Code mapping: 1=reorder, 2=retransmit, 3=jitter_drop, 4=frag_merge, 5=time_scale, 6=label_noise
- Scope: support "first N packets only" or "all packets in flow".
- Multiple perturbations can be chained in order and exported in one run directory.
- Implementation notes:
  * Packet-level perturbations (1~5) are applied in order on selected scope only.
  * When only first N packets are perturbed, tail timestamps are shifted to keep monotonic order.
  * Label noise (6) is always executed after packet perturbations.
- Output:
  - Single perturbation: out_root/<run_name><perturb>_<level>/<class>/<file>.pcap
  - Combo perturbation: out_root/<run_name>COMBO(<codes>)_<level>/<class>/<file>.pcap
  - manifest.jsonl with per-file result records and optional label-noise summary.
"""

import os, sys, json, copy, random, shutil, traceback
from glob import glob
from typing import List, Tuple, Dict, Any, DefaultDict
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import scapy.all as scapy
from scapy.layers.inet import IP, TCP, UDP
from scapy.packet import Packet, Raw

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_path(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)


# ===================== 0) User configuration =====================
CONFIG = {
    # Data paths
    "test_root": "datasets/cstnet_120_flow/test",
    "out_root":  "datasets/cstnet_120_flow/rtest",

    # Runtime controls
    "overwrite": True,
    "workers": 8,
    "global_seed": 42,
    "pcap_glob": "*.pcap",

    # Perturbation mode and level code
    # 1=reorder, 2=retransmit, 3=jitter_drop, 4=frag_merge, 5=time_scale, 6=label_noise
    "mode_codes": [1,2,3,4,5],
    #[]
    "level_code": 2,

    # Perturbation scope
    # target: "all" or "first_n"
    # n is used only when target == "first_n"
    "scope": {
        "target": "all",      # "all" or "first_n"
        "n": 5
    },

    # Optional output prefix.
    "run_name": "expD_",
}

# ===================== Mode mapping and description =====================
PERTURB_CODE = {1:"reorder",2:"retransmit",3:"jitter_drop",4:"frag_merge",5:"time_scale",6:"label_noise"}
LEVEL_CODE   = {1:"light",2:"medium",3:"heavy"}

MODE_DESCRIPTIONS = {
    1: "Local reorder in a small window.",
    2: "Retransmission-like duplicate insertion.",
    3: "IAT jitter with small random packet drop.",
    4: "Fragment and merge operations on payload bytes.",
    5: "Global timeline scaling while preserving order.",
    6: "Symmetric random label flip with rate epsilon.",
}

# ===================== 1) Perturbation parameters by level =====================
PARAMS = {
    "reorder": {
        "light":  {"window": 4,  "swap_prob": 0.15, "max_swaps_ratio": 0.10},
        "medium": {"window": 8,  "swap_prob": 0.35, "max_swaps_ratio": 0.20},
        "heavy":  {"window": 16, "swap_prob": 0.70, "max_swaps_ratio": 0.40},
    },
    "retransmit": {
        "light":  {"insert_ratio": 0.02, "nearby_span": 3},
        "medium": {"insert_ratio": 0.06, "nearby_span": 5},
        "heavy":  {"insert_ratio": 0.12, "nearby_span": 8},
    },
    "jitter_drop": {
        "light":  {"jitter_std": 0.003, "drop_prob": 0.010},
        "medium": {"jitter_std": 0.010, "drop_prob": 0.030},
        "heavy":  {"jitter_std": 0.030, "drop_prob": 0.080},
    },
    "frag_merge": {
        "light":  {"frag_prob": 0.10, "merge_prob": 0.10, "merge_max_len": 400, "min_fragment": 32},
        "medium": {"frag_prob": 0.20, "merge_prob": 0.20, "merge_max_len": 600, "min_fragment": 24},
        "heavy":  {"frag_prob": 0.35, "merge_prob": 0.35, "merge_max_len": 800, "min_fragment": 16},
    },
    "time_scale": {
        "light":  {"scale": 0.8},
        "medium": {"scale": 0.6},
        "heavy":  {"scale": 0.4},
    },
    "label_noise": {
        "light":  {"epsilon": 0.01},
        "medium": {"epsilon": 0.05},
        "heavy":  {"epsilon": 0.10},
    },
}


# ===================== 2) Utility helpers =====================
def set_seeds(seed: int):
    random.seed(seed); np.random.seed(seed)

def makedirs(p): os.makedirs(p, exist_ok=True)

def list_pcap_files(root: str, pattern: str) -> List[Tuple[str, str]]:
    ret = []
    if not os.path.isdir(root): return ret
    for cls in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))):
        for fp in sorted(glob(os.path.join(root, cls, pattern))):
            ret.append((cls, fp))
    return ret

def clone_head(pkt: Packet) -> Packet:
    layers = []; cur = pkt
    while cur:
        if isinstance(cur, Raw): break
        layers.append(cur.copy()); cur = cur.payload
    for i in range(len(layers)-1):
        layers[i].remove_payload(); layers[i] /= layers[i+1]
    head = layers[0] if layers else Packet()
    head.remove_payload(); return head

def safe_build(pkt: Packet) -> bytes:
    ip = pkt.getlayer(IP)
    if ip is not None: ip.len = None; ip.chksum = None
    tcp = pkt.getlayer(TCP)
    if tcp is not None: tcp.chksum = None
    udp = pkt.getlayer(UDP)
    if udp is not None: udp.len = None; udp.chksum = None
    return bytes(pkt)

def write_pcap(packets: List[Packet], out_path: str):
    built = []
    for p in packets:
        try: _ = safe_build(p)
        finally: built.append(p)
    scapy.wrpcap(out_path, built)

def five_tuple(pkt: Packet):
    ip = pkt.getlayer(IP)
    if ip is None: return None
    l4 = pkt.getlayer(TCP) or pkt.getlayer(UDP)
    if l4 is None: return (ip.src, ip.dst, None, None, 'IP')
    proto = 'TCP' if isinstance(l4, TCP) else 'UDP'
    return (ip.src, ip.dst, l4.sport, l4.dport, proto)

def iats_from(packets: List[Packet]) -> List[float]:
    ts = [float(p.time) for p in packets]
    out = [0.0]
    for i in range(1, len(ts)):
        out.append(max(0.0, ts[i] - ts[i - 1]))
    return out

def apply_iats(packets: List[Packet], iats: List[float]):
    if not packets: return
    t0 = float(packets[0].time); cur = t0
    packets[0].time = t0
    for i in range(1, len(packets)):
        cur += max(0.0, iats[i]); packets[i].time = cur

# ===================== 3) Perturbation implementations (1~5) =====================
def p_reorder(pkts, cfg, rng):
    n = len(pkts); idx = list(range(n))
    window = int(cfg["window"]); swap_prob = float(cfg["swap_prob"])
    max_swaps = int(cfg["max_swaps_ratio"] * max(1, n))
    swaps = 0
    for i in range(n):
        if swaps >= max_swaps: break
        if rng.random() < swap_prob:
            j = min(n-1, i + rng.randint(1, max(1, window-1)))
            idx[i], idx[j] = idx[j], idx[i]; swaps += 1
    return [pkts[k] for k in idx]

def p_retransmit(pkts, cfg, rng):
    n = len(pkts)
    if n == 0: return pkts
    ratio = float(cfg["insert_ratio"]); span = int(cfg["nearby_span"])
    k = int(round(ratio * n)); out = pkts.copy()
    for _ in range(k):
        src = rng.randint(0, len(out) - 1)
        dup = copy.deepcopy(out[src])
        pos = min(len(out), max(0, src + rng.randint(-span, span)))
        if 0 < pos < len(out): t = (float(out[pos-1].time) + float(out[pos].time)) / 2.0
        elif pos == 0: t = float(out[0].time) - 1e-6
        else: t = float(out[-1].time) + 1e-6
        dup.time = t; out.insert(pos, dup)
    out.sort(key=lambda p: float(p.time))
    return out

def p_jitter_drop(pkts, cfg, rng):
    if len(pkts) <= 1: return pkts
    std = float(cfg["jitter_std"]); drop = float(cfg["drop_prob"])
    kept = [p for p in pkts if rng.random() > drop] or [pkts[0]]
    iats = iats_from(kept)
    noisy = [max(0.0, i + rng.gauss(0.0, std)) for i in iats]
    apply_iats(kept, noisy); return kept

def _split_payload(b: bytes, minfrag: int, rng) -> List[bytes]:
    if len(b) < 2 * minfrag: return [b]
    cut = rng.randint(minfrag, len(b) - minfrag)
    return [b[:cut], b[cut:]]

def _head_with_payload(origin: Packet, payload: bytes, seq_shift: int = 0) -> Packet:
    head = clone_head(origin)
    tcp = head.getlayer(TCP)
    if tcp is not None and hasattr(tcp, "seq"):
        try: tcp.seq = (int(tcp.seq) + max(0, seq_shift)) & 0xFFFFFFFF
        except: pass
    head /= Raw(payload) if payload else Raw(b"")
    _ = safe_build(head); return head

def p_frag_merge(pkts, cfg, rng):
    frag_p = float(cfg["frag_prob"]); merge_p = float(cfg["merge_prob"])
    max_len = int(cfg["merge_max_len"]); minfrag = int(cfg["min_fragment"])
    out = []; i = 0
    while i < len(pkts):
        p = pkts[i]; did_merge = False
        if (i+1) < len(pkts) and rng.random() < merge_p:
            p2 = pkts[i+1]
            if five_tuple(p) == five_tuple(p2):
                r1, r2 = p.getlayer(Raw), p2.getlayer(Raw)
                b1 = bytes(r1.load) if r1 else b""; b2 = bytes(r2.load) if r2 else b""
                if len(b1) + len(b2) <= max_len:
                    m = _head_with_payload(p, b1+b2, seq_shift=0)
                    m.time = (float(p.time)+float(p2.time))/2.0
                    out.append(m); i += 2; did_merge = True
        if did_merge: continue
        raw = p.getlayer(Raw)
        if raw is not None and rng.random() < frag_p:
            b = bytes(raw.load); parts = _split_payload(b, minfrag, rng)
            if len(parts) >= 2:
                f = _head_with_payload(p, parts[0], 0)
                s = _head_with_payload(p, parts[1], len(parts[0]))
                f.time = float(p.time); s.time = float(p.time) + 1e-6
                out += [f, s]; i += 1; continue
        out.append(p); i += 1
    out.sort(key=lambda x: float(x.time)); return out

def p_time_scale(pkts, cfg, rng):
    if len(pkts) <= 1: return pkts
    scale = float(cfg["scale"])
    iats = iats_from(pkts); apply_iats(pkts, [i*scale for i in iats]); return pkts

PERTURBATIONS = {
    "reorder": p_reorder,
    "retransmit": p_retransmit,
    "jitter_drop": p_jitter_drop,
    "frag_merge": p_frag_merge,
    "time_scale": p_time_scale,
}

# ===================== 3.5) Scope helpers =====================
def split_by_scope(pkts: List[Packet], scope_cfg: Dict[str, Any]) -> Tuple[List[Packet], List[Packet], int]:
    """
    Return (head, tail, k)
    - target == "all": head = pkts, tail = [], k=len(pkts)
    - target == "first_n": head = pkts[:n], tail = pkts[n:], k=n (falls back to all when n>=len(pkts))
    """
    target = (scope_cfg.get("target") or "all").lower()
    if target not in ("all", "first_n"):
        target = "all"
    if target == "all":
        return list(pkts), [], len(pkts)
    n = int(scope_cfg.get("n", 0))
    if n <= 0 or n >= len(pkts):
        return list(pkts), [], len(pkts)
    return list(pkts[:n]), list(pkts[n:]), n

def shift_tail_times_to_follow(head_new: List[Packet], tail_orig: List[Packet],
                               orig_head_last_t: float, orig_tail_first_t: float):
    """
    Shift tail timestamps to keep monotonic order after concatenation.
    Shift = (new_last + max(1e-6, original_boundary_gap)) - tail_orig_first
    """
    if not head_new or not tail_orig: return
    new_last = float(head_new[-1].time)
    # Minimum safe gap at the original boundary.
    base_gap = max(1e-6, float(orig_tail_first_t) - float(orig_head_last_t))
    shift = (new_last + base_gap) - float(orig_tail_first_t)
    if abs(shift) < 1e-12:
        return
    for p in tail_orig:
        p.time = float(p.time) + shift

# ===================== 4) Processing logic (combo + label noise + scope) =====================
def build_run_id(modes: List[int], level_code: int, run_name: str, is_combo: bool, scope: Dict[str, Any]) -> str:
    level = LEVEL_CODE[level_code]
    prefix = run_name or ""
    scope_tag = ""
    target = (scope.get("target") or "all").lower()
    if target == "first_n":
        scope_tag = f"_first{int(scope.get('n',0))}"
    if is_combo:
        combo = "COMBO(" + "+".join(str(m) for m in modes) + ")"
        return f"{prefix}{combo}_{level}{scope_tag}"
    else:
        name = PERTURB_CODE[modes[0]]
        return f"{prefix}{name}_{level}{scope_tag}"

def out_dir_for(cfg: Dict[str, Any], run_id: str) -> str:
    return os.path.join(cfg["out_root"], run_id)

def apply_chain_on_scope(pkts: List[Packet], chain_names: List[str], level: str,
                         rng_seed: int, scope_cfg: Dict[str, Any]) -> List[Packet]:
    """
    Apply packet-level perturbations (1~5) only within selected scope, then concatenate.
    """
    rng = random.Random(rng_seed)

    # Split by selected scope.
    head, tail, k = split_by_scope(pkts, scope_cfg)
    # Record original boundary times for tail shifting.
    orig_head_last_t = float(head[-1].time) if head else (float(pkts[-1].time) if pkts else 0.0)
    orig_tail_first_t = float(tail[0].time) if tail else 0.0

    cur = list(head)
    for name in chain_names:
        if name == "label_noise":
            continue
        func = PERTURBATIONS[name]
        cur = func(cur, PARAMS[name][level], rng)

    # Concatenate: perturbed head + untouched tail.
    head_new = cur
    tail_orig = tail

    # Shift tail to preserve monotonic timestamps.
    if tail_orig:
        shift_tail_times_to_follow(head_new, tail_orig, orig_head_last_t, orig_tail_first_t)

    return head_new + tail_orig

def process_one(task: Dict[str, Any]) -> Dict[str, Any]:
    cls = task["class"]; in_path = task["in_path"]
    out_dir = task["out_dir"]; out_path = os.path.join(out_dir, cls, os.path.basename(in_path))
    makedirs(os.path.dirname(out_path))

    try:
        pkts = scapy.rdpcap(in_path)
    except Exception as e:
        return {"ok": False, "class": cls, "in_path": in_path, "out_path": None, "error": f"rdpcap: {repr(e)}"}

    try:
        out_pkts = apply_chain_on_scope(
            pkts,
            task["chain_names"],
            task["level"],
            task["seed"],
            task["scope"]
        )
        write_pcap(out_pkts, out_path)
        return {
            "ok": True,
            "class": cls,
            "in_path": in_path,
            "out_path": out_path,
            "n_in": len(pkts),
            "n_out": len(out_pkts),
            "chain": task["chain_names"],
            "level": task["level"],
            "seed": task["seed"],
            "scope": task["scope"],
            "params": {n: PARAMS[n][task["level"]] for n in task["chain_names"] if n != "label_noise"},
        }
    except Exception as e:
        return {"ok": False, "class": cls, "in_path": in_path, "out_path": None, "error": f"perturb: {repr(e)}\n{traceback.format_exc()}"}

def collect_outputs_by_class(out_dir: str, pattern: str="*.pcap") -> DefaultDict[str, List[str]]:
    by_cls: DefaultDict[str, List[str]] = defaultdict(list)
    if not os.path.isdir(out_dir): return by_cls
    for cls in sorted(d for d in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, d))):
        for fp in sorted(glob(os.path.join(out_dir, cls, pattern))):
            by_cls[cls].append(fp)
    return by_cls

def run_label_noise(out_dir: str, level: str, seed: int, manifest_fp: str):
    eps = float(PARAMS["label_noise"][level]["epsilon"])
    rng = random.Random(seed ^ 0x9E3779B1)

    by_cls = collect_outputs_by_class(out_dir)
    classes = sorted(by_cls.keys())
    if len(classes) < 2:
        print("[WARN] label_noise requires at least 2 classes; skipped.")
        return {"enabled": False}

    moves = []
    for cls in classes:
        files = by_cls[cls]
        k = int(round(eps * len(files)))
        if k <= 0: continue
        to_flip = rng.sample(files, k) if len(files) >= k else files
        for fp in to_flip:
            candidates = [c for c in classes if c != cls]
            target_cls = rng.choice(candidates)
            new_path = os.path.join(os.path.dirname(os.path.dirname(fp)), target_cls, os.path.basename(fp))
            makedirs(os.path.dirname(new_path))
            shutil.move(fp, new_path)
            moves.append({
                "file": os.path.basename(fp),
                "original_class": cls,
                "new_class": target_cls,
                "from": fp,
                "to": new_path
            })

    with open(manifest_fp, "a", encoding="utf-8") as fw:
        fw.write(json.dumps({
            "ok": True,
            "label_noise": True,
            "epsilon": eps,
            "moved": len(moves),
            "moves": moves[:50],
        }, ensure_ascii=False) + "\n")

    print(f"[LABEL NOISE] epsilon={eps} moved={len(moves)} (showing <=50 in manifest)")
    return {"enabled": True, "epsilon": eps, "moved": len(moves)}

def print_selection_banner(modes: List[int], level_code: int, scope: Dict[str, Any]):
    level = LEVEL_CODE[level_code]
    target = (scope.get("target") or "all").lower()
    scope_line = f"scope: ALL packets"
    if target == "first_n":
        scope_line = f"scope: FIRST {int(scope.get('n',0))} packets"
    print("="*80)
    print(f"  Level: {level}  (1=light, 2=medium, 3=heavy)")
    print(f"  {scope_line}")
    print("  Ordered mode chain:")
    for m in modes:
        tag = PERTURB_CODE.get(m, f"UNKNOWN({m})")
        desc = MODE_DESCRIPTIONS.get(m, "")
        print(f"    {m} = {tag:12s}  - {desc}")
    print("="*80)

def main():
    cfg = CONFIG
    cfg["test_root"] = _resolve_path(cfg["test_root"])
    cfg["out_root"] = _resolve_path(cfg["out_root"])
    # Validate selections.
    modes = cfg["mode_codes"]
    if not modes:
        print("[ERROR] mode_codes is empty", file=sys.stderr); sys.exit(2)
    for m in modes:
        if m not in PERTURB_CODE:
            print(f"[ERROR] Unknown mode code: {m}", file=sys.stderr); sys.exit(2)
    if cfg["level_code"] not in LEVEL_CODE:
        print("[ERROR] level_code must be 1/2/3", file=sys.stderr); sys.exit(2)

    # Build chain from user input, deduplicated while preserving order.
    raw_chain = [PERTURB_CODE[m] for m in modes]
    seen = set(); chain_names = []
    for name in raw_chain:
        if name not in seen:
            seen.add(name); chain_names.append(name)
    # Force label_noise to run at the end.
    if "label_noise" in chain_names:
        chain_names = [n for n in chain_names if n != "label_noise"] + ["label_noise"]

    level = LEVEL_CODE[cfg["level_code"]]
    scope = cfg.get("scope", {"target":"all"})
    is_combo = len(chain_names) > 1
    run_id = build_run_id(modes, cfg["level_code"], cfg["run_name"], is_combo, scope)
    out_dir = out_dir_for(cfg, run_id)

    # Print selected configuration.
    print_selection_banner(modes, cfg["level_code"], scope)

    # Clean existing output directory when overwrite is enabled.
    if cfg["overwrite"] and os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    makedirs(out_dir)

    # Collect input pcap files.
    files = list_pcap_files(cfg["test_root"], cfg["pcap_glob"])
    if not files:
        print(f"[ERROR] No PCAP found: {cfg['test_root']}/<class>/{cfg['pcap_glob']}", file=sys.stderr); sys.exit(2)
    print(f"[INFO] {len(files)} files | chain={chain_names} | level={level} | run={run_id}")

    # Build tasks.
    manifest = os.path.join(out_dir, "manifest.jsonl")
    set_seeds(cfg["global_seed"])
    tasks = []
    for i, (cls, path) in enumerate(files):
        seed = (cfg["global_seed"] * 1000003 + i * 997 + (hash(path) & 0x7fffffff))
        tasks.append({
            "class": cls,
            "in_path": path,
            "out_dir": out_dir,
            "chain_names": chain_names,
            "level": level,
            "seed": seed,
            "scope": scope
        })

    # Execute packet-level perturbations in parallel.
    results = []
    with ProcessPoolExecutor(max_workers=cfg["workers"]) as ex:
        futures = [ex.submit(process_one, t) for t in tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc=run_id):
            results.append(f.result())

    ok = sum(1 for r in results if r.get("ok"))
    err = len(results) - ok
    with open(manifest, "w", encoding="utf-8") as fw:
        for r in results: fw.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[DONE packets] ok={ok}, err={err}, out_dir={out_dir}, manifest={manifest}")

    # Post-process label noise (mode 6) by moving files across class directories.
    if "label_noise" in chain_names:
        ln_info = run_label_noise(out_dir, level, cfg["global_seed"], manifest)
        if not ln_info.get("enabled", False):
            print("[INFO] label_noise did not run (insufficient class count).")

    print("[ALL DONE]")

if __name__ == "__main__":
    main()
