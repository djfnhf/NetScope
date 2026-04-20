# flow_stage/flow_split.py

import math
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

from utils.splitcap_utils import splitcap_session
from utils.io_utils import ensure_dir

def worker_split_session(args):
    exe, big, out_dir = args
    splitcap_session(exe, big, out_dir)
    return str(big)

def perform_session_split(class_dirs, out_root, exe, workers):
    """
    Run SplitCap -s session.
    Return: [(cls_name, src_tag_dir), ...]
    """
    session_tasks = []

    for cls in class_dirs:
        for big in sorted(cls.iterdir()):
            if not big.is_file():
                continue

            src_tag = f"{cls.name}_{big.stem}_temp"
            out_dir = out_root / cls.name / src_tag

            if out_dir.exists() and any(out_dir.iterdir()):
                continue

            session_tasks.append((exe, big, out_dir))

    if not session_tasks:
        print("[Flow] Session splitting already exists, skip.")
        return

    ensure_dir(out_root)

    cs = max(1, math.ceil(len(session_tasks) / workers))
    print(f"[Flow] Session splitting: tasks={len(session_tasks)} workers={workers} chunksize={cs}")

    with Pool(workers) as pool, tqdm(total=len(session_tasks),
                                     desc="SplitCap session (flow)",
                                     ncols=100, leave=False) as pbar:
        for _ in pool.imap_unordered(worker_split_session, session_tasks, chunksize=cs):
            pbar.update(1)
