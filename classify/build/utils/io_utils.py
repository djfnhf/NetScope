# utils/io_utils.py

import os
import shutil
from pathlib import Path

def ensure_dir(p: Path):
    """Create directory recursively if not exist."""
    p.mkdir(parents=True, exist_ok=True)

def bytes_of_file(p: Path) -> int:
    """Return file size (safe)."""
    try:
        return p.stat().st_size
    except Exception:
        return 0

def materialize_file(src: Path, dst: Path):
    """
    Try link(src → dst). If fails, copy2.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)
