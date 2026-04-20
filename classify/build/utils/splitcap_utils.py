# utils/splitcap_utils.py

import subprocess
from pathlib import Path
from .io_utils import ensure_dir

def run_cmd(cmd: list):
    subprocess.run(cmd, check=False,
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)

def splitcap_session(exe: str, in_pcap: Path, out_dir: Path):
    """
    SplitCap -s session
    """
    ensure_dir(out_dir)
    run_cmd(["mono", exe, "-r", str(in_pcap), "-s", "session", "-o", str(out_dir)])

def splitcap_packets(exe: str, in_pcap: Path, out_dir: Path):
    """
    SplitCap -s packets 1
    """
    ensure_dir(out_dir)
    run_cmd(["mono", exe, "-r", str(in_pcap), "-s", "packets", "1", "-o", str(out_dir)])
