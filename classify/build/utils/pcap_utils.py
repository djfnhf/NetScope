# utils/pcap_utils.py

import math
from scapy.all import rdpcap, IP, IPv6, TCP, UDP
from scapy.utils import RawPcapReader

def fast_count_packets(path_str: str):
    """
    Light-weight raw packet counter using RawPcapReader.
    """
    try:
        cnt = 0
        for _, _ in RawPcapReader(path_str):
            cnt += 1
        return path_str, cnt
    except Exception:
        return path_str, 0

def ip_proto_len(pkt):
    """Extract (proto_str, L3_len)."""
    if IP in pkt:
        proto = "tcp" if TCP in pkt else ("udp" if UDP in pkt else "other")
        try:
            L = int(pkt[IP].len)
        except Exception:
            L = len(bytes(pkt[IP]))
        return proto, L

    if IPv6 in pkt:
        proto = "tcp" if TCP in pkt else ("udp" if UDP in pkt else "other")
        try:
            L = int(pkt[IPv6].plen) + 40
        except Exception:
            L = len(bytes(pkt[IPv6]))
        return proto, L

    return "other", 0

def rdpcap_first_pkt_info(pcap_path):
    """
    Returns: (proto_str, L3_len, pkt_cnt)
    """
    try:
        pkts = rdpcap(str(pcap_path))
        if len(pkts) == 0:
            return None, 0, 0
        p0 = pkts[0]
        proto, L = ip_proto_len(p0)
        return proto, L, len(pkts)
    except Exception:
        return None, 0, 0
