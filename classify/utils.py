#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Basic traffic preprocessing utility functions used by pcap2tsv.py."""

import os, csv, json, random, ipaddress, binascii
from typing import List, Dict
from scapy.all import Raw
from scapy.layers.l2 import Ether
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6
from scapy.layers.tls.handshake import TLSClientHello
from scapy.layers.tls.extensions import TLS_Ext_ServerName


# ==========================================================
# 0. Generic helpers
# ==========================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_hex(data: bytes) -> str:
    """Convert bytes to hex string safely."""
    try:
        return binascii.hexlify(data).decode()
    except Exception:
        return ""


def random_field(bits: int) -> int:
    return random.randint(0, 2 ** bits - 1)


# ==========================================================
# 1. Truncation logic
# ==========================================================

def truncate_bytes(data: bytes, start: int = 76, length: int = 64, pad_bytes: bool = True) -> bytes:
    """Take length bytes from data[start:] and optionally zero-pad."""
    sub = data[start:start + length]
    if pad_bytes and len(sub) < length:
        sub += b"\x00" * (length - len(sub))
    return sub


# ==========================================================
# 2. Privacy field processing
# ==========================================================

def randomize_sensitive_fields(pkt):
    pkt = pkt.copy()
    try:
        if Ether in pkt:
            pkt[Ether].src = "02:00:%02x:%02x:%02x:%02x" % tuple(random.randint(0, 255) for _ in range(4))
            pkt[Ether].dst = "02:00:%02x:%02x:%02x:%02x" % tuple(random.randint(0, 255) for _ in range(4))
        if IP in pkt:
            pkt[IP].src = str(ipaddress.IPv4Address(random_field(32)))
            pkt[IP].dst = str(ipaddress.IPv4Address(random_field(32)))
            pkt[IP].id = random_field(16)
        elif IPv6 in pkt:
            pkt[IPv6].src = str(ipaddress.IPv6Address(random_field(128)))
            pkt[IPv6].dst = str(ipaddress.IPv6Address(random_field(128)))
        if TCP in pkt:
            pkt[TCP].sport = random_field(16)
            pkt[TCP].dport = random_field(16)
            pkt[TCP].seq = random_field(32)
            pkt[TCP].ack = random_field(32)
        elif UDP in pkt:
            pkt[UDP].sport = random_field(16)
            pkt[UDP].dport = random_field(16)
        if pkt.haslayer(TLSClientHello):
            hello = pkt[TLSClientHello]
            for ext in hello.extensions or []:
                if isinstance(ext, TLS_Ext_ServerName):
                    ext.servernames = [("www.%d.com" % random.randint(100, 999))]
    except Exception:
        pass
    return pkt


def zeroize_sensitive_fields(pkt):
    pkt = pkt.copy()
    try:
        if Ether in pkt:
            pkt[Ether].src = "00:00:00:00:00:00"
            pkt[Ether].dst = "00:00:00:00:00:00"
        if IP in pkt:
            pkt[IP].src = "0.0.0.0"
            pkt[IP].dst = "0.0.0.0"
            pkt[IP].id = 0
        elif IPv6 in pkt:
            pkt[IPv6].src = "::"
            pkt[IPv6].dst = "::"
        if TCP in pkt:
            pkt[TCP].sport = pkt[TCP].dport = pkt[TCP].seq = pkt[TCP].ack = 0
        elif UDP in pkt:
            pkt[UDP].sport = pkt[UDP].dport = 0
        if pkt.haslayer(TLSClientHello):
            hello = pkt[TLSClientHello]
            for ext in hello.extensions or []:
                if isinstance(ext, TLS_Ext_ServerName):
                    ext.servernames = [("zeroed")]
    except Exception:
        pass
    return pkt


# ==========================================================
# 3. Tokenization
# ==========================================================

def sliding_bigram_generation(hex_str: str,
                              window_size: int = 4,
                              stride: int = 2,
                              token_limit: int = 512) -> List[str]:
    """Sliding-window bigram tokenization."""
    tokens = []
    for i in range(0, len(hex_str) - window_size + 1, stride):
        tokens.append(hex_str[i:i + window_size])
        if token_limit and len(tokens) >= token_limit:
            break
    return tokens


# ==========================================================
# 4. Data output
# ==========================================================

def write_dataset_tsv(samples: List[dict], out_path: str):
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["label", "text_a"])
        for s in samples:
            w.writerow([s["label"], " ".join(s["token_sequence"])])


# ==========================================================
# 5. Label mapping
# ==========================================================

def build_label_map(input_root: str) -> Dict[str, int]:
    if not os.path.exists(input_root):
        raise FileNotFoundError(f"[build_label_map] Input path not found: {input_root}")
    classes = sorted([d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))])
    if not classes:
        raise ValueError(f"[build_label_map] No class folders found under path: {input_root}")
    label_map = {cls: idx for idx, cls in enumerate(classes)}
    print(f"[✓] build_label_map: {len(label_map)} classes → {label_map}")
    return label_map
