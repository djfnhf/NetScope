#!/usr/bin/env python3
import argparse
import json
import os
import re
import base64
import binascii
from typing import Optional, Tuple

from scapy.all import Ether, IP, TCP, UDP, Raw, wrpcap, Packet


def extract_field(text: str, key: str) -> Optional[str]:
    pattern = rf"{re.escape(key)}: ([^,]+)"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return None


def parse_mac(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = value.strip()
    if re.fullmatch(r"[0-9a-fA-F]{2}(:[0-9a-fA-F]{2}){5}", value):
        return value.lower()
    return None


def parse_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    value = value.strip()
    try:
        if value.startswith("0x") or value.startswith("0X"):
            return int(value, 16)
        return int(value)
    except Exception:
        return None


def parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def decode_payload(payload_text: Optional[str], encoding: str = "auto") -> Optional[bytes]:
    if payload_text is None:
        return None
    text = payload_text.strip()
    if text == "":
        return b""

    def try_hex(s: str) -> Optional[bytes]:
        s_clean = re.sub(r"[^0-9a-fA-F]", "", s)
        if len(s_clean) == 0:
            return None
        try:
            return bytes.fromhex(s_clean)
        except Exception:
            return None

    def try_b64(s: str) -> Optional[bytes]:
        try:
            return base64.b64decode(s, validate=True)
        except Exception:
            return None

    if encoding == "hex":
        return try_hex(text)
    if encoding == "base64":
        return try_b64(text)
    if encoding == "utf8":
        return text.encode("utf-8", errors="replace")

    # auto-detect: 0x... or looks-like-hex => hex; else valid base64; else utf8
    if text.lower().startswith("0x"):
        hex_bytes = try_hex(text[2:])
        if hex_bytes is not None:
            return hex_bytes
    # hex-like if mostly hex chars and even length after stripping
    hex_candidate = re.sub(r"[^0-9a-fA-F]", "", text)
    if len(hex_candidate) >= 2 and len(hex_candidate) % 2 == 0 and len(hex_candidate) >= int(len(text) * 0.6):
        hex_bytes = try_hex(text)
        if hex_bytes is not None:
            return hex_bytes
    b64_bytes = try_b64(text)
    if b64_bytes is not None:
        return b64_bytes
    return text.encode("utf-8", errors="replace")


def extract_payload_from_output(output_text: str, payload_key_candidates: Tuple[str, ...]) -> Optional[str]:
    for k in payload_key_candidates:
        v = extract_field(output_text, k)
        if v is not None:
            return v
    return None


def build_packet_from_output(output_text: str, payload_bytes: Optional[bytes] = None) -> Tuple[Packet, Optional[float]]:
    eth_src = parse_mac(extract_field(output_text, "eth.src"))
    eth_dst = parse_mac(extract_field(output_text, "eth.dst"))

    ip_src = extract_field(output_text, "ip.src")
    ip_dst = extract_field(output_text, "ip.dst")
    ip_proto = parse_int(extract_field(output_text, "ip.proto"))

    tcp_sport = parse_int(extract_field(output_text, "tcp.srcport"))
    tcp_dport = parse_int(extract_field(output_text, "tcp.dstport"))
    tcp_len = parse_int(extract_field(output_text, "tcp.len"))

    udp_sport = parse_int(extract_field(output_text, "udp.srcport"))
    udp_dport = parse_int(extract_field(output_text, "udp.dstport"))
    udp_length = parse_int(extract_field(output_text, "udp.length"))

    frame_len = parse_int(extract_field(output_text, "frame.len"))
    ts_epoch = parse_float(extract_field(output_text, "frame.time_epoch"))

    eth = Ether()
    if eth_src:
        eth.src = eth_src
    if eth_dst:
        eth.dst = eth_dst

    ip = IP()
    if ip_src:
        ip.src = ip_src
    if ip_dst:
        ip.dst = ip_dst

    payload = payload_bytes or b""

    if (ip_proto == 6) or (tcp_sport is not None and tcp_dport is not None):
        ip.proto = 6
        tcp = TCP()
        if tcp_sport is not None:
            tcp.sport = tcp_sport
        if tcp_dport is not None:
            tcp.dport = tcp_dport
        if tcp_len is not None and tcp_len > 0 and not payload:
            payload = b"A" * tcp_len
        pkt = eth / ip / tcp
        if payload:
            pkt = pkt / Raw(payload)
    elif (ip_proto == 17) or (udp_sport is not None and udp_dport is not None):
        ip.proto = 17
        udp = UDP()
        if udp_sport is not None:
            udp.sport = udp_sport
        if udp_dport is not None:
            udp.dport = udp_dport
        # udp.length includes header (8 bytes)
        if udp_length is not None and udp_length > 8 and not payload:
            payload = b"A" * max(0, udp_length - 8)
        pkt = eth / ip / udp
        if payload:
            pkt = pkt / Raw(payload)
    else:
        # Default to TCP without ports if protocol unknown
        ip.proto = 6
        pkt = eth / ip / TCP()

    # Optionally pad to frame.len to approximate original size
    if frame_len is not None:
        current_len = len(bytes(pkt))
        if frame_len > current_len:
            pad_len = frame_len - current_len
            pkt = pkt / Raw(b"\x00" * pad_len)

    if ts_epoch is not None:
        pkt.time = ts_epoch

    return pkt, ts_epoch


def process_input_file(input_path: str, limit: Optional[int] = None, payload_key: str = "payload", payload_encoding: str = "auto"):
    packets = []
    timestamps = []
    count = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Try to handle a non-JSONL file which might be a JSON array
                # Fall back to reading entire file once
                f.seek(0)
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        output_text = item.get("output") if isinstance(item, dict) else None
                        if not output_text:
                            continue
                        payload_text = None
                        if isinstance(item, dict):
                            payload_text = item.get(payload_key)
                        if payload_text is None:
                            payload_text = extract_payload_from_output(output_text, (payload_key,))
                        payload_bytes = decode_payload(payload_text, payload_encoding) if payload_text is not None else None

                        pkt, ts = build_packet_from_output(output_text, payload_bytes=payload_bytes)
                        packets.append(pkt)
                        timestamps.append(ts)
                        count += 1
                        if limit is not None and count >= limit:
                            return packets
                    return packets
                else:
                    raise

            output_text = obj.get("output") if isinstance(obj, dict) else None
            if not output_text:
                continue
            payload_text = None
            if isinstance(obj, dict):
                payload_text = obj.get(payload_key)
            if payload_text is None:
                payload_text = extract_payload_from_output(output_text, (payload_key,))
            payload_bytes = decode_payload(payload_text, payload_encoding) if payload_text is not None else None

            pkt, ts = build_packet_from_output(output_text, payload_bytes=payload_bytes)
            packets.append(pkt)
            timestamps.append(ts)
            count += 1
            if limit is not None and count >= limit:
                break

    return packets


def main():
    parser = argparse.ArgumentParser(description="Convert custom JSON lines with Wireshark-like fields to PCAP")
    parser.add_argument("--input", "-i", required=True, help="输入 JSON 或 JSONL 文件路径")
    parser.add_argument("--output", "-o", required=True, help="输出 PCAP 文件路径")
    parser.add_argument("--limit", type=int, default=None, help="最多转换的记录数，用于抽样调试")
    parser.add_argument("--payload-key", default="payload", help="JSON 对象中负载字段键名，默认 'payload'")
    parser.add_argument("--payload-encoding", choices=["auto", "hex", "base64", "utf8"], default="auto", help="payload 编码：auto/hex/base64/utf8")

    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    packets = process_input_file(input_path, limit=args.limit, payload_key=args.payload_key, payload_encoding=args.payload_encoding)
    if not packets:
        raise SystemExit("未解析到任何可用的报文记录（检查 input 与字段格式）")

    wrpcap(output_path, packets)
    print(f"写入 PCAP: {output_path}, 共 {len(packets)} 个报文")


if __name__ == "__main__":
    main()


