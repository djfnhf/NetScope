import argparse
import json
import sys
import time
from typing import Any, Dict, Iterable, List, Tuple

import ast

try:
    from scapy.all import IP, TCP, UDP, Raw, wrpcap  # type: ignore
except Exception as exc:  # pragma: no cover
    print("[ERROR] 需要安装 scapy：pip install scapy", file=sys.stderr)
    raise


def _safe_to_dict(item: Any) -> Any:
    """将字符串形式的字典安全解析为对象；其他类型原样返回。"""
    if isinstance(item, str):
        s = item.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                return ast.literal_eval(s)
            except Exception:
                # 不是合法的字面量，原样返回字符串
                return item
    return item


def _iter_pairs(
    headers: Dict[str, List[Any]], payloads: Dict[str, List[Any]], days: Iterable[str]
) -> Iterable[Tuple[Dict[str, Any], Any]]:
    """按指定 days 迭代对齐后的 (header, payload) 对。"""
    for day in days:
        h_list = headers.get(day)
        p_list = payloads.get(day)
        if not isinstance(h_list, list) or not isinstance(p_list, list):
            continue
        limit = min(len(h_list), len(p_list))
        for i in range(limit):
            h = _safe_to_dict(h_list[i])
            p = _safe_to_dict(p_list[i])
            if not isinstance(h, dict):
                continue
            yield h, p


def _payload_to_bytes(p: Any) -> bytes:
    """尽量从 payload 条目解析出字节序列。支持多种常见格式。"""
    # 如果是 dict，尝试常见字段
    if isinstance(p, dict):
        for key in ("payload", "data", "raw", "hex"):
            if key in p:
                return _string_to_bytes(str(p[key]))
        # 没有常见字段则为空
        return b""
    # 如果是纯字符串，尝试解析
    if isinstance(p, str):
        return _string_to_bytes(p)
    # 其他类型不支持
    return b""


def _string_to_bytes(s: str) -> bytes:
    s = s.strip()
    if not s:
        return b""
    # 优先尝试十六进制（允许带空格/冒号）
    hex_candidate = s.replace(" ", "").replace(":", "").replace("0x", "")
    if all(c in "0123456789abcdefABCDEF" for c in hex_candidate) and len(hex_candidate) % 2 == 0:
        try:
            return bytes.fromhex(hex_candidate)
        except Exception:
            pass
    # 退化为 utf-8 文本
    try:
        return s.encode("utf-8")
    except Exception:
        return b""


def _build_packet(h: Dict[str, Any], payload: bytes):
    """根据 header 构造 IP/TCP/UDP/Raw 包。

    返回:
        (packet, normalized_flag)
        normalized_flag 为 1 表示此包的 sport 或 dport 发生归一化；否则为 0。
    """

    def _norm_port(val: Any, fallback: int) -> Tuple[int, bool]:
        try:
            v = int(val)
        except Exception:
            return fallback, True
        if 0 <= v <= 65535:
            return v, False
        # 将异常端口折叠到 1024-65535 区间
        return 1024 + (abs(v) % 64512), True

    src = h.get("src")
    dst = h.get("dst")
    try:
        proto = int(h.get("proto", 6))
    except Exception:
        proto = 6
    sport, s_norm = _norm_port(h.get("sport", 12345), 12345)
    dport, d_norm = _norm_port(h.get("dport", 80), 80)

    ip = IP(src=src, dst=dst)
    l4 = None
    if proto == 6:
        l4 = TCP(sport=sport, dport=dport)
    elif proto == 17:
        l4 = UDP(sport=sport, dport=dport)
    else:
        # 默认使用 TCP（大多数样本为 6）
        l4 = TCP(sport=sport, dport=dport)

    if payload:
        return ip / l4 / Raw(load=payload), int(s_norm or d_norm)
    return ip / l4, int(s_norm or d_norm)


def load_json(path: str) -> Dict[str, List[Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} 不是字典格式的 JSON")
    return data  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="合并 header 与 payload 的 JSON 文件，生成 PCAP"
    )
    parser.add_argument(
        "--headers",
        required=True,
        help="header JSON 路径（示例：data/header_generation.json）",
    )
    parser.add_argument(
        "--payloads",
        required=True,
        help="payload JSON 路径（示例：data/payload_generation.json）",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="输出 PCAP 文件路径",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--day",
        help="仅处理指定的键/分组（如 M/Tu/We/Th/F）",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="处理两个 JSON 中共同存在的所有分组键",
    )
    parser.add_argument(
        "--ts-start",
        type=float,
        default=time.time(),
        help="起始时间戳（默认当前时间）",
    )
    parser.add_argument(
        "--ts-step",
        type=float,
        default=0.0001,
        help="连续包之间的时间增量（秒）",
    )

    args = parser.parse_args()

    headers = load_json(args.headers)
    payloads = load_json(args.payloads)

    if args.day:
        days = [args.day]
    elif args.all:
        days = sorted(set(headers.keys()) & set(payloads.keys()))
    else:
        # 默认选择双方共有的第一个分组（若无交集则尝试 headers 的第一个）
        common = [k for k in headers.keys() if k in payloads]
        days = [common[0]] if common else [next(iter(headers.keys()))]

    packets = []
    ts = float(args.ts_start)
    attempted = 0
    error_count = 0
    normalized_count = 0
    for h, p in _iter_pairs(headers, payloads, days):
        attempted += 1
        try:
            payload_bytes = _payload_to_bytes(p)
            pkt, normed = _build_packet(h, payload_bytes)
            pkt.time = ts  # type: ignore[attr-defined]
            ts += float(args.ts_step)
            packets.append(pkt)
            normalized_count += int(normed)
        except Exception:
            error_count += 1
            continue

    if not packets:
        print("[WARN] 未生成任何数据包（请检查 day 键或输入文件内容）", file=sys.stderr)
    else:
        wrpcap(args.output, packets)
    print(
        f"处理统计 -> 成功: {len(packets)}, 异常跳过: {error_count}, 端口归一化: {normalized_count}, 总计: {attempted}"
    )
    if packets:
        print(f"写出 PCAP: {args.output}（包数: {len(packets)}）")


if __name__ == "__main__":
    main()


