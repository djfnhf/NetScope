import argparse
import json
import sys
import time
import os
import socket
import re
from typing import Any, Dict, Iterable, List, Tuple, Optional
from pathlib import Path

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


def _validate_and_clean_ip(ip_str: Any) -> Optional[str]:
    """验证并清理IP地址字符串。
    
    Args:
        ip_str: 可能的IP地址字符串
        
    Returns:
        清理后的有效IP地址，如果无效则返回None
    """
    if not ip_str:
        return None
    
    ip_str = str(ip_str).strip()
    
    # 检查是否为空
    if not ip_str:
        return None
    
    # 移除常见的无效字符
    ip_str = re.sub(r'[^\d\.]', '', ip_str)
    
    # 检查IPv4格式
    ipv4_pattern = r'^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$'
    match = re.match(ipv4_pattern, ip_str)
    if match:
        # 验证每个段是否在0-255范围内
        segments = [int(x) for x in match.groups()]
        if all(0 <= seg <= 255 for seg in segments):
            try:
                # 使用socket验证IP地址
                socket.inet_pton(socket.AF_INET, ip_str)
                return ip_str
            except (OSError, ValueError):
                pass
    
    # 如果IPv4验证失败，返回None表示无效
    return None


def _validate_and_clean_port(port_val: Any) -> Optional[int]:
    """验证并清理端口号。
    
    Args:
        port_val: 端口值
        
    Returns:
        有效的端口号，如果无效则返回None
    """
    try:
        port = int(port_val)
        if 0 <= port <= 65535:
            return port
        else:
            # 端口超出范围，返回None表示无效
            return None
    except (ValueError, TypeError):
        return None


def _build_packet(h: Dict[str, Any], payload: bytes):
    """根据 header 构造 IP/TCP/UDP/Raw 包。

    返回:
        (packet, None, None) 如果成功
        (None, error_type, error_message) 如果失败 # <--- MODIFIED: 返回值结构变更
    """
    # 验证和清理IP地址
    src = _validate_and_clean_ip(h.get("src"))
    dst = _validate_and_clean_ip(h.get("dst"))
    
    # 检查IP地址是否有效
    if not src:
        return None, "INVALID_IP", f"无效的源IP地址: {h.get('src')}" # <--- MODIFIED
    if not dst:
        return None, "INVALID_IP", f"无效的目标IP地址: {h.get('dst')}" # <--- MODIFIED
    
    # 验证协议号
    try:
        proto = int(h.get("proto", 6))
    except Exception:
        proto = 6
    
    # 验证和清理端口号
    sport = _validate_and_clean_port(h.get("sport", 12345))
    dport = _validate_and_clean_port(h.get("dport", 80))
    
    # 检查端口是否有效
    if sport is None:
        return None, "INVALID_PORT", f"无效的源端口: {h.get('sport')}" # <--- MODIFIED
    if dport is None:
        return None, "INVALID_PORT", f"无效的目标端口: {h.get('dport')}" # <--- MODIFIED

    # 创建IP层
    ip = IP(src=src, dst=dst)
    
    # 创建传输层
    l4 = None
    if proto == 6:
        l4 = TCP(sport=sport, dport=dport)
    elif proto == 17:
        l4 = UDP(sport=sport, dport=dport)
    else:
        # 默认使用 TCP（大多数样本为 6）
        l4 = TCP(sport=sport, dport=dport)

    if payload:
        return ip / l4 / Raw(load=payload), None, None # <--- MODIFIED
    return ip / l4, None, None # <--- MODIFIED


def load_json(path: str) -> Dict[str, List[Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} 不是字典格式的 JSON")
    return data  # type: ignore[return-value]


def scan_folder_for_json_files(folder_path: str) -> Dict[str, List[str]]:
    """扫描文件夹，按子文件夹名称分组返回JSON文件路径。
    
    Returns:
        Dict[str, List[str]]: 子文件夹名称 -> JSON文件路径列表的映射
    """
    folder_dict = {}
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        raise ValueError(f"文件夹不存在: {folder_path}")
    
    # 遍历所有子文件夹
    for subfolder in folder_path.iterdir():
        if subfolder.is_dir():
            json_files = []
            # 在子文件夹中查找JSON文件
            for file_path in subfolder.glob("*.json"):
                json_files.append(str(file_path))
            
            if json_files:
                folder_dict[subfolder.name] = sorted(json_files)
    
    return folder_dict


def merge_json_files(json_files: List[str]) -> Dict[str, List[Any]]:
    """合并多个JSON文件的内容。
    
    Args:
        json_files: JSON文件路径列表
        
    Returns:
        合并后的字典数据
    """
    merged_data = {}
    
    for json_file in json_files:
        try:
            data = load_json(json_file)
            # 合并数据，如果键相同则合并列表
            for key, value in data.items():
                if key in merged_data:
                    if isinstance(merged_data[key], list) and isinstance(value, list):
                        merged_data[key].extend(value)
                    else:
                        # 如果类型不匹配，转换为列表
                        if not isinstance(merged_data[key], list):
                            merged_data[key] = [merged_data[key]]
                        if not isinstance(value, list):
                            value = [value]
                        merged_data[key].extend(value)
                else:
                    merged_data[key] = value
        except Exception as e:
            print(f"[WARN] 跳过文件 {json_file}: {e}", file=sys.stderr)
            continue
    
    return merged_data


def find_matching_subfolders(folder1_dict: Dict[str, List[str]], 
                            folder2_dict: Dict[str, List[str]]) -> List[str]:
    """找到两个文件夹中共同存在的子文件夹名称。
    
    Returns:
        共同子文件夹名称列表
    """
    common_folders = set(folder1_dict.keys()) & set(folder2_dict.keys())
    return sorted(list(common_folders))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按子文件夹名称合并两个文件夹的JSON文件并生成PCAP"
    )
    
    # 添加模式选择
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--folder-mode",
        action="store_true",
        help="文件夹模式：按子文件夹名称合并两个文件夹的JSON文件"
    )
    mode_group.add_argument(
        "--file-mode",
        action="store_true",
        help="文件模式：合并指定的header和payload JSON文件"
    )
    
    # 文件夹模式参数
    parser.add_argument(
        "--headers-folder",
        help="header JSON文件所在的文件夹路径",
    )
    parser.add_argument(
        "--payloads-folder", 
        help="payload JSON文件所在的文件夹路径",
    )
    
    # 文件模式参数（保持向后兼容）
    parser.add_argument(
        "--headers",
        help="header JSON 路径（示例：data/header_generation.json）",
    )
    parser.add_argument(
        "--payloads",
        help="payload JSON 路径（示例：data/payload_generation.json）",
    )
    
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="输出文件夹路径（文件夹模式）或PCAP文件路径（文件模式）",
    )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--subfolder",
        help="仅处理指定的子文件夹名称",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="处理所有匹配的子文件夹",
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

    if args.folder_mode:
        # 文件夹模式
        if not args.headers_folder or not args.payloads_folder:
            parser.error("文件夹模式需要指定 --headers-folder 和 --payloads-folder")
        
        # 扫描两个文件夹
        print(f"扫描headers文件夹: {args.headers_folder}")
        headers_folder_dict = scan_folder_for_json_files(args.headers_folder)
        print(f"找到 {len(headers_folder_dict)} 个子文件夹: {list(headers_folder_dict.keys())}")
        
        print(f"扫描payloads文件夹: {args.payloads_folder}")
        payloads_folder_dict = scan_folder_for_json_files(args.payloads_folder)
        print(f"找到 {len(payloads_folder_dict)} 个子文件夹: {list(payloads_folder_dict.keys())}")
        
        # 找到匹配的子文件夹
        if args.subfolder:
            if args.subfolder not in headers_folder_dict or args.subfolder not in payloads_folder_dict:
                print(f"[ERROR] 子文件夹 '{args.subfolder}' 在两个文件夹中不都存在", file=sys.stderr)
                return
            subfolders = [args.subfolder]
        elif args.all:
            subfolders = find_matching_subfolders(headers_folder_dict, payloads_folder_dict)
        else:
            # 默认处理第一个匹配的子文件夹
            subfolders = find_matching_subfolders(headers_folder_dict, payloads_folder_dict)
            if not subfolders:
                print("[ERROR] 没有找到匹配的子文件夹", file=sys.stderr)
                return
            subfolders = [subfolders[0]]
        
        print(f"将处理以下子文件夹: {subfolders}")
        
        # 创建输出文件夹
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # <--- ADDED: 初始化总的分类计数器
        total_error_counts = {"INVALID_IP": 0, "INVALID_PORT": 0, "OTHER": 0}
        total_packets = 0
        total_attempted = 0
        
        # 处理每个子文件夹
        for subfolder in subfolders:
            print(f"\n处理子文件夹: {subfolder}")
            
            # 合并该子文件夹中的所有JSON文件
            headers_files = headers_folder_dict[subfolder]
            payloads_files = payloads_folder_dict[subfolder]
            
            print(f"  合并 {len(headers_files)} 个headers文件")
            headers_data = merge_json_files(headers_files)
            
            print(f"  合并 {len(payloads_files)} 个payloads文件")
            payloads_data = merge_json_files(payloads_files)
            
            # 生成PCAP文件
            packets = []
            ts = float(args.ts_start)
            attempted = 0
            # <--- ADDED: 初始化子文件夹的分类计数器
            error_counts = {"INVALID_IP": 0, "INVALID_PORT": 0, "OTHER": 0}
            
            # 获取所有可用的键
            available_keys = sorted(set(headers_data.keys()) & set(payloads_data.keys()))
            if not available_keys:
                print(f"  [WARN] 子文件夹 {subfolder} 中没有找到匹配的键")
                continue
            
            print(f"  找到 {len(available_keys)} 个匹配的键: {available_keys}")
            
            for h, p in _iter_pairs(headers_data, payloads_data, available_keys):
                attempted += 1
                try:
                    payload_bytes = _payload_to_bytes(p)
                    # <--- MODIFIED: 接收三个返回值
                    pkt, error_type, error_msg = _build_packet(h, payload_bytes)
                    
                    if pkt is None:
                        # <--- MODIFIED: 根据错误类型进行分类计数
                        error_counts[error_type] += 1
                        if sum(error_counts.values()) <= 5:  # 只显示前5个错误
                            print(f"    [WARN] 跳过数据包 ({error_type}): {error_msg}", file=sys.stderr)
                        continue
                    
                    # 数据包构建成功
                    pkt.time = ts  # type: ignore[attr-defined]
                    ts += float(args.ts_step)
                    packets.append(pkt)
                    
                except Exception as e:
                    # <--- MODIFIED: 计入 OTHER 错误
                    error_counts["OTHER"] += 1
                    if sum(error_counts.values()) <= 5:  # 只显示前5个错误
                        print(f"    [WARN] 跳过数据包 (OTHER): {e}", file=sys.stderr)
                    continue
            
            # 保存PCAP文件：在输出目录下为每个子文件夹创建同名子目录
            if packets:
                sub_output_dir = output_path / subfolder
                sub_output_dir.mkdir(parents=True, exist_ok=True)
                output_file = sub_output_dir / f"{subfolder}.pcap"
                wrpcap(str(output_file), packets)
                print(f"  生成PCAP: {output_file}（包数: {len(packets)}）")
            else:
                print(f"  [WARN] 子文件夹 {subfolder} 未生成任何数据包")
            
            # 累加到总计数器
            total_packets += len(packets)
            total_attempted += attempted
            for key in total_error_counts:
                total_error_counts[key] += error_counts[key]
        
        total_errors = sum(total_error_counts.values())
        print(f"\n--- 总体统计 ---") # <--- MODIFIED: 更新最终的打印报告
        print(f"成功生成数据包: {total_packets}")
        print(f"总共尝试处理: {total_attempted}")
        print(f"总共跳过数据包: {total_errors}")
        print(f"  - 因无效IP地址跳过: {total_error_counts['INVALID_IP']}")
        print(f"  - 因无效端口号跳过: {total_error_counts['INVALID_PORT']}")
        print(f"  - 因其他错误跳过: {total_error_counts['OTHER']}")
        print(f"输出文件夹: {output_path}")
        
    else:
        # 文件模式（原有功能）
        if not args.headers or not args.payloads:
            parser.error("文件模式需要指定 --headers 和 --payloads")
        
        headers = load_json(args.headers)
        payloads = load_json(args.payloads)

        if args.subfolder:
            days = [args.subfolder]
        elif args.all:
            days = sorted(set(headers.keys()) & set(payloads.keys()))
        else:
            # 默认选择双方共有的第一个分组（若无交集则尝试 headers 的第一个）
            common = [k for k in headers.keys() if k in payloads]
            days = [common[0]] if common else [next(iter(headers.keys()))]

        packets = []
        ts = float(args.ts_start)
        attempted = 0
        # <--- ADDED: 初始化文件模式的分类计数器
        error_counts = {"INVALID_IP": 0, "INVALID_PORT": 0, "OTHER": 0}
        
        for h, p in _iter_pairs(headers, payloads, days):
            attempted += 1
            try:
                payload_bytes = _payload_to_bytes(p)
                # <--- MODIFIED: 接收三个返回值
                pkt, error_type, error_msg = _build_packet(h, payload_bytes)
                
                if pkt is None:
                    # <--- MODIFIED: 根据错误类型进行分类计数
                    error_counts[error_type] += 1
                    if sum(error_counts.values()) <= 5:  # 只显示前5个错误
                        print(f"[WARN] 跳过数据包 ({error_type}): {error_msg}", file=sys.stderr)
                    continue
                
                # 数据包构建成功
                pkt.time = ts  # type: ignore[attr-defined]
                ts += float(args.ts_step)
                packets.append(pkt)
                
            except Exception as e:
                # <--- MODIFIED: 计入 OTHER 错误
                error_counts["OTHER"] += 1
                if sum(error_counts.values()) <= 5:  # 只显示前5个错误
                    print(f"[WARN] 跳过数据包 (OTHER): {e}", file=sys.stderr)
                continue

        if not packets:
            print("[WARN] 未生成任何数据包（请检查 day 键或输入文件内容）", file=sys.stderr)
        
        total_errors = sum(error_counts.values())
        print("\n--- 处理统计 ---") # <--- MODIFIED: 更新最终的打印报告
        print(f"成功生成数据包: {len(packets)}")
        print(f"总共尝试处理: {attempted}")
        print(f"总共跳过数据包: {total_errors}")
        print(f"  - 因无效IP地址跳过: {error_counts['INVALID_IP']}")
        print(f"  - 因无效端口号跳过: {error_counts['INVALID_PORT']}")
        print(f"  - 因其他错误跳过: {error_counts['OTHER']}")
        
        if packets:
            wrpcap(args.output, packets)
            print(f"写出 PCAP: {args.output}（包数: {len(packets)}）")


if __name__ == "__main__":
    main()