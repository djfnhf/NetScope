#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCAP文件IP地址筛选工具
用于从pcap文件中筛选出源IP和目的IP符合指定条件的包，并生成新的pcap文件
"""

import argparse
import sys
from pathlib import Path
from scapy.all import *
from scapy.layers.inet import IP
import ipaddress


def is_ip_match(packet_ip, target_ips):
    """
    检查包的IP地址是否匹配目标IP列表
    
    Args:
        packet_ip: 包的IP地址
        target_ips: 目标IP列表，支持单个IP、IP范围或CIDR
    
    Returns:
        bool: 是否匹配
    """
    if not packet_ip:
        return False
    
    try:
        packet_ip_obj = ipaddress.ip_address(packet_ip)
    except ValueError:
        return False
    
    for target_ip in target_ips:
        try:
            # 检查是否为CIDR格式
            if '/' in target_ip:
                network = ipaddress.ip_network(target_ip, strict=False)
                if packet_ip_obj in network:
                    return True
            else:
                # 单个IP地址
                target_ip_obj = ipaddress.ip_address(target_ip)
                if packet_ip_obj == target_ip_obj:
                    return True
        except ValueError:
            # 如果无法解析IP，跳过
            continue
    
    return False


def filter_pcap_by_ips(input_file, output_file, src_ips=None, dst_ips=None, 
                       src_ip_ranges=None, dst_ip_ranges=None,
                       src_dst_pairs=None):
    """
    根据源IP和目的IP筛选pcap文件
    
    Args:
        input_file: 输入pcap文件路径
        output_file: 输出pcap文件路径
        src_ips: 源IP列表
        dst_ips: 目的IP列表
        src_ip_ranges: 源IP范围列表（CIDR格式）
        dst_ip_ranges: 目的IP范围列表（CIDR格式）
        src_dst_pairs: 多组源-目的对，例如 [{'src': '1.1.1.1', 'dst': '2.2.2.2'}, ...]
    
    Returns:
        tuple: (匹配的包数量, 总包数量)
    """
    print(f"正在读取pcap文件: {input_file}")
    
    try:
        # 读取pcap文件
        packets = rdpcap(input_file)
        total_packets = len(packets)
        print(f"总共读取到 {total_packets} 个包")
        
        # 合并源IP和IP范围
        all_src_ips = []
        if src_ips:
            all_src_ips.extend(src_ips)
        if src_ip_ranges:
            all_src_ips.extend(src_ip_ranges)
        
        # 合并目的IP和IP范围
        all_dst_ips = []
        if dst_ips:
            all_dst_ips.extend(dst_ips)
        if dst_ip_ranges:
            all_dst_ips.extend(dst_ip_ranges)
        
        # 规范化pair列表
        normalized_pairs = []
        if src_dst_pairs:
            for pair in src_dst_pairs:
                src_val = pair.get('src')
                dst_val = pair.get('dst')
                if src_val and dst_val:
                    normalized_pairs.append({'src': src_val, 'dst': dst_val})

        # 筛选包
        filtered_packets = []
        matched_count = 0
        
        for i, packet in enumerate(packets):
            if i % 10000 == 0 and i > 0:
                print(f"已处理 {i}/{total_packets} 个包...")
            
            # 检查是否包含IP层
            if not packet.haslayer(IP):
                continue
            
            ip_layer = packet[IP]
            src_ip = ip_layer.src
            dst_ip = ip_layer.dst
            
            # 检查源IP匹配
            src_match = True
            if all_src_ips:
                src_match = is_ip_match(src_ip, all_src_ips)
            
            # 检查目的IP匹配
            dst_match = True
            if all_dst_ips:
                dst_match = is_ip_match(dst_ip, all_dst_ips)
            
            # 如传入pair，则匹配任一对：src匹配对的src且dst匹配对的dst
            pair_match = True
            if normalized_pairs:
                pair_match = False
                for pair in normalized_pairs:
                    if is_ip_match(src_ip, [pair['src']]) and is_ip_match(dst_ip, [pair['dst']]):
                        pair_match = True
                        break

            # 如果源IP和目的IP都匹配、且pair（若有）也匹配，则保留该包
            if src_match and dst_match and pair_match:
                filtered_packets.append(packet)
                matched_count += 1
        
        print(f"匹配到 {matched_count} 个包")
        
        if matched_count > 0:
            # 写入筛选后的包到新文件
            print(f"正在写入筛选后的包到: {output_file}")
            wrpcap(output_file, filtered_packets)
            print(f"成功生成筛选后的pcap文件: {output_file}")
        else:
            print("没有找到匹配的包，未生成输出文件")
        
        return matched_count, total_packets
        
    except Exception as e:
        print(f"处理pcap文件时出错: {e}")
        return 0, 0


def main():
    parser = argparse.ArgumentParser(
        description="PCAP文件IP地址筛选工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 筛选源IP为192.168.1.1的包
  python pcap_filter.py input.pcap output.pcap --src-ip 192.168.1.1
  
  # 筛选目的IP为10.0.0.0/8网段的包
  python pcap_filter.py input.pcap output.pcap --dst-ip-range 10.0.0.0/8
  
  # 同时筛选源IP和目的IP
  python pcap_filter.py input.pcap output.pcap --src-ip 192.168.1.1 --dst-ip 8.8.8.8
  
  # 筛选多个IP地址
  python pcap_filter.py input.pcap output.pcap --src-ip 192.168.1.1 --src-ip 192.168.1.2

  # 使用多组源-目的对进行匹配（任一对匹配即可）
  python pcap_filter.py input.pcap output.pcap \
      --pair src=192.168.1.1,dst=8.8.8.8 \
      --pair src=10.0.0.1,dst=1.1.1.1
        """
    )
    
    parser.add_argument('input_file', help='输入pcap文件路径')
    parser.add_argument('output_file', help='输出pcap文件路径')
    parser.add_argument('--src-ip', action='append', dest='src_ips', 
                       help='源IP地址（可多次指定）')
    parser.add_argument('--dst-ip', action='append', dest='dst_ips',
                       help='目的IP地址（可多次指定）')
    parser.add_argument('--src-ip-range', action='append', dest='src_ip_ranges',
                       help='源IP范围，CIDR格式（可多次指定）')
    parser.add_argument('--dst-ip-range', action='append', dest='dst_ip_ranges',
                       help='目的IP范围，CIDR格式（可多次指定）')
    parser.add_argument('--pair', action='append', dest='pairs',
                       help='源-目的对，格式为 src=x.x.x.x,dst=y.y.y.y （可多次指定）')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not Path(args.input_file).exists():
        print(f"错误: 输入文件不存在: {args.input_file}")
        sys.exit(1)
    
    # 检查是否至少指定了一个筛选条件
    if not any([args.src_ips, args.dst_ips, args.src_ip_ranges, args.dst_ip_ranges, args.pairs]):
        print("错误: 必须至少指定一个IP筛选条件")
        print("使用 --help 查看使用说明")
        sys.exit(1)
    
    # 创建输出目录（如果不存在）
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 执行筛选
    # 解析pairs
    parsed_pairs = []
    if args.pairs:
        for item in args.pairs:
            # 支持空格可选：src=...,dst=...
            parts = item.split(',')
            pair_dict = {}
            for part in parts:
                kv = part.split('=')
                if len(kv) == 2:
                    key = kv[0].strip().lower()
                    val = kv[1].strip()
                    if key in ('src', 'dst'):
                        pair_dict[key] = val
            if 'src' in pair_dict and 'dst' in pair_dict:
                parsed_pairs.append(pair_dict)

    matched, total = filter_pcap_by_ips(
        args.input_file, 
        args.output_file,
        src_ips=args.src_ips,
        dst_ips=args.dst_ips,
        src_ip_ranges=args.src_ip_ranges,
        dst_ip_ranges=args.dst_ip_ranges,
        src_dst_pairs=parsed_pairs
    )
    
    print(f"\n筛选完成:")
    print(f"  总包数: {total}")
    print(f"  匹配包数: {matched}")
    print(f"  匹配率: {matched/total*100:.2f}%" if total > 0 else "  匹配率: 0%")


if __name__ == "__main__":
    main()
