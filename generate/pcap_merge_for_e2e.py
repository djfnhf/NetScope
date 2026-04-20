#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCAP文件随机选取与合并工具
用于从两个文件夹的子目录中随机选取指定数量的pcap文件并进行合并
"""

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

try:
    from scapy.all import rdpcap, wrpcap
    from scapy.utils import PcapReader, PcapWriter
except ImportError:
    print("[ERROR] 需要安装 scapy：pip install scapy", file=sys.stderr)
    sys.exit(1)


class PcapRandomMerger:
    """PCAP文件随机选取与合并器"""
    
    def __init__(self, seed: Optional[int] = None):
        """
        初始化合并器
        
        Args:
            seed: 随机种子，用于可重现的结果
        """
        if seed is not None:
            random.seed(seed)
        self.seed = seed
    
    def scan_directory(self, directory: Path) -> Dict[str, List[Path]]:
        """
        扫描目录，收集所有子目录中的pcap文件
        
        Args:
            directory: 要扫描的目录路径
            
        Returns:
            字典，键为子目录名，值为该子目录下的pcap文件路径列表
        """
        pcap_files = {}
        
        if not directory.exists():
            print(f"警告: 目录不存在: {directory}")
            return pcap_files
        
        print(f"正在扫描目录: {directory}")
        
        # 遍历所有子目录
        for subdir in directory.iterdir():
            if subdir.is_dir():
                subdir_name = subdir.name
                pcap_files[subdir_name] = []
                
                # 收集该子目录下的所有pcap文件
                for file_path in subdir.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() == '.pcap':
                        pcap_files[subdir_name].append(file_path)
                
                print(f"  子目录 '{subdir_name}': 找到 {len(pcap_files[subdir_name])} 个pcap文件")
        
        return pcap_files

    def _sum_packets_per_subdir(self, pcap_files_by_subdir: Dict[str, List[Path]]) -> Dict[str, int]:
        """
        统计每个子目录下所有pcap文件的总包数。

        Returns:
            {subdir: total_packets}
        """
        totals: Dict[str, int] = {}
        for subdir, files in pcap_files_by_subdir.items():
            total = 0
            for fp in files:
                total += self.count_packets_in_pcap(fp)
            totals[subdir] = total
        return totals

    @staticmethod
    def _allocate_quota_by_proportion(totals: Dict[str, int], target_total: int) -> Dict[str, int]:
        """
        按各子目录可用包数占比分配全局配额，使用“最大余数法”保证整数且总和等于target_total。
        """
        if target_total <= 0 or not totals:
            return {k: 0 for k in totals}

        positive_totals = {k: max(v, 0) for k, v in totals.items()}
        eligible = [k for k, v in positive_totals.items() if v > 0]
        if not eligible:
            return {k: 0 for k in totals}

        sum_all = sum(positive_totals.values())
        if sum_all <= 0:
            return {k: 0 for k in totals}

        if target_total < len(eligible):
            print(
                f"警告: 目标包数量 {target_total} 小于非空子目录数量 {len(eligible)}，"
                "无法保证每个子目录至少1个包。"
            )

        exact = {k: target_total * (positive_totals[k] / sum_all) for k in totals.keys()}
        floor_part = {k: int(max(v, 0)) for k, v in exact.items()}
        assigned = sum(floor_part.values())
        remainder = target_total - assigned

        frac = sorted(((k, exact[k] - floor_part[k]) for k in totals.keys()), key=lambda x: x[1], reverse=True)
        result = dict(floor_part)
        idx = 0
        while remainder > 0 and idx < len(frac):
            k = frac[idx][0]
            result[k] = result.get(k, 0) + 1
            remainder -= 1
            idx += 1
            if idx == len(frac):
                idx = 0

        zero_need = [k for k in eligible if result.get(k, 0) == 0]
        for subdir in zero_need:
            if remainder > 0:
                result[subdir] = result.get(subdir, 0) + 1
                remainder -= 1
            else:
                donors = sorted(((k, v) for k, v in result.items() if v > 1), key=lambda x: x[1], reverse=True)
                if not donors:
                    print(f"警告: 子目录 {subdir} 无法分配至少1个包，请增大目标包数量。")
                    break
                donor_key, _ = donors[0]
                result[donor_key] -= 1
                result[subdir] = result.get(subdir, 0) + 1

        return result
    
    def count_packets_in_pcap(self, pcap_file: Path) -> int:
        """
        统计pcap文件中的包数量
        
        Args:
            pcap_file: pcap文件路径
            
        Returns:
            包数量
        """
        try:
            # 使用流式读取统计，避免一次性加载超大文件
            count = 0
            with PcapReader(str(pcap_file)) as reader:
                for _ in reader:
                    count += 1
            return count
        except Exception as e:
            print(f"警告: 无法读取pcap文件 {pcap_file}: {e}")
            return 0
    
    def random_select_files(self, pcap_files: List[Path], target_count: int) -> List[Path]:
        """
        随机选取pcap文件，直到达到目标包数量
        
        Args:
            pcap_files: 可选的pcap文件列表
            target_count: 目标包数量
            
        Returns:
            选中的pcap文件列表
        """
        if not pcap_files:
            return []
        
        selected_files = []
        current_count = 0
        
        # 随机打乱文件列表
        shuffled_files = pcap_files.copy()
        random.shuffle(shuffled_files)
        
        for pcap_file in shuffled_files:
            if current_count >= target_count:
                break
            
            packet_count = self.count_packets_in_pcap(pcap_file)
            if packet_count > 0:
                selected_files.append(pcap_file)
                current_count += packet_count
                print(f"  选中文件: {pcap_file.name} (包含 {packet_count} 个包，累计 {current_count} 个包)")
        
        return selected_files
    
    def merge_pcap_files(self, pcap_files: List[Path], output_file: Path, max_packets: Optional[int] = None) -> int:
        """
        合并多个pcap文件
        
        Args:
            pcap_files: 要合并的pcap文件列表
            output_file: 输出文件路径
            
        Returns:
            实际写入的包数量（受 max_packets 限制）
        """
        if not pcap_files:
            return 0
        
        written = 0
        remaining = max_packets if isinstance(max_packets, int) and max_packets > 0 else None
        
        print(f"正在合并 {len(pcap_files)} 个pcap文件到: {output_file}")
        
        # 确保输出目录存在
        output_file.parent.mkdir(parents=True, exist_ok=True)
        writer = PcapWriter(str(output_file), append=False, sync=True)
        
        try:
            for pcap_file in pcap_files:
                if remaining is not None and remaining <= 0:
                    break
                try:
                    read_count = 0
                    with PcapReader(str(pcap_file)) as reader:
                        for pkt in reader:
                            if remaining is not None and remaining <= 0:
                                break
                            writer.write(pkt)
                            written += 1
                            read_count += 1
                            if remaining is not None:
                                remaining -= 1
                    print(f"  读取文件: {pcap_file.name} ({read_count} 个包，已写入累计 {written})")
                except Exception as e:
                    print(f"  警告: 无法读取文件 {pcap_file.name}: {e}")
        finally:
            try:
                writer.close()
            except Exception:
                pass
        
        print(f"成功合并 {written} 个包到: {output_file}")
        return written
    
    def process_single_folder(self, folder_path: Path, target_count: int, output_dir: Path) -> Dict[str, int]:
        """
        处理单个文件夹
        
        Args:
            folder_path: 输入文件夹路径
            target_count: 目标包数量
            output_dir: 输出目录
            
        Returns:
            字典，键为子目录名，值为实际合并的包数量
        """
        print(f"\n=== 处理单个文件夹: {folder_path} ===")
        
        # 扫描文件夹
        pcap_files_by_subdir = self.scan_directory(folder_path)
        
        results = {}
        
        # 先统计每个子目录可用包数并按占比分配全局配额
        totals = self._sum_packets_per_subdir(pcap_files_by_subdir)
        quota = self._allocate_quota_by_proportion(totals, int(target_count))
        # 随机遍历子目录以平衡随机性，但每个子目录严格按其分配配额写入
        subdirs = list(pcap_files_by_subdir.items())
        random.shuffle(subdirs)
        for subdir_name, pcap_files in subdirs:
            alloc = int(quota.get(subdir_name, 0))
            if alloc <= 0:
                results[subdir_name] = 0
                continue
            if not pcap_files:
                print(f"子目录 '{subdir_name}' 中没有pcap文件，跳过")
                results[subdir_name] = 0
                continue
            
            print(f"\n处理子目录: {subdir_name}")
            print(f"  可用文件数: {len(pcap_files)}；为该子目录分配配额: {alloc}")
            
            # 针对当前子目录，仅按“该子目录配额”去选择文件并写入
            selected_files = self.random_select_files(pcap_files, alloc)
            if selected_files:
                output_subdir = output_dir / subdir_name
                output_file = output_subdir / f"merged_{subdir_name}.pcap"
                written = self.merge_pcap_files(selected_files, output_file, max_packets=alloc)
                results[subdir_name] = written
            else:
                print(f"  没有选中任何文件")
                results[subdir_name] = 0
        
        return results
    
    def process_two_folders(self, folder1_path: Path, folder2_path: Path, 
                          target_count1: int, target_count2: int, output_dir: Path) -> Dict[str, int]:
        """
        处理两个文件夹
        
        Args:
            folder1_path: 第一个输入文件夹路径
            folder2_path: 第二个输入文件夹路径
            target_count1: 第一个文件夹的目标包数量
            target_count2: 第二个文件夹的目标包数量
            output_dir: 输出目录
            
        Returns:
            字典，键为子目录名，值为实际合并的包数量
        """
        print(f"\n=== 处理两个文件夹 ===")
        print(f"文件夹1: {folder1_path} (目标: {target_count1} 个包)")
        print(f"文件夹2: {folder2_path} (目标: {target_count2} 个包)")
        
        # 扫描两个文件夹
        pcap_files1 = self.scan_directory(folder1_path)
        pcap_files2 = self.scan_directory(folder2_path)
        
        # 获取共同的子目录名
        common_subdirs = set(pcap_files1.keys()) & set(pcap_files2.keys())
        
        if not common_subdirs:
            print("警告: 两个文件夹中没有共同的子目录")
            return {}
        
        print(f"找到共同子目录: {sorted(common_subdirs)}")
        
        results = {}
        
        # 统计两个目录在共同子目录中的可用包数，并分别按占比分配配额
        totals1 = {k: 0 for k in common_subdirs}
        totals2 = {k: 0 for k in common_subdirs}
        for k in common_subdirs:
            for fp in pcap_files1.get(k, []):
                totals1[k] += self.count_packets_in_pcap(fp)
            for fp in pcap_files2.get(k, []):
                totals2[k] += self.count_packets_in_pcap(fp)
        quota1 = self._allocate_quota_by_proportion(totals1, int(target_count1))
        quota2 = self._allocate_quota_by_proportion(totals2, int(target_count2))

        subdirs = list(common_subdirs)
        random.shuffle(subdirs)
        for subdir_name in subdirs:
            alloc1 = int(quota1.get(subdir_name, 0))
            alloc2 = int(quota2.get(subdir_name, 0))
            if alloc1 <= 0 and alloc2 <= 0:
                results[subdir_name] = 0
                continue
            print(f"\n处理子目录: {subdir_name}")
            files1 = pcap_files1.get(subdir_name, [])
            files2 = pcap_files2.get(subdir_name, [])
            print(f"  文件夹1文件数: {len(files1)}；为该子目录分配配额: {alloc1}")
            print(f"  文件夹2文件数: {len(files2)}；为该子目录分配配额: {alloc2}")

            selected_files1 = self.random_select_files(files1, alloc1) if alloc1 > 0 else []
            selected_files2 = self.random_select_files(files2, alloc2) if alloc2 > 0 else []

            if selected_files1 or selected_files2:
                output_subdir = output_dir / subdir_name
                output_file = output_subdir / f"merged_{subdir_name}.pcap"

                written1 = 0
                if selected_files1 and alloc1 > 0:
                    written1 = self.merge_pcap_files(selected_files1, output_file, max_packets=alloc1)

                written2 = 0
                if selected_files2 and alloc2 > 0:
                    # 以追加方式写 folder2 的剩余额度
                    rem2 = alloc2
                    try:
                        output_subdir.mkdir(parents=True, exist_ok=True)
                        writer = PcapWriter(str(output_file), append=True, sync=True)
                        for pcap_file in selected_files2:
                            if rem2 <= 0:
                                break
                            try:
                                with PcapReader(str(pcap_file)) as reader:
                                    for pkt in reader:
                                        if rem2 <= 0:
                                            break
                                        writer.write(pkt)
                                        written2 += 1
                                        rem2 -= 1
                            except Exception as e:
                                print(f"  警告: 无法读取文件 {pcap_file.name}: {e}")
                        try:
                            writer.close()
                        except Exception:
                            pass
                    except Exception as e:
                        print(f"  警告: 追加写入失败: {e}")
                        written2 = 0

                merged_count = written1 + written2
                print(f"  子目录 {subdir_name} 合并完成：folder1 {written1} 个，folder2 {written2} 个，总计 {merged_count} 个")
                results[subdir_name] = merged_count
            else:
                print(f"  没有选中任何文件")
                results[subdir_name] = 0
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="PCAP文件随机选取与合并工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 处理单个文件夹，每个子目录随机选取250000个包
  python pcap_random_merge.py --single-folder .../data/CIC2017_sorted --count 2500 --output .../data/cic2017_netdiffusion_10000/CIC2017_test_pcap

  # 处理两个文件夹
  python pcap_random_merge.py --folder1 .../data/CIC2017_sorted --folder2 .../data/cic2017_trafficllm_10000/cic2017_trafficllm_generation --count1 7500 --count2 2500 --output .../data/cic2017_trafficllm_10000/7525

        """
    )
    
    # 输入选项
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--single-folder', type=Path, 
                      help='处理单个文件夹')
    group.add_argument('--folder1', type=Path, 
                      help='第一个输入文件夹路径')
    
    parser.add_argument('--folder2', type=Path,
                       help='第二个输入文件夹路径（与--folder1配合使用）')
    
    # 数量选项
    parser.add_argument('--count', type=int,
                       help='单个文件夹的目标包数量')
    parser.add_argument('--count1', type=int,
                       help='第一个文件夹的目标包数量')
    parser.add_argument('--count2', type=int,
                       help='第二个文件夹的目标包数量')
    
    # 输出选项
    parser.add_argument('--output', type=Path, required=True,
                       help='输出目录路径')
    
    # 其他选项
    parser.add_argument('--seed', type=int,
                       help='随机种子（用于可重现的结果）')
    
    args = parser.parse_args()
    
    # 验证参数
    if args.single_folder:
        if args.count is None:
            print("错误: 使用 --single-folder 时必须指定 --count")
            sys.exit(1)
        if args.folder2 is not None:
            print("错误: 使用 --single-folder 时不能指定 --folder2")
            sys.exit(1)
    else:
        if args.folder2 is None:
            print("错误: 使用 --folder1 时必须指定 --folder2")
            sys.exit(1)
        if args.count1 is None or args.count2 is None:
            print("错误: 使用两个文件夹时必须指定 --count1 和 --count2")
            sys.exit(1)
    
    # 检查输入目录是否存在
    if args.single_folder and not args.single_folder.exists():
        print(f"错误: 输入目录不存在: {args.single_folder}")
        sys.exit(1)
    
    if args.folder1 and not args.folder1.exists():
        print(f"错误: 输入目录不存在: {args.folder1}")
        sys.exit(1)
    
    if args.folder2 and not args.folder2.exists():
        print(f"错误: 输入目录不存在: {args.folder2}")
        sys.exit(1)
    
    # 创建输出目录
    args.output.mkdir(parents=True, exist_ok=True)
    
    # 创建合并器
    merger = PcapRandomMerger(seed=args.seed)
    
    if args.seed:
        print(f"使用随机种子: {args.seed}")
    
    start_time = time.time()
    
    # 执行处理
    if args.single_folder:
        results = merger.process_single_folder(args.single_folder, args.count, args.output)
    else:
        results = merger.process_two_folders(args.folder1, args.folder2, 
                                           args.count1, args.count2, args.output)
    
    end_time = time.time()
    
    # 输出结果统计
    print(f"\n=== 处理完成 ===")
    print(f"处理时间: {end_time - start_time:.2f} 秒")
    print(f"处理的子目录数: {len(results)}")
    
    total_packets = 0
    for subdir_name, packet_count in results.items():
        print(f"  {subdir_name}: {packet_count} 个包")
        total_packets += packet_count
    
    print(f"总包数: {total_packets}")
    print(f"输出目录: {args.output}")


if __name__ == "__main__":
    main()
