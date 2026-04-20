import argparse
import sys
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import entropy

try:
    from pcap_similarity import _extract_features_from_pcap
except Exception as exc:
    print("[ERROR] 无法从 pcap_similarity 导入解析函数，请确保两者位于同一目录。", file=sys.stderr)
    raise


class DiversityMetrics:
    """流量生成模型多样性评估指标集合"""
    
    def __init__(self, real_df: pd.DataFrame, gen_df: pd.DataFrame):
        self.real_df = real_df
        self.gen_df = gen_df
        self.common_columns = list(set(real_df.columns) & set(gen_df.columns))
        
    def calculate_entropy_diversity(self) -> Dict[str, Any]:
        """
        1. 熵多样性 (Entropy Diversity)
        计算每个特征的熵值，熵值越高表示分布越均匀，多样性越好
        """
        results = {
            'columns': [],
            'real_entropy': [],
            'gen_entropy': [],
            'entropy_ratio': [],
            'skipped': []
        }
        
        for col in self.common_columns:
            real_data = self.real_df[col].dropna().values
            gen_data = self.gen_df[col].dropna().values
            
            if len(real_data) < 2 or len(gen_data) < 2:
                results['skipped'].append(f"{col} (insufficient data)")
                continue
                
            # 计算概率分布
            real_counts = Counter(real_data)
            gen_counts = Counter(gen_data)
            
            # 归一化为概率
            real_probs = np.array(list(real_counts.values())) / len(real_data)
            gen_probs = np.array(list(gen_counts.values())) / len(gen_data)
            
            # 计算熵
            real_entropy = entropy(real_probs)
            # gen_entropy = entropy(gen_probs)
            
            # 计算熵比例 (生成数据熵 / 真实数据熵)
            # entropy_ratio = gen_entropy / real_entropy if real_entropy > 0 else 0
            
            results['columns'].append(col)
            results['real_entropy'].append(real_entropy)
            # results['gen_entropy'].append(gen_entropy)
            # results['entropy_ratio'].append(entropy_ratio)
        
        results['avg_entropy'] = np.mean(results['real_entropy']) if results['real_entropy'] else 0
        return results
    
    def calculate_coverage_diversity(self) -> Dict[str, Any]:
        """
        2. 覆盖多样性 (Coverage Diversity)
        计算生成数据对真实数据值域的覆盖程度
        """
        results = {
            'columns': [],
            'coverage_ratio': [],
            'real_unique': [],
            'gen_unique': [],
            'intersection_unique': [],
            'skipped': []
        }
        
        for col in self.common_columns:
            real_data = set(self.real_df[col].dropna().values)
            gen_data = set(self.gen_df[col].dropna().values)
            
            if len(real_data) == 0 or len(gen_data) == 0:
                results['skipped'].append(f"{col} (no data)")
                continue
            
            # 计算交集
            intersection = real_data & gen_data
            
            # 覆盖率 = 交集大小 / 真实数据唯一值数量
            coverage_ratio = len(intersection) / len(real_data) if len(real_data) > 0 else 0
            
            results['columns'].append(col)
            results['coverage_ratio'].append(coverage_ratio)
            results['real_unique'].append(len(real_data))
            results['gen_unique'].append(len(gen_data))
            results['intersection_unique'].append(len(intersection))
        
        results['avg_coverage_ratio'] = np.mean(results['coverage_ratio']) if results['coverage_ratio'] else 0
        return results
    
    def calculate_novelty_diversity(self) -> Dict[str, Any]:
        """
        3. 新颖性多样性 (Novelty Diversity)
        计算生成数据中不在真实数据中的新值的比例
        """
        results = {
            'columns': [],
            'novelty_ratio': [],
            'novel_count': [],
            'gen_unique': [],
            'skipped': []
        }
        
        for col in self.common_columns:
            real_data = set(self.real_df[col].dropna().values)
            gen_data = set(self.gen_df[col].dropna().values)
            
            if len(gen_data) == 0:
                results['skipped'].append(f"{col} (no generated data)")
                continue
            
            # 计算新颖值（生成数据中不在真实数据中的值）
            novel_values = gen_data - real_data
            novelty_ratio = len(novel_values) / len(gen_data) if len(gen_data) > 0 else 0
            
            results['columns'].append(col)
            results['novelty_ratio'].append(novelty_ratio)
            results['novel_count'].append(len(novel_values))
            results['gen_unique'].append(len(gen_data))
        
        results['avg_novelty_ratio'] = np.mean(results['novelty_ratio']) if results['novelty_ratio'] else 0
        return results

    def generate_diversity_report(self) -> Dict[str, Any]:
        """生成完整的多样性评估报告"""
        report = {
            'entropy_diversity': self.calculate_entropy_diversity(),
            'coverage_diversity': self.calculate_coverage_diversity(),
            'novelty_diversity': self.calculate_novelty_diversity()
        }
        
        return report


def print_diversity_report(report: Dict[str, Any]) -> None:
    """打印多样性评估报告"""
    print("\n" + "=" * 70)
    print("流量生成模型多样性评估报告")
    print("=" * 70)
    
    # 1. 熵多样性
    entropy_data = report['entropy_diversity']
    print(f"\n1. 熵多样性 (Entropy Diversity)")
    print(f"   平均熵: {entropy_data['avg_entropy']:.4f}")
    print(f"   处理列数: {len(entropy_data['columns'])}")
    
    # 2. 覆盖多样性
    coverage_data = report['coverage_diversity']
    print(f"\n3. 覆盖多样性 (Coverage Diversity)")
    print(f"   平均覆盖率: {coverage_data['avg_coverage_ratio']:.4f}")
    print(f"   处理列数: {len(coverage_data['columns'])}")
    
    # 3. 新颖性多样性
    novelty_data = report['novelty_diversity']
    print(f"\n4. 新颖性多样性 (Novelty Diversity)")
    print(f"   平均新颖性比例: {novelty_data['avg_novelty_ratio']:.4f}")
    print(f"   处理列数: {len(novelty_data['columns'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="计算流量生成模型的多样性指标")
    parser.add_argument("--real_pcap", required=True, help="真实流量PCAP文件路径")
    parser.add_argument("--gen_pcap", required=True, help="生成流量PCAP文件路径")
    parser.add_argument("--output", type=str, default="diversity_report.pk", help="输出报告文件")
    args = parser.parse_args()
    
    print(f"读取真实流量PCAP: {args.real_pcap}")
    real_df = _extract_features_from_pcap(args.real_pcap)
    print(f"读取生成流量PCAP: {args.gen_pcap}")
    gen_df = _extract_features_from_pcap(args.gen_pcap)
    
    print("\n计算多样性指标...")
    diversity_metrics = DiversityMetrics(real_df, gen_df)
    report = diversity_metrics.generate_diversity_report()
    
    # 保存报告
    import pickle
    with open(args.output, "wb") as f:
        pickle.dump(report, f)
    
    # 打印报告
    print_diversity_report(report)
    
    print(f"\n报告已保存到: {args.output}")


if __name__ == "__main__":
    main()