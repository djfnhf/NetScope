import argparse
import sys
from typing import Any, Dict, List, Optional
import numpy as np
from scipy.spatial import distance
from scipy.stats import entropy
import pandas as pd
import pickle
import sys
from collections import Counter

import pandas as pd

try:
    from scapy.all import rdpcap, IP, TCP, UDP  # type: ignore
except Exception as exc:  # pragma: no cover
    print("[ERROR] 需要安装 scapy：pip install scapy", file=sys.stderr)
    raise


def _extract_features_from_pcap(path: str) -> pd.DataFrame:
    """解析PCAP为逐包特征表，包含数值与分类特征。

    特征设计遵循通用性与可比较性，供 calculate_metrics 直接使用：
    - 分类特征: src_ip, dst_ip, l4_proto_name, tcp_flags_str
    - 数值特征: pkt_len, ip_ttl, ip_ihl, l4_sport, l4_dport, tcp_window, udp_len, inter_arrival
    """
    packets = rdpcap(path)

    rows: List[Dict[str, Any]] = []
    prev_ts: Optional[float] = None

    for pkt in packets:
        row: Dict[str, Any] = {}

        # 通用
        ts = float(getattr(pkt, "time", 0.0))
        row["pkt_len"] = int(len(pkt))
        row["inter_arrival"] = float(ts - prev_ts) if prev_ts is not None else float("nan")
        prev_ts = ts

        # IP 层
        if IP in pkt:
            ip = pkt[IP]
            row["src_ip"] = ip.src
            row["dst_ip"] = ip.dst
            row["ip_ttl"] = int(ip.ttl)
            # ihl 可能不存在（某些解析器未填），做保护
            ihl = getattr(ip, "ihl", None)
            row["ip_ihl"] = int(ihl) if ihl is not None else float("nan")
            proto = int(ip.proto)
        else:
            row["src_ip"] = None
            row["dst_ip"] = None
            row["ip_ttl"] = float("nan")
            row["ip_ihl"] = float("nan")
            proto = -1

        # 传输层标识
        if TCP in pkt:
            tcp = pkt[TCP]
            row["l4_proto_name"] = "TCP"
            row["l4_sport"] = int(tcp.sport)
            row["l4_dport"] = int(tcp.dport)
            row["tcp_window"] = int(getattr(tcp, "window", 0))
            # 将 flags 统一为字符串（如 "SYN,ACK"），便于作为分类特征
            flags = tcp.flags
            parts: List[str] = []
            if flags & 0x01:
                parts.append("FIN")
            if flags & 0x02:
                parts.append("SYN")
            if flags & 0x04:
                parts.append("RST")
            if flags & 0x08:
                parts.append("PSH")
            if flags & 0x10:
                parts.append("ACK")
            if flags & 0x20:
                parts.append("URG")
            row["tcp_flags_str"] = ",".join(parts) if parts else "NONE"
            row["udp_len"] = float("nan")
        elif UDP in pkt:
            udp = pkt[UDP]
            row["l4_proto_name"] = "UDP"
            row["l4_sport"] = int(udp.sport)
            row["l4_dport"] = int(udp.dport)
            row["udp_len"] = int(getattr(udp, "len", 0))
            row["tcp_window"] = float("nan")
            row["tcp_flags_str"] = None
        else:
            row["l4_proto_name"] = "OTHER"
            row["l4_sport"] = float("nan")
            row["l4_dport"] = float("nan")
            row["tcp_window"] = float("nan")
            row["udp_len"] = float("nan")
            row["tcp_flags_str"] = None

        rows.append(row)

    df = pd.DataFrame(rows)
    return df

def normalize_to_probability(dist):
    """将数据归一化为概率分布

    仅做非负截断，避免通过整体平移改变相对比例。
    """
    dist = np.array(dist, dtype=np.float64)
    dist = np.clip(dist, a_min=0.0, a_max=None)
    dist_sum = np.sum(dist)
    if dist_sum == 0:
        # 全零时回退为均匀分布
        return np.ones_like(dist) / len(dist)
    return dist / dist_sum

def calculate_categorical_metrics(real_data, generated_data):
    """计算分类数据的指标"""
    # 计算频率分布
    real_counts = Counter(real_data)
    gen_counts = Counter(generated_data)
    
    # 获取所有可能的类别
    all_categories = set(real_counts.keys()).union(set(gen_counts.keys()))
    
    # 创建概率分布向量
    real_probs = np.array([real_counts.get(cat, 0) for cat in all_categories])
    gen_probs = np.array([gen_counts.get(cat, 0) for cat in all_categories])
    
    # 归一化
    real_probs = normalize_to_probability(real_probs)
    gen_probs = normalize_to_probability(gen_probs)
    
    # 计算JSD
    m = 0.5 * (real_probs + gen_probs)
    jsd = 0.5 * (entropy(real_probs, m) + entropy(gen_probs, m))
    
    # 计算TVD
    tvd = 0.5 * np.sum(np.abs(real_probs - gen_probs))
    
    return {
        'jsd': jsd,
        'tvd': tvd
    }

def calculate_numerical_metrics(real_data, generated_data):
    """计算数值数据的指标"""
    real_data = np.asarray(real_data, dtype=np.float64)
    generated_data = np.asarray(generated_data, dtype=np.float64)
    
    # 使用直方图估计概率分布
    hist_min = min(np.min(real_data), np.min(generated_data))
    hist_max = max(np.max(real_data), np.max(generated_data))
    
    # 处理所有值相同的情况
    if hist_min == hist_max:
        hist_max += 1e-6
    
    bins = np.linspace(hist_min, hist_max, 100)
    
    # 计算直方图
    p_hist, _ = np.histogram(real_data, bins=bins, density=True)
    q_hist, _ = np.histogram(generated_data, bins=bins, density=True)
    
    # 归一化为概率分布
    p_hist = normalize_to_probability(p_hist)
    q_hist = normalize_to_probability(q_hist)
    
    # 计算JSD
    m = 0.5 * (p_hist + q_hist)
    jsd = 0.5 * (entropy(p_hist, m) + entropy(q_hist, m))
    
    # 计算TVD
    tvd = 0.5 * np.sum(np.abs(p_hist - q_hist))
    
    return {
        'jsd': jsd,
        'tvd': tvd
    }

def calculate_metrics(real_df, generated_df):
    """计算两个数据集之间的指标"""
    metrics = {
        'numerical': {'columns': [], 'jsd': [], 'tvd': []},
        'categorical': {'columns': [], 'jsd': [], 'tvd': []},
        'skipped': []
    }
    
    # 确保列名一致
    common_columns = set(real_df.columns) & set(generated_df.columns)
    if not common_columns:
        raise ValueError("No common columns found between datasets")
    
    for col in common_columns:
        real_data = real_df[col].dropna().values
        generated_data = generated_df[col].dropna().values
        
        # 确保有足够的数据点
        if len(real_data) < 5 or len(generated_data) < 5:
            metrics['skipped'].append(f"{col} (insufficient data: real={len(real_data)}, gen={len(generated_data)})")
            continue
        
        # 判断列类型
        is_numeric = pd.api.types.is_numeric_dtype(real_df[col]) and pd.api.types.is_numeric_dtype(generated_df[col])
        
        try:
            if is_numeric:
                # 数值列计算
                col_metrics = calculate_numerical_metrics(real_data, generated_data)
                metrics['numerical']['columns'].append(col)
                metrics['numerical']['jsd'].append(col_metrics['jsd'])
                metrics['numerical']['tvd'].append(col_metrics['tvd'])
            else:
                # 非数值列计算
                col_metrics = calculate_categorical_metrics(real_data, generated_data)
                metrics['categorical']['columns'].append(col)
                metrics['categorical']['jsd'].append(col_metrics['jsd'])
                metrics['categorical']['tvd'].append(col_metrics['tvd'])
                
        except Exception as e:
            metrics['skipped'].append(f"{col} (error: {str(e)})")
            continue
    
    # 计算平均指标
    def safe_mean(values):
        return np.nanmean(values) if values else np.nan
    
    results = {
        'numerical': {
            'num_columns': len(metrics['numerical']['columns']),
            'avg_jsd': safe_mean(metrics['numerical']['jsd']),
            'avg_tvd': safe_mean(metrics['numerical']['tvd']),
            'columns': metrics['numerical']['columns']
        },
        'categorical': {
            'num_columns': len(metrics['categorical']['columns']),
            'avg_jsd': safe_mean(metrics['categorical']['jsd']),
            'avg_tvd': safe_mean(metrics['categorical']['tvd']),
            'columns': metrics['categorical']['columns']
        },
        'num_skipped': len(metrics['skipped']),
        'skipped_columns': metrics['skipped']
    }
    
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="比较两个PCAP文件，评测指标（JSD/TVD）"
    )
    parser.add_argument("--pcap1", required=True, help="第一个PCAP路径")
    parser.add_argument("--pcap2", required=True, help="第二个PCAP路径")
    parser.add_argument(
        "--output",
        type=str,
        default="pcap_metrics_result.pk",
        help="输出指标结果的pickle文件",
    )
    args = parser.parse_args()

    # 解析两个PCAP为特征DataFrame
    print(f"读取PCAP: {args.pcap1}")
    df1 = _extract_features_from_pcap(args.pcap1)
    print(f"读取PCAP: {args.pcap2}")
    df2 = _extract_features_from_pcap(args.pcap2)

    # 使用既有评测指标函数（保持不变）
    print("\n计算评测指标（保持与CSV评测一致的度量）...")
    metrics = calculate_metrics(df1, df2)

    # 保存与打印
    import pickle

    with open(args.output, "wb") as f:
        pickle.dump(metrics, f)

    print("\n" + "=" * 60)
    print("PCAP Comparison Results Summary")
    print("=" * 60)

    if metrics["numerical"]["num_columns"] > 0:
        print("\nNumerical Columns Metrics:")
        print(
            f"- Processed {metrics['numerical']['num_columns']} columns: "+
            ", ".join(metrics["numerical"]["columns"])
        )
        print(f"- Average JSD: {metrics['numerical']['avg_jsd']:.4f}")
        print(f"- Average TVD: {metrics['numerical']['avg_tvd']:.4f}")
        pass

    if metrics["categorical"]["num_columns"] > 0:
        print("\nCategorical Columns Metrics:")
        print(
            f"- Processed {metrics['categorical']['num_columns']} columns: "+
            ", ".join(metrics["categorical"]["columns"])
        )
        print(f"- Average JSD: {metrics['categorical']['avg_jsd']:.4f}")
        print(f"- Average TVD: {metrics['categorical']['avg_tvd']:.4f}")
        pass

    if metrics["num_skipped"] > 0:
        print(f"\nSkipped {metrics['num_skipped']} columns:")
        for col in metrics["skipped_columns"][:10]:
            print(f"- {col}")
        if metrics["num_skipped"] > 10:
            print(f"- ... and {metrics['num_skipped'] - 10} more")

    print(f"\nResults saved to {args.output}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()


