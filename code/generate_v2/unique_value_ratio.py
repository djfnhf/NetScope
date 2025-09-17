import argparse
import sys
from typing import Any, Dict, List, Optional
from collections import Counter

import numpy as np
import pandas as pd

try:
    # 仅复用 pcap 解析逻辑
    from pcap_similarity import _extract_features_from_pcap  # type: ignore
except Exception as exc:  # pragma: no cover
    print("[ERROR] 无法从 pcap_similarity 导入解析函数，请确保两者位于同一目录。", file=sys.stderr)
    raise


def calculate_unique_value_ratio_for_categorical(real_df: pd.DataFrame, gen_df: pd.DataFrame) -> Dict[str, Any]:
    common_columns = list(set(real_df.columns) & set(gen_df.columns))
    if not common_columns:
        raise ValueError("No common columns between two DataFrames")

    columns: List[str] = []
    real_uniques: List[int] = []
    gen_uniques: List[int] = []
    unique_ratios: List[float] = []
    skipped: List[str] = []

    for col in common_columns:
        if pd.api.types.is_numeric_dtype(real_df[col]) and pd.api.types.is_numeric_dtype(gen_df[col]):
            continue

        real_vals = real_df[col].dropna().values
        gen_vals = gen_df[col].dropna().values

        if len(real_vals) < 1 or len(gen_vals) < 1:
            skipped.append(f"{col} (insufficient data)")
            continue

        real_counts = Counter(real_vals)
        gen_counts = Counter(gen_vals)

        real_u = len(real_counts)
        gen_u = len(gen_counts)
        denom = max(real_u, gen_u)
        ratio = (min(real_u, gen_u) / denom) if denom != 0 else 0.0

        columns.append(col)
        real_uniques.append(real_u)
        gen_uniques.append(gen_u)
        unique_ratios.append(float(ratio))

    def safe_mean(values: List[float]) -> float:
        return float(np.nanmean(values)) if values else float("nan")

    return {
        "num_columns": len(columns),
        "columns": columns,
        "real_unique": real_uniques,
        "gen_unique": gen_uniques,
        "unique_ratio": unique_ratios,
        "avg_unique_ratio": safe_mean(unique_ratios),
        "num_skipped": len(skipped),
        "skipped": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="计算两个 PCAP 在分类特征上的 Unique Value Ratio")
    parser.add_argument("--pcap1", required=True, help="第一个PCAP路径")
    parser.add_argument("--pcap2", required=True, help="第二个PCAP路径")
    args = parser.parse_args()

    print(f"读取PCAP: {args.pcap1}")
    df1 = _extract_features_from_pcap(args.pcap1)
    print(f"读取PCAP: {args.pcap2}")
    df2 = _extract_features_from_pcap(args.pcap2)

    print("\n计算 Unique Value Ratio ...")
    res = calculate_unique_value_ratio_for_categorical(df1, df2)

    print("\n" + "=" * 60)
    print("Unique Value Ratio Summary (Categorical Columns)")
    print("=" * 60)
    if res["num_columns"] > 0:
        print(f"- Processed {res['num_columns']} columns: "+", ".join(res["columns"]))
        print(f"- Average Unique Value Ratio: {res['avg_unique_ratio']:.2%}")
    if res["num_skipped"] > 0:
        print(f"- Skipped {res['num_skipped']} columns")
        for item in res["skipped"][:10]:
            print(f"  - {item}")


if __name__ == "__main__":
    main()


