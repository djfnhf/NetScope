import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def inject_network_errors(input_csv, output_csv, error_rate=0.3, severity_levels=[0.05, 0.15, 0.3]):
    """
    为测试集注入网络错误（乱序、重传、丢包）
    
    参数：
        input_csv (str): 原始测试集路径
        output_csv (str): 注入错误后的输出路径
        error_rate (float): 注入错误的样本比例（默认30%）
        severity_levels (list): 错误严重等级 [轻度, 中度, 重度] 的扰动比例
    """
    # 加载测试集
    df = pd.read_csv(input_csv)
    
    # 随机选择注入错误的样本
    error_samples, _ = train_test_split(
        df, test_size=1 - error_rate, random_state=42
    )
    normal_samples = df.drop(error_samples.index)
    
    # 定义错误注入函数
    def apply_errors(row, severity):
        # 随机选择注入的错误类型
        error_type = np.random.choice(["reorder", "retransmit", "loss"])
        
        # -------------------------- 乱序模拟 --------------------------
        if error_type == "reorder":
            # 增大流持续时间（按严重等级增加比例）
            row["Flow Duration"] *= (1 + severity * np.random.uniform(0.1, 0.5))
            
            # 增大包长方差（模拟乱序）
            row["Packet Length Variance"] *= (1 + severity * np.random.uniform(0.2, 1.0))
            
        # -------------------------- 重传模拟 --------------------------
        elif error_type == "retransmit":
            # 增加重传次数和总包数
            retransmit_times = int(severity * 10)  # 严重等级决定重传次数
            row["Total Fwd Packet"] += retransmit_times
            row["Total Bwd packets"] += retransmit_times
            
            # 降低传输速率（模拟重传延迟）
            # row["Fwd Packet Rate"] /= (1 + severity * np.random.uniform(0.5, 2.0))
            # row["Bwd Packet Rate"] /= (1 + severity * np.random.uniform(0.5, 2.0))
            
        # -------------------------- 丢包模拟 --------------------------
        elif error_type == "loss":
            # 减少总包数（按严重等级随机丢弃）
            loss_ratio = severity * np.random.uniform(0.1, 0.5)
            row["Total Fwd Packet"] = max(1, int(row["Total Fwd Packet"] * (1 - loss_ratio)))
            row["Total Bwd packets"] = max(1, int(row["Total Bwd packets"] * (1 - loss_ratio)))
            
            # 修改相关统计量（如平均包长）
            row["Avg Packet Size"] *= (1 - loss_ratio * 0.2)
            
        return row

    # 对每个错误样本分配严重等级并注入错误
    error_data = []
    for idx, row in error_samples.iterrows():
        # 随机分配严重等级（轻度/中度/重度）
        severity = np.random.choice(severity_levels, p=[0.4, 0.4, 0.2])  # 等级概率可调
        modified_row = apply_errors(row.copy(), severity)
        error_data.append(modified_row)
    
    # 合并数据并保存
    error_df = pd.DataFrame(error_data)
    final_df = pd.concat([normal_samples, error_df], axis=0)
    final_df.to_csv(output_csv, index=False)
    print(f"A test set with network errors has been generated: {output_csv}")

# 示例调用
if __name__ == "__main__":
    # 输入路径（假设测试集已划分）
    test_csv = "/data2/intern/sunyu/data/CICIDS2018_classification/CICIDS2018_classification_test.csv"
    # 输出路径
    adv_test_csv = "/data2/intern/sunyu/data/CICIDS2018_classification/CICIDS2018_classification_test.csv_adv_network_errors.csv"
    
    # 注入错误（30%样本，分三个严重等级）
    inject_network_errors(test_csv, adv_test_csv)