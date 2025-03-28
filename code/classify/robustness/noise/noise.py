import pandas as pd
import numpy as np
import random

# 定义攻击类型列表
attack_types = [
    "FTP-Patator - Attempted", "FTP-Patator", "SSH-Patator", "SSH-Patator - Attempted",
    "DoS Slowloris", "DoS Slowloris - Attempted", "DoS Slowhttptest", "DoS Slowhttptest - Attempted",
    "DoS Hulk", "DoS Hulk - Attempted", "DoS GoldenEye", "Heartbleed", "DoS GoldenEye - Attempted",
    "Web Attack - Brute Force - Attempted", "Web Attack - Brute Force", "Infiltration - Attempted",
    "Infiltration", "Infiltration - Portscan", "Web Attack - XSS - Attempted", "Web Attack - XSS",
    "Web Attack - SQL Injection - Attempted", "Web Attack - SQL Injection", "Botnet - Attempted",
    "Botnet", "Portscan", "DDoS"
]


# 读取Excel文件
file_path = '/data2/intern/sunyu/TMbench/data/CICIDS2017_classification/CICIDS2018_classification_train.csv'  # 替换为你的Excel文件路径
df = pd.read_excel(file_path)

# 确保标签列存在
if 'label' not in df.columns:
    raise ValueError("The 'Label' column is missing in the CSV file")

# 随机选择10%的样本
num_samples = len(df)
num_noisy_samples = int(0.1 * num_samples)
noisy_indices = np.random.choice(df.index, num_noisy_samples, replace=False)

# 更改标签
for idx in noisy_indices:
    if df.at[idx, 'label'] != 'Benign':
        df.at[idx, 'label'] = 'Benign'
    elif df.at[idx, 'label'] == 'Benign':
         df.at[idx, 'label'] = random.choice(attack_types)
    # 如果有其他标签，可以在这里添加更多的条件

# 保存修改后的数据集
output_file_path = '/data2/intern/sunyu/TMbench/data/CICIDS2017_classification/noisy_dataset.csv'  # 替换为你想保存的文件路径
df.to_excel(output_file_path, index=False)

print(f"The labels of {num_noisy_samples} samples have been successfully changed and saved to {output_file_path}")