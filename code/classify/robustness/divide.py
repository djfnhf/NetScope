import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse

# 参数解析
parser = argparse.ArgumentParser(description='Dataset splitting script')
parser.add_argument('--input', '-i', required=True, help='Input folder containing CSV files')
parser.add_argument('--output', '-o', required=True, help='Output folder for split datasets')
args = parser.parse_args()

# 验证输入路径存在
if not os.path.exists(args.input):
    raise FileNotFoundError(f"Input directory '{args.input}' does not exist")

# 创建输出目录
os.makedirs(args.output, exist_ok=True)

# 1. 合并所有CSV文件
all_files = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith(".csv")]
df_list = [pd.read_csv(file) for file in all_files]
combined_df = pd.concat(df_list, ignore_index=True)

# 检查是否存在标签列
if "Label" not in combined_df.columns:
    raise ValueError("The 'Label' column is missing in the CSV files")

# 2. 分层划分数据集（保持原有划分逻辑不变）
train_df, temp_df = train_test_split(
    combined_df,
    test_size=0.5,
    stratify=combined_df["Label"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["Label"],
    random_state=42
)

# 3. 保存划分后的数据集
train_df.to_csv(os.path.join(args.output, "train.csv"), index=False)
val_df.to_csv(os.path.join(args.output, "val.csv"), index=False)
test_df.to_csv(os.path.join(args.output, "test.csv"), index=False)

print("Dataset split completed successfully!")
print(f"Training set: {len(train_df)} samples")
print(f"Validation set: {len(val_df)} samples")
print(f"Test set: {len(test_df)} samples")