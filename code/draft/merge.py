import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 参数设置
input_folder = "/data2/intern/sunyu/data/generalization"  # 替换为你的CSV文件夹路径
output_folder = "/data2/intern/sunyu/data/generalization"         # 替换为输出文件夹路径
random_seed = 42                         # 随机种子，确保结果可复现

# 创建输出目录
os.makedirs(output_folder, exist_ok=True)

# 1. 合并所有CSV文件
all_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".csv")]
df_list = [pd.read_csv(file) for file in all_files]
combined_df = pd.concat(df_list, ignore_index=True)

# 检查是否存在标签列
if "Label" not in combined_df.columns:
    raise ValueError("The 'Label' column is missing in the CSV file")

# 2. 分层划分数据集
# 第一次划分：50%训练集，剩余50%为临时集
train_df, temp_df = train_test_split(
    combined_df,
    test_size=0.5,
    stratify=combined_df["Label"],
    random_state=random_seed
)

# 第二次划分：临时集按4:6划分为验证集（20%）和测试集（30%）
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.6,  # 60%的temp_df对应总数据的30%（0.5*0.6=0.3）
    stratify=temp_df["Label"],
    random_state=random_seed
)

# 3. 保存划分后的数据集
train_df.to_csv(os.path.join(output_folder, "CICIDS2018_classification_train.csv"), index=False)
val_df.to_csv(os.path.join(output_folder, "CICIDS2018_classification_val.csv"), index=False)
test_df.to_csv(os.path.join(output_folder, "CICIDS2018_classification_test.csv"), index=False)

print("Dataset split done!")
print(f"Number of training set samples: {len(train_df)}")
print(f"Number of validation set samples: {len(val_df)}")
print(f"Number of test set samples: {len(test_df)}")