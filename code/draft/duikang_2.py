import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod

# 参数配置
data_folder = "path/to/output"       # 已划分数据集的目录（包含 train.csv/test.csv）
adv_folder = "path/to/adv_samples"   # 对抗样本输出目录
model_weights_path = "model_weights.pth"  # 预训练模型权重路径
model_def_path = "path/to/model.py"  # 模型定义文件路径（需添加到系统路径）
random_seed = 42                     # 随机种子

# 创建输出目录
os.makedirs(adv_folder, exist_ok=True)

# 固定随机种子
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# -------------------------- 1. 加载数据 --------------------------
def load_dataset(data_path):
    """加载数据集并分离特征和标签"""
    df = pd.read_csv(data_path)
    X = df.drop("Label", axis=1).values.astype(np.float32)
    y = df["Label"].values
    return X, y

# 加载训练集（仅用于标准化参数拟合）
X_train, _ = load_dataset(os.path.join(data_folder, "train.csv"))

# 加载测试集（生成对抗样本）
X_test, y_test = load_dataset(os.path.join(data_folder, "test.csv"))
test_df = pd.read_csv(os.path.join(data_folder, "test.csv"))  # 用于列名匹配

# -------------------------- 2. 数据标准化 --------------------------
# 关键：使用训练集的均值和标准差标准化测试集
scaler = StandardScaler()
scaler.fit(X_train)          # 仅用训练集拟合
X_test = scaler.transform(X_test)

# -------------------------- 3. 加载预训练模型 --------------------------
# 添加模型定义文件所在目录到Python路径
sys.path.append(os.path.dirname(model_def_path))

# 从独立文件导入模型类（假设模型类名为 TrafficClassifier）
from model import TrafficClassifier # type: ignore

# 初始化模型（参数需与训练时完全一致）
model = TrafficClassifier(
    input_dim=X_train.shape[1],
    num_classes=len(np.unique(y_test))
)

# 加载预训练权重（必须存在）
if not os.path.exists(model_weights_path):
    raise FileNotFoundError(f"模型权重文件 {model_weights_path} 不存在！")
model.load_state_dict(torch.load(model_weights_path))

# -------------------------- 4. 包装模型为ART分类器 --------------------------
art_classifier = PyTorchClassifier(
    model=model,
    loss=nn.CrossEntropyLoss(),
    input_shape=(X_train.shape[1],),
    nb_classes=len(np.unique(y_test)),
    clip_values=(-5, 5)  # 限制扰动范围
)

# -------------------------- 5. 生成对抗样本 --------------------------
# 初始化FGSM攻击（白盒）
attack = FastGradientMethod(
    estimator=art_classifier,
    eps=0.1  # 扰动强度（根据特征范围调整）
)

# 生成对抗样本
X_test_adv = attack.generate(X_test)

# -------------------------- 6. 后处理离散特征 --------------------------
# 示例：修正协议类型为合法整数（假设是第5列）
protocol_col_idx = 4
X_test_adv[:, protocol_col_idx] = np.clip(X_test_adv[:, protocol_col_idx], 0, 2)
X_test_adv[:, protocol_col_idx] = np.round(X_test_adv[:, protocol_col_idx]).astype(int)

# -------------------------- 7. 保存对抗样本 --------------------------
# 转换为DataFrame并保留原始标签
adv_df = pd.DataFrame(X_test_adv, columns=test_df.drop("Label", axis=1).columns)
adv_df["Label"] = y_test

# 保存为CSV文件
adv_df.to_csv(os.path.join(adv_folder, "test_adv.csv"), index=False)
print(f"对抗样本已保存至：{os.path.join(adv_folder, 'test_adv.csv')}")