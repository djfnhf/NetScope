import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod

# 参数配置
data_folder = "path/to/output"  # 数据集划分后的文件夹路径（包含train.csv, val.csv, test.csv）
adv_folder = "path/to/adv_data"  # 对抗样本输出路径
model_save_path = "model_weights.pth"  # 模型权重保存路径（可选）
random_seed = 42

# 创建输出目录
os.makedirs(adv_folder, exist_ok=True)

# 固定随机种子（确保可复现）
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# -------------------------- 1. 加载已划分的数据集 --------------------------
def load_dataset(data_path):
    """加载数据集并分离特征和标签"""
    df = pd.read_csv(data_path)
    X = df.drop("Label", axis=1).values.astype(np.float32)
    y = df["Label"].values
    return X, y

# 加载训练集、验证集、测试集
X_train, y_train = load_dataset(os.path.join(data_folder, "train.csv"))
X_val, y_val = load_dataset(os.path.join(data_folder, "val.csv"))
X_test, y_test = load_dataset(os.path.join(data_folder, "test.csv"))

# -------------------------- 2. 数据标准化 --------------------------
# 注意：标准化参数必须与训练模型时一致
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # 仅用训练集拟合
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# -------------------------- 3. 定义并加载模型 --------------------------
class TrafficClassifier(nn.Module):
    """定义一个简单的神经网络模型（需与训练时结构一致）"""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.fc(x)

# 初始化模型
model = TrafficClassifier(
    input_dim=X_train.shape[1],
    num_classes=len(np.unique(y_train))
)

# 加载预训练权重（如果已有）
if os.path.exists(model_save_path):
    model.load_state_dict(torch.load(model_save_path))
    print(f"已加载预训练权重: {model_save_path}")
else:
    print("未找到预训练模型，需重新训练...")
    # 训练模型（此处为简化示例，实际需补充完整训练逻辑）
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(10):
        inputs = torch.FloatTensor(X_train)
        labels = torch.LongTensor(y_train)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # 保存模型权重（可选）
    torch.save(model.state_dict(), model_save_path)

# -------------------------- 4. 包装模型为ART分类器 --------------------------
art_classifier = PyTorchClassifier(
    model=model,
    loss=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters()),
    input_shape=(X_train.shape[1],),
    nb_classes=len(np.unique(y_train)),
    clip_values=(-5, 5)  # 限制扰动范围，防止特征值越界
)

# -------------------------- 5. 生成对抗样本 --------------------------
def generate_adv_samples(attack, X, y, original_df, output_name):
    """生成并保存对抗样本"""
    # 生成对抗样本
    X_adv = attack.generate(X)
    
    # 后处理：修正离散特征（示例：协议类型必须为整数）
    # 假设协议类型是第5列（索引4），取值范围为0,1,2
    protocol_col_idx = 4
    X_adv[:, protocol_col_idx] = np.clip(X_adv[:, protocol_col_idx], 0, 2)
    X_adv[:, protocol_col_idx] = np.round(X_adv[:, protocol_col_idx]).astype(int)
    
    # 保存为CSV
    adv_df = pd.DataFrame(X_adv, columns=original_df.drop("Label", axis=1).columns)
    adv_df["Label"] = y  # 保留原始标签
    adv_df.to_csv(os.path.join(adv_folder, output_name), index=False)
    print(f"已生成对抗样本: {output_name}")

# 初始化攻击方法（以FGSM为例）
attack = FastGradientMethod(
    estimator=art_classifier,
    eps=0.1  # 扰动强度（根据特征范围调整）
)

# 加载原始数据用于列名匹配（仅需加载一次）
test_df = pd.read_csv(os.path.join(data_folder, "test.csv"))

# 生成测试集对抗样本
generate_adv_samples(attack, X_test, y_test, test_df, "test_adv.csv")

# 可选：生成验证集对抗样本
# val_df = pd.read_csv(os.path.join(data_folder, "val.csv"))
# generate_adv_samples(attack, X_val, y_val, val_df, "val_adv.csv")

# -------------------------- 6. 评估对抗样本效果 --------------------------
def evaluate(model, X, y):
    """评估模型在数据集上的准确率"""
    y_pred = np.argmax(model.predict(X), axis=1)
    return accuracy_score(y, y_pred)

# 原始测试集性能
acc_clean = evaluate(art_classifier, X_test, y_test)
# 对抗测试集性能
X_test_adv, _ = load_dataset(os.path.join(adv_folder, "test_adv.csv"))
X_test_adv = scaler.transform(X_test_adv)  # 需重新标准化（如果对抗样本未标准化保存）
acc_adv = evaluate(art_classifier, X_test_adv, y_test)

print(f"[原始测试集] 准确率: {acc_clean:.4f}")
print(f"[对抗测试集] 准确率: {acc_adv:.4f}")
print(f"攻击成功率: {1 - acc_adv/acc_clean:.2%}")