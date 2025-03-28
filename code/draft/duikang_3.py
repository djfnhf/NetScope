import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_weights_path",type=str,required=True,help="预训练模型权重路径（.pth文件）")
    parser.add_argument("--model_def_path",type=str,required=True,help="模型定义文件路径（.py文件）")
    parser.add_argument('--model_class', type=str, default='AttentionModel', help="模型类名称（默认：AttentionModel）")
    parser.add_argument("--data_folder",type=str,default="data",help="已划分数据集的目录（默认：data）")
    parser.add_argument("--adv_folder",type=str,default="adv_samples",help="对抗样本输出目录（默认：adv_samples）")
    parser.add_argument("--random_seed",type=int,default=42,help="随机种子（默认：42）")
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()

    # 创建输出目录
    os.makedirs(args.adv_folder, exist_ok=True)

    # 固定随机种子
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # -------------------------- 1. 加载数据 --------------------------
    def load_dataset(data_path):
        df = pd.read_csv(data_path)
        X = df.drop("Label", axis=1).values.astype(np.float32)
        y = df["Label"].values
        return X, y

    # 加载训练集（用于标准化参数）
    X_train, _ = load_dataset(os.path.join(args.data_folder, "CICIDS2018_classification_train.csv"))

    # 加载测试集（生成对抗样本）
    X_test, y_test = load_dataset(os.path.join(args.data_folder, "CICIDS2018_classification_test.csv"))
    test_df = pd.read_csv(os.path.join(args.data_folder, "CICIDS2018_classification_train.csv"))

    # -------------------------- 2. 数据标准化 --------------------------
    scaler = StandardScaler()
    scaler.fit(X_train)  # 仅用训练集拟合
    X_test = scaler.transform(X_test)

    # -------------------------- 3. 加载预训练模型 --------------------------
    # 添加模型定义文件所在目录到Python路径
    model_def_dir = os.path.dirname(args.model_def_path)
    sys.path.append(model_def_dir)

    # 动态导入模型类（使用命令行参数指定的类名）
    module_name = os.path.splitext(os.path.basename(args.model_def_path))[0]
    model_module = __import__(module_name)
    ModelClass = getattr(model_module, args.model_class)

    # 初始化模型（参数需与训练时一致）
    model = ModelClass(
        input_dim=X_train.shape[1],
        num_classes=len(np.unique(y_test))
    )

    # 加载预训练权重
    if not os.path.exists(args.model_weights_path):
        raise FileNotFoundError(f"Model weights file {args.model_weights_path} does not exist!")
    model.load_state_dict(torch.load(args.model_weights_path))

    # -------------------------- 4. 包装模型为ART分类器 --------------------------
    art_classifier = PyTorchClassifier(
        model=model,
        loss=torch.nn.CrossEntropyLoss(),
        input_shape=(X_train.shape[1],),
        nb_classes=len(np.unique(y_test)),
        clip_values=(-5, 5)
    )

    # -------------------------- 5. 生成对抗样本 --------------------------
    attack = FastGradientMethod(estimator=art_classifier, eps=0.1)
    X_test_adv = attack.generate(X_test)

    # -------------------------- 6. 后处理离散特征 --------------------------
    # 示例：修正协议类型为合法整数（假设是第5列）
    protocol_col_idx = 4
    X_test_adv[:, protocol_col_idx] = np.clip(X_test_adv[:, protocol_col_idx], 0, 2)
    X_test_adv[:, protocol_col_idx] = np.round(X_test_adv[:, protocol_col_idx]).astype(int)

    # -------------------------- 7. 保存对抗样本 --------------------------
    adv_df = pd.DataFrame(X_test_adv, columns=test_df.drop("Label", axis=1).columns)
    adv_df["Label"] = y_test
    adv_df.to_csv(os.path.join(args.adv_folder, "CICIDS2018_classification_test_adv.csv"), index=False)
    print(f"Adversarial examples have been saved to: {os.path.join(args.adv_folder, 'CICIDS2018_classification_test_adv.csv')}")

if __name__ == "__main__":
    main()