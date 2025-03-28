import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def preprocess_datasets(train_df, val_df, test_df, target_col):
    """三阶段数据集预处理"""
    common_features = list(
        set(train_df.columns) & 
        set(val_df.columns) & 
        set(test_df.columns) - {target_col}
    )
    
    datasets = [train_df, val_df, test_df]
    processed = []
    for df in datasets:
        df = df[common_features + [target_col]].dropna()
        processed.append(df)
    
    return tuple(processed)

def align_labels(datasets, target_col):
    """统一标签编码"""
    all_labels = set()
    for df in datasets:
        all_labels.update(df[target_col].unique())
    
    label_map = {lbl: idx for idx, lbl in enumerate(sorted(all_labels))}
    
    processed = []
    for df in datasets:
        df = df.copy()
        df[target_col] = df[target_col].map(label_map)
        processed.append(df)
    
    return processed, label_map

def evaluate_cross_dataset(train_path, val_path, test_path, model=None, 
                          target_col='Label', 
                          val_finetune=False):
    # 加载数据
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    # 预处理与特征对齐
    train_df, val_df, test_df = preprocess_datasets(
        train_df, val_df, test_df, target_col
    )
    
    # 标签对齐
    (train_df, val_df, test_df), label_map = align_labels(
        [train_df, val_df, test_df], target_col
    )
    
    # 分离特征和标签
    X_train = train_df.drop(columns=[target_col]).values
    y_train = train_df[target_col].values
    X_val = val_df.drop(columns=[target_col]).values
    y_val = val_df[target_col].values
    X_test = test_df.drop(columns=[target_col]).values
    y_test = test_df[target_col].values
    
    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # 初始化模型
    if model is None:
        model = RandomForestClassifier(random_state=42)
    
    # 训练流程
    model.fit(X_train, y_train)
    
    # 验证集微调
    if val_finetune:
        print("使用验证集进行领域适应...")
        model.fit(X_val, y_val)
    
    # 最终评估
    y_pred = model.predict(X_test)
    
    # 评估指标
    accuracy = accuracy_score(y_test, y_pred)
    
    # 输出结果
    print("\n=== 评估结果 ===")
    print(f"标签映射：{label_map}")
    print(f"验证集微调：{val_finetune}")
    print(f"准确率：{accuracy:.4f}")
    
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='数据集评估工具')
    # 数据集路径
    parser.add_argument('--train', required=True, help='训练集路径')
    parser.add_argument('--val', required=True, help='验证集路径')
    parser.add_argument('--test', required=True, help='测试集路径')
    # 其他参数
    parser.add_argument('--model', help='预训练模型路径')
    parser.add_argument('--target_col', default='Label', help='标签列名')
    parser.add_argument('--val_finetune', action='store_true', 
                       help='是否使用验证集进行领域适应')
    
    args = parser.parse_args()

    # 评估数据集
    print("评估数据集...")
    model = joblib.load(args.model) if args.model else RandomForestClassifier(random_state=42)
    acc_clean = evaluate_cross_dataset(
        args.train,
        args.val,
        args.test,
        model=model,
        target_col=args.target_col,
        val_finetune=args.val_finetune
    )

    # 输出最终结果
    print("\n最终评估指标：")
    print(f"准确率 (acc): {acc_clean:.4f}")