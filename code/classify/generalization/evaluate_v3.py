import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def preprocess_datasets(train_df, val_df, test_df, target_col):
    """三阶段数据集预处理"""
    # 特征对齐
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
                          val_finetune=False,
                          test_finetune_ratio=0.0):
    """跨数据集评估（支持双重微调）"""
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
    
    # 测试集微调
    if test_finetune_ratio > 0:
        print(f"使用{test_finetune_ratio*100:.0f}%测试集数据进行微调")
        X_finetune, X_test_remain, y_finetune, y_test_remain = train_test_split(
            X_test, y_test,
            train_size=test_finetune_ratio,
            stratify=y_test,
            random_state=42
        )
        model.fit(X_finetune, y_finetune)
        X_test, y_test = X_test_remain, y_test_remain
    else:
        X_test_remain, y_test_remain = X_test, y_test
    
    # 最终评估
    y_pred = model.predict(X_test)
    
    # 评估指标
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    # 分类报告
    print("\n=== 评估结果 ===")
    print(f"标签映射：{label_map}")
    print(f"验证集微调：{val_finetune}")
    print(f"测试集微调比例：{test_finetune_ratio*100:.0f}%")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, 
                               target_names=[f"Class {k}" for k in label_map.keys()],
                               zero_division=0))
    
    return accuracy, f1_macro

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='跨数据集分类评估工具（双重微调）')
    parser.add_argument('--train', required=True, help='训练集路径')
    parser.add_argument('--val', required=True, help='验证集路径')
    parser.add_argument('--test', required=True, help='测试集路径')
    parser.add_argument('--model', help='预训练模型路径')
    parser.add_argument('--target_col', default='Label', help='标签列名')
    
    # 微调参数
    parser.add_argument('--val_finetune', action='store_true', 
                       help='是否使用验证集进行领域适应')
    parser.add_argument('--test_finetune_ratio', type=float, default=0.0,
                       help='测试集微调比例（0.0-1.0）')
    
    args = parser.parse_args()

    # 参数校验
    if not (0.0 <= args.test_finetune_ratio <= 1.0):
        raise ValueError("测试集微调比例必须在0.0到1.0之间")

    # 加载模型
    if args.model:
        print(f"加载预训练模型：{args.model}")
        model = joblib.load(args.model)
    else:
        model = RandomForestClassifier(random_state=42)
        print("使用默认随机森林模型")

    # 执行评估
    acc, f1 = evaluate_cross_dataset(
        args.train,
        args.val,
        args.test,
        model=model,
        target_col=args.target_col,
        val_finetune=args.val_finetune,
        test_finetune_ratio=args.test_finetune_ratio
    )

    print("\n最终评估指标：")
    print(f"准确率：{acc:.4f}")
    print(f"F1分数（Macro）：{f1:.4f}")

'''
调用方法：python evaluate.py --train A.csv --val B_val.csv --test B.csv --val_finetune --test_finetune_ratio 0.1
'''