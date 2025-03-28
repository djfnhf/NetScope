import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def align_features(train_df, test_df):
    """特征对齐处理"""
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    
    # 找出共同特征
    common_cols = train_cols.intersection(test_cols)
    if 'label' in common_cols:
        common_cols.remove('label')
    
    # 处理缺失特征
    missing_in_test = train_cols - test_cols
    if missing_in_test:
        print(f"警告：测试集缺失特征 {missing_in_test}，将从训练集中删除这些特征")
    
    # 处理多余特征
    extra_in_test = test_cols - train_cols
    if extra_in_test:
        print(f"警告：测试集包含多余特征 {extra_in_test}，将自动删除")
    
    # 对齐特征
    aligned_train = train_df[list(common_cols) + ['label']]
    aligned_test = test_df[list(common_cols) + ['label']]
    
    return aligned_train, aligned_test

def evaluate_traffic_classifier(train_path, test_path, model=None, target_col='label', finetune=False):
    """增强版评估函数"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # 特征对齐处理
    train_df, test_df = align_features(train_df, test_df)
    
    # 分离特征和标签
    X_train = train_df.drop(columns=[target_col]).values
    y_train = train_df[target_col].values
    X_test = test_df.drop(columns=[target_col]).values
    y_test = test_df[target_col].values
    
    # 处理缺失值
    def process_missing(X, y):
        mask = ~pd.isnull(X).any(axis=1)
        return X[mask], y[mask]
    
    X_train, y_train = process_missing(X_train, y_train)
    X_test, y_test = process_missing(X_test, y_test)
    
    # 标签编码
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    
    # 特征标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 模型处理
    if model is None:
        model = RandomForestClassifier(random_state=42)
    
    # 微调模式
    if finetune:
        print("警告：正在使用测试集进行微调，这可能导致评估结果过于乐观！")
        model.fit(X_train, y_train)
        model.fit(X_test, y_test)  # 这里会污染模型
        
    # 正常评估模式
    else:
        model.fit(X_train, y_train)
    
    # 评估
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    return accuracy, f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='网络流量分类评估工具（支持特征对齐）')
    parser.add_argument('--train', required=True, help='训练集路径')
    parser.add_argument('--test', required=True, help='测试集路径')
    parser.add_argument('--model', help='预训练模型路径')
    parser.add_argument('--target_col', default='label', help='标签列名')
    parser.add_argument('--finetune', action='store_true', help='是否使用测试集微调')
    
    args = parser.parse_args()

    # 加载模型
    if args.model:
        print(f"加载预训练模型：{args.model}")
        model = joblib.load(args.model)
    else:
        model = RandomForestClassifier(random_state=42)
        print("使用默认随机森林模型")

    # 执行评估
    acc, f1 = evaluate_traffic_classifier(
        args.train,
        args.test,
        model=model,
        target_col=args.target_col,
        finetune=args.finetune
    )

    # 输出结果
    print("\n评估结果：")
    print(f"准确率：{acc:.4f}")
    print(f"F1分数：{f1:.4f}")
    if args.finetune:
        print("注意：结果包含测试集微调，实际部署时请谨慎使用！")