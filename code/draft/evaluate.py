import argparse
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import joblib

def evaluate_traffic_classifier(train_path, val_path, test_path, model=None, target_col='Label'):
    """
    网络流量分类模型评估函数
    
    参数:
    train_path (str): 训练集CSV文件路径
    val_path (str): 验证集CSV文件路径
    test_path (str): 测试集CSV文件路径
    model: 机器学习模型实例（默认随机森林）
    target_col (str): 标签列名称
    
    返回:
    accuracy (float): 测试集准确率
    f1 (float): 测试集F1分数
    """
    
    # 读取数据集
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    # 分离特征和标签
    X_train = train_df.drop(columns=[target_col]).values
    y_train = train_df[target_col].values
    X_val = val_df.drop(columns=[target_col]).values
    y_val = val_df[target_col].values
    X_test = test_df.drop(columns=[target_col]).values
    y_test = test_df[target_col].values
    
    # 处理缺失值（简单删除含缺失值的样本）
    def process_missing(X, y):
        mask = ~pd.isnull(X).any(axis=1)
        return X[mask], y[mask]
    
    X_train, y_train = process_missing(X_train, y_train)
    X_val, y_val = process_missing(X_val, y_val)
    X_test, y_test = process_missing(X_test, y_test)
    
    # 标签编码
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_val = le.transform(y_val)
    y_test = le.transform(y_test)
    
    # 特征标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # 使用默认模型（随机森林）
    if model is None:
        model = RandomForestClassifier(random_state=42)
    
    # 模型训练
    model.fit(X_train, y_train)
    
    # 在测试集预测
    y_pred = model.predict(X_test)
    
    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')  # 使用macro平均处理多分类
    
    return accuracy, f1

# 使用示例
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='网络流量分类模型评估工具')
    
    # 必需参数
    parser.add_argument('--train', required=True, help='训练集CSV文件路径')
    parser.add_argument('--val', required=True, help='验证集CSV文件路径')
    parser.add_argument('--test', required=True, help='测试集CSV文件路径')
    
    # 可选参数
    parser.add_argument('--model_weights', help='预训练模型权重路径（joblib格式）')
    parser.add_argument('--target_col', default='Label', help='CSV中的标签列名（默认：Label）')
    
    args = parser.parse_args()

    # 加载模型（优先使用预训练模型）
    if args.model_weights:
        print(f"加载预训练模型: {args.model_weights}")
        model = joblib.load(args.model_weights)
    else:
        print("使用默认随机森林模型")
        model = RandomForestClassifier(random_state=42)

    # 执行评估流程
    accuracy, f1 = evaluate_traffic_classifier(
        args.train,
        args.val,
        args.test,
        model=model,
        target_col=args.target_col
    )

    # 输出结果
    print("\n评估结果:")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"F1分数 (Macro): {f1:.4f}")