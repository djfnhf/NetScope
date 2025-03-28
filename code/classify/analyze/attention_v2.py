import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, mutual_info_score
from sklearn.ensemble import RandomForestClassifier
from captum.attr import Saliency
import importlib.util

# 命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--model_weights_path', type=str, required=True)
parser.add_argument('--model_def_path', type=str, required=True)
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--model_class', type=str, default='AttentionModel', 
                    help='模型类名称（默认：AttentionModel）')
args = parser.parse_args()

# 动态加载模型定义
def load_model_class(module_path, class_name):
    spec = importlib.util.spec_from_file_location("model_module", module_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    return getattr(model_module, class_name)

# 数据集类
class NetFlowDataset(torch.utils.data.Dataset):
    IMPORTANT_FEATURES = [
        'Dst Port', 'Protocol', 'Flow Duration', 'Tot Fwd Pkts',
        'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
        'Flow Byts/s', 'Flow Pkts/s'
    ]
    
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        
        # 处理标签列
        if 'Label' not in df.columns:
            df.columns = df.columns.str.strip().str.title()
        if 'Label' not in df.columns:
            raise ValueError("数据必须包含'Label'列")
        
        # 处理特征和标签
        self.feature_names = df.columns[df.columns != 'Label'].tolist()
        self.data = torch.tensor(df.drop(columns='Label').values, dtype=torch.float32)
        self.labels = torch.tensor(df['Label'].values, dtype=torch.long)
        
        # 计算真实特征重要性
        self.true_importance = self._calculate_feature_importance(df)
        
        # 打印数据摘要
        print(f"加载数据: {len(self)} samples")
        print(f"特征维度: {self.data.shape[1]}")
        print(f"标签分布: {torch.bincount(self.labels)}")
        
    def _calculate_feature_importance(self, df):
        """使用统计方法计算特征重要性"""
        X = df.drop(columns='Label')
        y = df['Label']
        
        # 方法1：互信息
        mi_scores = []
        for col in X.columns:
            mi_scores.append(mutual_info_score(y, X[col]))
        
        # 方法2：随机森林特征重要性
        rf = RandomForestClassifier(n_estimators=10)
        rf.fit(X, y)
        rf_scores = rf.feature_importances_
        
        # 综合评分（可调整权重）
        final_scores = np.array(mi_scores) * 0.4 + np.array(rf_scores) * 0.6
        
        # 归一化
        return torch.tensor(final_scores / final_scores.max())

    def get_true_feature_importance(self):
        return self.true_importance

# 主分析类
class AttentionValidator:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
        self.feature_names = dataset.feature_names

    def analyze_attention(model, dataloader, feature_names):
            model.eval()
            all_weights = []
            all_labels = []
    
            with torch.no_grad():
                for inputs, labels in dataloader:
                    _, attn_weights = model(inputs)
                    all_weights.append(attn_weights.cpu().numpy())
                    all_labels.append(labels.numpy())
    
            weights = np.concatenate(all_weights)
            labels = np.concatenate(all_labels)
    
            # 计算类别相关注意力模式
            class_attention = {}
            for cls in np.unique(labels):
                class_attention[cls] = weights[labels == cls].mean(axis=0)
    
            # 可视化关键特征
            plt.figure(figsize=(12,6))
            for i, cls in enumerate(class_attention):
                plt.subplot(1, len(class_attention), i+1)
                top_features = np.argsort(class_attention[cls].mean(axis=0))[-5:]
                plt.barh(feature_names[top_features], 
                        class_attention[cls].mean(axis=0)[top_features])
                plt.title(f"Class {cls} Important Features")
            plt.tight_layout()
            plt.show()

    def validate_attention_correlation(self):
        saliency = Saliency(self.model)
        aucs = []
        true_importance = self.dataset.get_true_feature_importance().numpy()

        for inputs, labels in self.dataloader:
            inputs.requires_grad_()
            outputs, _ = self.model(inputs)
            preds = torch.argmax(outputs, dim=1)
            
            # 计算Saliency Maps
            attributions = saliency.attribute(inputs, target=preds)
            attributions = attributions.abs().mean(dim=(0,2)).numpy()
            
            # 计算AUC
            auc = roc_auc_score(true_importance, attributions)
            aucs.append(auc)
        
        print(f"Attention Correlation AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}") # AUC > 0.7 表明注意力与真实特征相关

# 执行流程
if __name__ == "__main__":
    # 1. 加载数据
    dataset = NetFlowDataset(args.data_path)
    
    # 2. 加载模型
    ModelClass = load_model_class(args.model_def_path, args.model_class)
    model = ModelClass(input_dim=dataset.data.shape[-1], 
                      num_classes=len(torch.unique(dataset.labels)))
    model.load_state_dict(torch.load(args.model_weights_path))
    
    # 3. 执行验证
    validator = AttentionValidator(model, dataset)
    validator.analyze_attention()
    validator.validate_attention_correlation()