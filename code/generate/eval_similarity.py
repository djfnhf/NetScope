import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
from scipy.spatial import distance
import argparse
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import importlib.util
import sys

class CSVDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file).values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)

def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=5,
    foldername="",
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    output_path = f"{foldername}/model.pth" if foldername else ""
    best_valid_loss = float('inf')
    
    # 学习率调度器
    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )
    
    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch}/{config['epochs']}")
        model.train()
        train_loss = 0.0
        
        # 训练阶段
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as pbar:
            for batch in pbar:
                optimizer.zero_grad()
                loss = model(batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                pbar.set_postfix({"Train Loss": loss.item()})
        
        # 验证阶段
        if valid_loader and (epoch % valid_epoch_interval == 0):
            model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for batch in valid_loader:
                    loss = model(batch)
                    valid_loss += loss.item()
            
            # 保存最佳模型
            avg_valid_loss = valid_loss / len(valid_loader)
            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                if foldername:
                    torch.save(model.state_dict(), output_path)
                    print(f"New best model saved with loss: {best_valid_loss:.4f}")
            
            print(f"Validation Loss: {avg_valid_loss:.4f}")
        
        lr_scheduler.step()  


def crps(y_true, y_pred, sample_weight=None):
     num_samples=y_pred.shape[0]
     absolute_error=np.mean(np.abs(y_pred-y_true), axis=0)
 
     if num_samples==1:
         return np.average(absolute_error, weights=sample_weight)
 
     y_pred=np.sort(y_pred, axis=0)
     diff=y_pred[1:] -y_pred[:-1]
     weight=np.arange(1, num_samples) *np.arange(num_samples-1, 0, -1)
     weight=np.expand_dims(weight, -1)
 
     per_obs_crps=absolute_error-np.sum(diff*weight, axis=0) /num_samples**2
     return np.average(per_obs_crps, weights=sample_weight)

def distribution_jsd(generated_data, real_dataset):

    
    
    
    n_real = real_dataset.flatten()
    n_gene = generated_data.flatten()
    JSD = distance.jensenshannon(n_real.flatten(), n_gene.flatten(), 2.0)
    
    return JSD
    

    
    
def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername=""):
    with torch.no_grad():
        model.eval()
        # 初始化指标存储
        feature_metrics = {
            'jsd': [],
            'tv': [],
            'crps': []
        }

        for batch_no, test_batch in enumerate(tqdm(test_loader)):
            output = model.evaluate(test_batch, nsample)
            samples, c_targets, _ = output
            samples = samples.permute(0, 1, 3, 2)  # (B, nsample, L, K)
            c_targets = c_targets.permute(0, 2, 1)  # (B, L, K)
            
            B, L, K = c_targets.shape
            
            # 遍历每个特征维度
            for k in range(K):
                # 提取当前特征的数据
                feature_samples = samples[..., k].cpu().numpy().flatten()
                feature_targets = c_targets[..., k].cpu().numpy().flatten()
                
                # 计算JSD
                jsd = distribution_jsd(feature_samples, feature_targets)
                feature_metrics['jsd'].append(jsd)
                
                # 计算TV距离
                tv = 0.5 * np.sum(np.abs(feature_samples - feature_targets)) / len(feature_targets)
                feature_metrics['tv'].append(tv)
                
                # 计算CRPS
                crps_val = crps(feature_targets, feature_samples)
                feature_metrics['crps'].append(crps_val)

        # 计算平均指标
        avg_metrics = {
            'jsd': np.mean(feature_metrics['jsd']),
            'tv': np.mean(feature_metrics['tv']),
            'crps': np.mean(feature_metrics['crps'])
        }

        # 保存结果
        with open(f"{foldername}/result_nsample{nsample}.pk", "wb") as f:
            pickle.dump(avg_metrics, f)
        
        print(f"Average JSD: {avg_metrics['jsd']:.4f}")
        print(f"Average TV Distance: {avg_metrics['tv']:.4f}")
        print(f"Average CRPS: {avg_metrics['crps']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--valid_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_structure", type=str, required=True, 
                      help="Path to model structure Python file (without .py extension)")
    parser.add_argument("--model_class", type=str, required=True, 
                  help="Name of the model class in the structure file")
    parser.add_argument("--mode", type=str, choices=["train", "eval"], required=True)
    args = parser.parse_args()

    # 动态导入模型结构
    try:
        # 加载模型结构模块
        model_module = importlib.import_module(args.model_structure)
        # 假设模型类名为 YourModelClass
        model_class = getattr(model_module, args.model_class)
        model = model_class()
    except ImportError:
        print(f"Error: Could not import model structure from {args.model_structure}.py")
        sys.exit(1)
    except AttributeError:
        print(f"Error: Model class 'ModelClass' not found in {args.model_structure}.py")
        sys.exit(1)

    # 创建DataLoaders
    train_dataset = CSVDataset(args.train_data)
    valid_dataset = CSVDataset(args.valid_data)
    test_dataset = CSVDataset(args.test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    if args.mode == "train":
        train(
            model,
            config={"lr": 0.001, "epochs": 100},
            train_loader=train_loader,
            valid_loader=valid_loader,
            valid_epoch_interval=5,
            foldername=args.model_path
        )
        torch.save(model.state_dict(), f"{args.model_path}/model.pth")
    else:
        model.load_state_dict(torch.load(f"{args.model_path}/model.pth"))
        evaluate(model, test_loader, foldername=args.model_path)

# 训练时指定模型结构文件
# python script.py --train_data train.csv --valid_data valid.csv --test_data test.csv \
#                 --model_path ./output --model_struct ./ --model_class abc --mode train

# 评估时指定模型结构文件
# python script.py --test_data test.csv --model_path ./output --model_struct ./ --model_class abc --mode eval