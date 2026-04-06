import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from data import fMRI_Dataset
from model import SyncAsyncBrainNet

def calculate_metrics(output, target):
    # 计算准确率、敏感性、特异性
    pred = torch.argmax(output, dim=1)
    correct = (pred == target).sum().item()
    acc = correct / len(target)
    
    # 二分类指标
    if len(torch.unique(target)) == 2:
        tp = ((pred == 1) & (target == 1)).sum().item()
        tn = ((pred == 0) & (target == 0)).sum().item()
        fp = ((pred == 1) & (target == 0)).sum().item()
        fn = ((pred == 0) & (target == 1)).sum().item()
        
        sen = tp / (tp + fn) if (tp + fn) > 0 else 0
        spe = tn / (tn + fp) if (tn + fp) > 0 else 0
        return acc, sen, spe
    return acc, 0, 0

def main():
    # 配置参数
    args = {
        'data_dir': './data',
        'batch_size': 16,
        'lr': 0.001,
        'epochs': 50,
        'k_folds': [1, 2, 3, 4, 5],
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    
    # 5折交叉验证
    all_acc = []
    all_sen = []
    all_spe = []
    
    for fold in args['k_folds']:
        print(f"Fold {fold}:")
        
        # 数据加载
        train_dataset = fMRI_Dataset(args['data_dir'], fold, 'train')
        test_dataset = fMRI_Dataset(args['data_dir'], fold, 'test')
        
        train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        
        # 模型初始化
        model = SyncAsyncBrainNet().to(args['device'])
        optimizer = optim.Adam(model.parameters(), lr=args['lr'])
        criterion = nn.CrossEntropyLoss()
        
        # 训练
        for epoch in range(args['epochs']):
            model.train()
            train_loss = 0
            for data, target in train_loader:
                data, target = data.to(args['device']), target.to(args['device'])
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # 测试
            model.eval()
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(args['device']), target.to(args['device'])
                    output = model(data)
                    acc, sen, spe = calculate_metrics(output, target)
            
            print(f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, Acc={acc:.4f}, Sen={sen:.4f}, Spe={spe:.4f}")
        
        all_acc.append(acc)
        all_sen.append(sen)
        all_spe.append(spe)
    
    # 输出平均结果
    print("\nFinal Results:")
    print(f"Average Acc: {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"Average Sen: {np.mean(all_sen):.4f} ± {np.std(all_sen):.4f}")
    print(f"Average Spe: {np.mean(all_spe):.4f} ± {np.std(all_spe):.4f}")

if __name__ == "__main__":
    main()
