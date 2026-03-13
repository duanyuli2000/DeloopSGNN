"""
Experiment 1: Standard Task Accuracy Evaluation

本脚本用于在标准节点分类任务上评估模型的准确率

支持的数据集:
    - 同构图: Cora, Citeseer, Pubmed, Reddit, ogbn-arxiv
    - 异构图: Chameleon, Squirrel, Film, Texas, Wisconsin, Cornell

使用方法:
    python exp1_generalization.py --model DeloopSGNN --dataset cora --n_trials 3
"""

import argparse
import torch
import numpy as np
import pandas as pd
import yaml
import os

from data_utils import DataLoader, random_splits
from base_models import *


# 模型配置
MODEL_CONFIG = {
    'DeloopSGNN': {
        'class': DeloopSGNN,
        'params': {'K': 10, 'dropout': 0.5}
    },
    'ChebNet': {
        'class': ChebNet,
        'params': {'num_hops': 3, 'dropout': 0.5}
    },
    'BernNet': {
        'class': BernNet,
        'params': {'K': 8, 'dropout': 0.5}
    },
    # ... 其他模型配置
}


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Graph Classification Experiment')
    parser.add_argument('--model', type=str, default='DeloopSGNN',
                       help='Model name')
    parser.add_argument('--dataset', type=str, default='cora',
                       help='Dataset name')
    parser.add_argument('--n_trials', type=int, default=3,
                       help='Number of trials for averaging')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU id')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设备
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # 加载数据
    dataset, data = DataLoader(args.dataset)
    features = data.x
    labels = data.y
    edge_index = data.edge_index
    
    # 构建邻接矩阵
    n = features.shape[0]
    adj = np.zeros((n, n))
    adj[edge_index[0].numpy(), edge_index[1].numpy()] = 1
    
    # 数据划分
    data_split = random_splits(labels, 2000)
    idx_train, idx_val, idx_test = data_split[0], data_split[1], data_split[2]
    
    # 获取模型
    model_info = MODEL_CONFIG.get(args.model)
    if model_info is None:
        raise ValueError(f'Unknown model: {args.model}')
    
    # 运行实验
    accuracies = []
    for trial in range(args.n_trials):
        print(f'\n=== Trial {trial+1}/{args.n_trials} ===')
        
        # 创建模型
        model = model_info['class'](
            nfeat=features.shape[1],
            nclass=labels.max().item() + 1,
            nhid=64,
            device=device,
            **model_info['params']
        ).to(device)
        
        # 训练
        model.fit(features, adj, labels, idx_train, idx_val, 
                  train_iters=1000, patience=400, verbose=False)
        
        # 测试
        acc = model.test(idx_test)
        accuracies.append(acc)
        print(f'Trial {trial+1} Accuracy: {acc*100:.2f}%')
    
    # 汇总结果
    mean_acc = np.mean(accuracies) * 100
    std_acc = np.std(accuracies) * 100
    print(f'\n=== Results ===')
    print(f'Model: {args.model}')
    print(f'Dataset: {args.dataset}')
    print(f'Accuracy: {mean_acc:.1f} ± {std_acc:.1f}%')


if __name__ == '__main__':
    main()
