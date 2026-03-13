"""
Experiment 1: Standard Task Accuracy Evaluation

This script evaluates model accuracy on standard node classification tasks.

Supported datasets:
    - Homophilic: Cora, Citeseer, Pubmed, Reddit, ogbn-arxiv
    - Heterophilic: Chameleon, Squirrel, Film, Texas, Wisconsin, Cornell

Usage:
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


# Model configuration
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
}


def parse_args():
    """Parse command line arguments"""
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
    """Main function"""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Load data
    dataset, data = DataLoader(args.dataset)
    features = data.x
    labels = data.y
    edge_index = data.edge_index
    
    # Build adjacency matrix
    n = features.shape[0]
    adj = np.zeros((n, n))
    adj[edge_index[0].numpy(), edge_index[1].numpy()] = 1
    
    # Data splits
    data_split = random_splits(labels, 2000)
    idx_train, idx_val, idx_test = data_split[0], data_split[1], data_split[2]
    
    # Get model
    model_info = MODEL_CONFIG.get(args.model)
    if model_info is None:
        raise ValueError(f'Unknown model: {args.model}')
    
    # Run experiments
    accuracies = []
    for trial in range(args.n_trials):
        print(f'\n=== Trial {trial+1}/{args.n_trials} ===')
        
        # Create model
        model = model_info['class'](
            nfeat=features.shape[1],
            nclass=labels.max().item() + 1,
            nhid=64,
            device=device,
            **model_info['params']
        ).to(device)
        
        # Train
        model.fit(features, adj, labels, idx_train, idx_val, 
                  train_iters=1000, patience=400, verbose=False)
        
        # Test
        acc = model.test(idx_test)
        accuracies.append(acc)
        print(f'Trial {trial+1} Accuracy: {acc*100:.2f}%')
    
    # Summarize results
    mean_acc = np.mean(accuracies) * 100
    std_acc = np.std(accuracies) * 100
    print(f'\n=== Results ===')
    print(f'Model: {args.model}')
    print(f'Dataset: {args.dataset}')
    print(f'Accuracy: {mean_acc:.1f} +/- {std_acc:.1f}%')


if __name__ == '__main__':
    main()
