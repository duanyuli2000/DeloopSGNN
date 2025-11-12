# run_single.py
import torch
import numpy as np
import pandas as pd
import time
import argparse
from pytorch_lightning import seed_everything
from data_utils import DataLoader, random_splits, preprocess
from base_models import *
import yaml

def run_single(model_name, adj,features,labels, data_split, n_trials=10,device = 'cpu'):


    
    idx_train, idx_val, idx_test = data_split[0], data_split[1], data_split[2]

    config = MODEL_CONFIG[model_name]
    model_class = config['class']
    model_params = config['params']
    
    print(f"\n{'='*50}")
    print(f"Running model: {model_name} on dataset {dname} with attack {attack_name} and rate {ptb_rate}")
    print(f"Using parameters: {model_params}")
    print(f"{'='*50}")
    accuracies = []
    start_time = time.time()

    for trial in range(n_trials):
        # 初始化模型
        model = model_class(
            nfeat=features.shape[1],
            nclass=labels.max().item() + 1,
            nhid=64,
            lr=0.01,
            weight_decay=5e-4,
            device=device,
            **model_params
        ).to(device)
        
        model.fit(features, adj, labels, idx_train, idx_val,patience = 400, train_iters=1000, verbose=False)
        acc = model.test(idx_test)
        accuracies.append(acc)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  Trial {trial+1}/{n_trials} - Accuracy: {acc*100:.2f}%")
    
    mean_acc = np.mean(accuracies) * 100
    std_acc = np.std(accuracies) * 100
    new_result = f"{mean_acc:.1f} ± {std_acc:.1f}"
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"New result: {new_result}")
    
    try:
        df = pd.read_csv(result_file, index_col=0)
    except FileNotFoundError:
        print(f"File {result_file} not found, creating new result table")
        df = pd.DataFrame(columns=list(MODEL_CONFIG.keys()))
    
    line_name = f"{dname}_{attack_name}_{ptb_rate:.2f}"
    if line_name not in df.index:
        df.loc[line_name] = [None] * (len(df.columns))

    # 更新该模型的结果
    df.loc[line_name, model_name] = new_result
    df.to_csv(result_file)
    print(f"Results updated and saved to {result_file}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Re-run single model experiment')
    parser.add_argument('--model', type=str, default="DeloopSGNN", help='Model name to re-run (e.g., GPRGNN)')
    parser.add_argument('--dataset', type=str, default="cora", help="Supported datasets: ['cora', 'citeseer', 'photo', 'computers']")
    parser.add_argument('--attack', type=str, default="Metattack", help="Supported attacks: ['Metattack', 'PGDAttack']")
    parser.add_argument('--ptb_rate', type=float, default=0.0, help="Supported Ptb Rate: [0.0, 0.05, 0.1, 0.15, 0.20, 0.25]")
    parser.add_argument('--result_file', type=str, default='./result/exp2_result.csv', help='Path to existing result file')
    parser.add_argument('--n_trials', type=int, default=10, help='Number of trials to run')
    parser.add_argument('--base_seed', type=int, default=3, help='Base random seed')
    parser.add_argument('--device_num', type=int, default=0, help='GPU device number to use (default: 0)')
    
    model = parser.parse_args().model
    dname = parser.parse_args().dataset
    result_file = parser.parse_args().result_file
    n_trials = parser.parse_args().n_trials
    attack_name = parser.parse_args().attack
    ptb_rate = parser.parse_args().ptb_rate
    seed = parser.parse_args().base_seed
    device_num = parser.parse_args().device_num
    device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
    
    
    MODEL_CONFIG = {
        'GCN': {'class': GCN, 'params': {}},
        'GAT': {'class': GAT, 'params': {}},
        'GNNGuard': {'class': GNNGuard, 'params': {}},
        'GCNSVD': {'class': GCNSVD, 'params': {}},
        'GCNJaccard': {'class': GCNJaccard, 'params': {}},
        'MidGCN': {'class': MidGCN, 'params': {}},
        'NoisyGCN': {'class': NoisyGCN, 'params': {}},
        'DeloopSGNN': {'class': DeloopSGNN, 'params': {}},
    }
    assert model in MODEL_CONFIG, f"Model {model} not found in MODEL_CONFIG"
    assert dname in ['cora', 'citeseer', 'photo', 'computers'], f"Dataset {dname} not supported. Supported datasets: ['cora', 'citeseer', 'photo', 'computers']"
    attack_name = "Meta-Self" if attack_name == "Metattack" else "PGD-Self"
    with open('params.yaml', 'r') as f:
        dataset_config = yaml.safe_load(f)
        dataset_config = dataset_config["exp2"][dname][attack_name]
        MODEL_CONFIG["DeloopSGNN"]['params'].update(dataset_config.get(ptb_rate, {}))
        
    seed_everything(seed)
    device_num = parser.parse_args().device_num
    device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")

    print(f"Dataset: {dname}")

    dataset, data = DataLoader(dname)
    data = random_splits(data, dataset.num_classes, Flag=0)
    adj,features,labels = preprocess(data)
    features,labels = features.to(device), labels.to(device)

    print("-"*50)
    print(f"Attack Method: {attack_name}, Perturbation Rate: {ptb_rate}")

    adj_path = f"./attack_data/seed:{seed}_adj_{dname}_{attack_name}_{ptb_rate:.2f}.pt"
    try:
        attacked_adj, idx_train, idx_val, idx_test = torch.load(
            adj_path, map_location="cpu"
        )
        attacked_adj = attacked_adj.to(device)
        attacked_adj[attacked_adj < 0] = 1
        attacked_adj.fill_diagonal_(0)
    except:
        print(f"Error loading attacked adjacency matrix for dataset {dname} with attack {attack_name} and rate {ptb_rate}")
    try:
        updated_df = run_single(
            model_name=model,
            adj = attacked_adj,
            features=features,
            labels=labels,
            data_split=[idx_train, idx_val, idx_test],
            n_trials=n_trials,
            device=device,
        )
    except:
        print(f"Error running model {model} in dataset {dname} with attack {attack_name} and rate {ptb_rate}")
    print("\nUpdated Result Table:")
    print(updated_df)