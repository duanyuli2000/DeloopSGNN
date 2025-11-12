# run_single.py
import torch
import numpy as np
import pandas as pd
import time
import argparse
from pytorch_lightning import seed_everything
from data_utils import DataLoader, random_splits, preprocess, mask_to_index
from base_models import *
import yaml

def run_single(model_name, adj,features,labels, data_split, n_trials=10,device = 'cpu'):


    
    idx_train, idx_val, idx_test = data_split[0], data_split[1], data_split[2]

    config = MODEL_CONFIG[model_name]
    model_class = config['class']
    model_params = config['params']
    
    print(f"\n{'='*50}")
    print(f"Running model: {model_name} on dataset {dname}")
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
    
    line_name = f"{dname}"
    if line_name not in df.index:
        df.loc[line_name] = [None] * (len(df.columns))

    # 更新该模型的结果
    df.loc[line_name, model_name] = new_result
    df.to_csv(result_file)
    print(f"Results updated and saved to {result_file}")
    return df

if __name__ == "__main__":
    datasets = ["cornell", "texas", "reed98", "citeseer", "amherst41", "chameleon", "cora", "johnshopkins55", "squirrel", "photo", "actor", "film", "computers", "cornell5", "pubmed", "penn94"]     

    parser = argparse.ArgumentParser(description='Re-run single model experiment')
    parser.add_argument('--model', type=str, default="DeloopSGNN", help='Model name to re-run (e.g., GPRGNN)')
    parser.add_argument('--dataset', type=str, default="cora", help=f"Supported datasets: {datasets}")
    parser.add_argument('--result_file', type=str, default='./result/exp1_result.csv', help='Path to existing result file')
    parser.add_argument('--n_trials', type=int, default=10, help='Number of trials to run')
    parser.add_argument('--base_seed', type=int, default=3, help='Base random seed')
    parser.add_argument('--device_num', type=int, default=0, help='GPU device number to use (default: 0)')
    
    model = parser.parse_args().model
    dname = parser.parse_args().dataset
    result_file = parser.parse_args().result_file
    n_trials = parser.parse_args().n_trials
    seed = parser.parse_args().base_seed
    device_num = parser.parse_args().device_num
    device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
    
    
    MODEL_CONFIG = {
        'GCN': {'class': GCN, 'params': {}},
        'GAT': {'class': GAT, 'params': {}},
        'H2GCN': {'class': H2GCN, 'params': {}},
        'FAGCN': {'class': FAGCN, 'params': {"epsilon":0.01}},
        'GPRGNN': {'class': GPRGNN, 'params': {'K': 10, "dropout":0.4}}, 
        'ChebNet': {'class': ChebNet, 'params': {}},
        'EvenNet': {'class': EvenNet, 'params': {'K': 10}}, 
        'BernNet': {'class': BernNet, 'params': {'K': 8}},        
        'JacobiConvGNN': {'class': JacobiConvGNN, 'params': {'K': 5,"alpha":0.5, "dropout":0.7}},
        'DeloopSGNN': {'class': DeloopSGNN, 'params': {}},
    }
    
    
    assert model in MODEL_CONFIG, f"Model {model} not found in MODEL_CONFIG"
    assert dname in ['cora', 'citeseer', 'photo', 'computers'], f"Dataset {dname} not supported. Supported datasets: ['cora', 'citeseer', 'photo', 'computers']"
    with open('params.yaml', 'r') as f:
        dataset_config = yaml.safe_load(f)
        dataset_config = dataset_config["exp1"]
        MODEL_CONFIG["DeloopSGNN"]['params'].update(dataset_config.get(dname, {}))
        
    seed_everything(seed)
    device_num = parser.parse_args().device_num
    device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")

    print(f"Dataset: {dname}")

    # preprocess data
    if dname in ['penn94','photo','cora','computers','pubmed','citeseer','johnshopkins55']:
        split_rate = [0.1, 0.1, 0.8] 
    else:
        split_rate = [0.6, 0.2, 0.2]
    dataset, data = DataLoader(dname)
    data_split = random_splits(data, dataset.num_classes, 
                            train_rate=split_rate[0], 
                            val_rate=split_rate[1],
                            Flag=0)
    adj,features, labels = preprocess(data_split)
    adj, features, labels = adj.to(device), features.to(device), labels.to(device)
    
    n = adj.shape[0]
    idx_train = mask_to_index(data_split.train_mask, n)
    idx_val = mask_to_index(data_split.val_mask, n)
    idx_test = mask_to_index(data_split.test_mask, n)
    
    print("-"*50)

    try:
        updated_df = run_single(
            model_name=model,
            adj = adj,
            features=features,
            labels=labels,
            data_split=[idx_train, idx_val, idx_test],
            n_trials=n_trials,
            device=device,
        )
    except:
        print(f"Error running model {model} in dataset {dname}")
    print("\nUpdated Result Table:")
    print(updated_df)