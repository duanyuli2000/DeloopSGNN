import torch
import os.path as osp
import torch_geometric.transforms as T
import numpy as np
import scipy.sparse as sp

from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Amazon
from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.datasets import Actor, WebKB, LINKXDataset

def mask_to_index(index, size):
    all_idx = np.arange(size)
    return all_idx[index]


def DataLoader(name):
    name = name.lower()
    if name in ["cora", "citeseer", "pubmed"]:
        root_path = "./"
        path = osp.join(root_path, "data")
        dataset = Planetoid(
            path,
            name,
            transform=T.Compose(
                [
                    T.LargestConnectedComponents(1),
                    T.ToUndirected(),
                ]
            ),
        )
    elif name in ["computers", "photo"]:
        root_path = "./"
        path = osp.join(root_path, "data")
        dataset = Amazon(
            path,
            name,
            T.Compose(
                [
                    T.LargestConnectedComponents(1),
                    T.ToUndirected(),
                ]
            ),
        )
    elif name in ["chameleon", "squirrel"]:
        preProcDs = WikipediaNetwork(
            root="./data/",
            name=name,
            geom_gcn_preprocess=False,
            transform=T.Compose(
                [
                    T.LargestConnectedComponents(1),
                    T.ToUndirected(),
                ]
            ),
        )
        dataset = WikipediaNetwork(
            root="./data/",
            name=name,
            geom_gcn_preprocess=True,
            transform=T.Compose(
                [
                    T.LargestConnectedComponents(1),
                    T.ToUndirected(),
                ]
            ),
        )
        data = dataset[0]
        data.edge_index = preProcDs[0].edge_index
        return dataset, data

    elif name in ["film"]:
        dataset = Actor(
            root="./data/film",
            transform=T.Compose(
                [
                    T.LargestConnectedComponents(1),
                    T.ToUndirected(),
                ]
            ),
        )
    elif name in ["actor"]:
        dataset = Actor(
            root="./data/actor",
            transform=T.Compose(
                [
                    T.NormalizeFeatures(),
                    T.LargestConnectedComponents(1),
                    T.ToUndirected(),
                ]
            ),
        )
    elif name in ["texas", "cornell"]:
        dataset = WebKB(
            root="./data/",
            name=name,
            transform=T.Compose(
                [
                    T.LargestConnectedComponents(1),
                    T.ToUndirected(),
                ]
            ),
        )
    elif name in ["penn94", "reed98", "amherst41", "cornell5", "johnshopkins55", "genius"]:
        dataset = LINKXDataset(
            root="./data/",
            name=name,
            transform=T.Compose(
                [
                    T.LargestConnectedComponents(1),
                    T.ToUndirected(),
                ]
            ),
        )
    else:
        raise ValueError(f"dataset {name} not supported in dataloader")
    return dataset, dataset[0]


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def preprocess(data):

    labels = torch.LongTensor(data.y)
    n = data.x.shape[0]
    adj = sp.csr_matrix(
        (np.ones(data.edge_index.shape[1]), (data.edge_index[0], data.edge_index[1])),
        shape=(n, n),
    )
    features = data.x
    if sp.issparse(features):
        features = torch.FloatTensor(np.array(features.todense()))
    else:
        features = torch.FloatTensor(features)
    adj = torch.FloatTensor(adj.todense())
    return adj, features, labels
def random_splits(data, num_classes, train_rate=0.1, val_rate=0.1, Flag=0):
    percls_trn = int(round(train_rate * len(data.y) / num_classes))
    val_lb = int(round(val_rate * len(data.y)))
    indices = []

    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    if Flag == 0:
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(rest_index[:val_lb], size=data.num_nodes)
        data.test_mask = index_to_mask(rest_index[val_lb:], size=data.num_nodes)
    else:
        val_index = torch.cat(
            [i[percls_trn : percls_trn + val_lb] for i in indices], dim=0
        )
        rest_index = torch.cat([i[percls_trn + val_lb :] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(rest_index, size=data.num_nodes)
    return data

