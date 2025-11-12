import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from deeprobust.graph import utils
from deeprobust.graph.defense import GCN
from tqdm import tqdm
import scipy.sparse as sp
import numpy as np

class GCNJaccard(GCN):

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, with_relu=True, with_bias=True, device='cpu'):

        super(GCNJaccard, self).__init__(nfeat, nhid, nclass, dropout, lr, weight_decay, with_relu, with_bias, device=device)
        self.device = device
        self.k = None

    def jaccard_similarity(self, features):
        feature = features.clone()
        feature[feature > 0] = 1
        feature[feature <= 0] = 0
        feat_num = feature.shape[1]
        feature_bar =1 - feature
        f11 = feature@feature.T
        f00 = feature_bar@feature_bar.T
        mask = feat_num-f00
        mx = f11/(feat_num-f00)
        mx[mask==0] = 0
        return mx

    def fit(self, features, adj, labels, idx_train, idx_val=None, alpha=0, train_iters=1000,patience = 400, initialize=True, verbose=True, **kwargs):

        modified_adj = self.truncated_jaccard(features, adj, alpha)
        modified_adj[modified_adj < 0] = 0
        self.alpha = alpha
        
        features, modified_adj, labels =features.to(self.device), modified_adj.to(self.device), labels.to(self.device)
        self.adj_norm = utils.normalize_adj_tensor(modified_adj, sparse=False)
        self.modified_adj = modified_adj
        self.features = features
        self.labels = labels
        super().fit(features, modified_adj, labels, idx_train, idx_val, train_iters=train_iters,patience=patience, initialize=initialize, verbose=verbose)

    def truncated_jaccard(self, features, adj, alpha):

        jm  = self.jaccard_similarity(features)
        result = adj.clone()
        result[jm <= alpha] = 0
        return result

    def test(self, idx_test=None):
        """Evaluate on test set.

        Parameters
        ----------
        idx_test : np.ndarray or torch.Tensor, optional
            Test node indices. If None, use data.test_mask.
        """
        self.eval()
        idx_test = torch.tensor(idx_test, dtype=torch.long).to(self.device)
        output = self.forward(self.features, self.adj_norm)
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        return acc_test.item()