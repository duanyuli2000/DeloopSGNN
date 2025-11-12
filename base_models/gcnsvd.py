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

class GCNSVD(GCN):
    """GCNSVD is a 2 Layer Graph Convolutional Network with Truncated SVD as
    preprocessing. See more details in All You Need Is Low (Rank): Defending
    Against Adversarial Attacks on Graphs,
    https://dl.acm.org/doi/abs/10.1145/3336191.3371789.

    Parameters
    ----------
    nfeat : int
        size of input feature dimension
    nhid : int
        number of hidden units
    nclass : int
        size of output dimension
    dropout : float
        dropout rate for GCN
    lr : float
        learning rate for GCN
    weight_decay : float
        weight decay coefficient (l2 normalization) for GCN. When `with_relu` is True, `weight_decay` will be set to 0.
    with_relu : bool
        whether to use relu activation function. If False, GCN will be linearized.
    with_bias: bool
        whether to include bias term in GCN weights.
    device: str
        'cpu' or 'cuda'.

    Examples
    --------
	We can first load dataset and then train GCNSVD.

    >>> from deeprobust.graph.data import PrePtbDataset, Dataset
    >>> from deeprobust.graph.defense import GCNSVD
    >>> # load clean graph data
    >>> data = Dataset(root='/tmp/', name='cora', seed=15)
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> # load perturbed graph data
    >>> perturbed_data = PrePtbDataset(root='/tmp/', name='cora')
    >>> perturbed_adj = perturbed_data.adj
    >>> # train defense model
    >>> model = GCNSVD(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device='cpu').to('cpu')
    >>> model.fit(features, perturbed_adj, labels, idx_train, idx_val, k=20)

    """

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, with_relu=True, with_bias=True, device='cpu'):

        super(GCNSVD, self).__init__(nfeat, nhid, nclass, dropout, lr, weight_decay, with_relu, with_bias, device=device)
        self.device = device
        self.k = None

    def fit(self, features, adj, labels, idx_train, idx_val=None, k=50, train_iters=1000,patience = 400, initialize=True, verbose=True, **kwargs):
        """First perform rank-k approximation of adjacency matrix via
        truncated SVD, and then train the gcn model on the processed graph,
        when idx_val is not None, pick the best model according to
        the validation loss.

        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GCN training process will not adpot early stopping
        k : int
            number of singular values and vectors to compute.
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        """

        modified_adj = self.truncatedSVD(adj, k=k)
        modified_adj[modified_adj < 0] = 0
        
        self.adj_norm = utils.normalize_adj_tensor(modified_adj, sparse=False)
        self.k = k
        # modified_adj_tensor = utils.sparse_mx_to_torch_sparse_tensor(self.modified_adj)
        # features, modified_adj, labels = utils.to_tensor(features, modified_adj, labels, device=self.device)
        features, modified_adj, labels =features.to(self.device), modified_adj.to(self.device), labels.to(self.device)
        self.modified_adj = modified_adj
        self.features = features
        self.labels = labels
        super().fit(features, modified_adj, labels, idx_train, idx_val, train_iters=train_iters,patience=patience, initialize=initialize, verbose=verbose)

    def truncatedSVD(self, data, k=50):
        """Truncated SVD on input data.

        Parameters
        ----------
        data :
            input matrix to be decomposed
        k : int
            number of singular values and vectors to compute.

        Returns
        -------
        numpy.array
            reconstructed matrix.
        """
        print('=== GCN-SVD: rank={} ==='.format(k))
        U, S, V = torch.linalg.svd(data)
        U = U[:, :k]
        S = S[:k]
        V = V[:k, :]        
        return U @ torch.diag(S) @ V

    def predict(self, features=None, adj=None):
        """By default, the inputs should be unnormalized adjacency

        Parameters
        ----------
        features :
            node features. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        adj :
            adjcency matrix. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.


        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of GCNSVD
        """

        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            adj = self.truncatedSVD(adj, k=self.k)
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(self.features, self.adj_norm)


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
        # print(f"Test Results: Loss = {loss_test.item():.4f}, Accuracy = {acc_test.item():.4f}")
        return acc_test.item()