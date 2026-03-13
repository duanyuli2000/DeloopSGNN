"""
DeloopSGNN: Revisiting Spectral GNNs Through the Lens of Spatial Aggregation

This file implements the core model: DeloopSGNN
Published at AAAI 2026

Core Idea:
    - Problem: Spectral GNNs suffer from over-smoothing in deep aggregation
    - Cause: A^k contains computational loops, causing signal backflow and repeated dilution
    - Solution: Use loop-free adjacency matrices

Author: Duanyu Li, Huijun Wu, et al.
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter, Linear
from torch_geometric.nn import MessagePassing
from deeprobust.graph import utils
import torch.optim as optim
from copy import deepcopy


def adj_norm(adj, if_self_loop=True):
    """
    Symmetrically normalize adjacency matrix: D^(-1/2) * A * D^(-1/2)
    
    Parameters:
        adj: Adjacency matrix
        if_self_loop: Whether to add self-loops
    
    Returns:
        Normalized adjacency matrix
    """
    D = adj.sum(dim=1)
    if if_self_loop:
        D = D + 1
    D_inv_sqrt = torch.pow(D, -0.5)
    D_inv_sqrt[D <= 0] = 0
    D_inv_sqrt = torch.diag(D_inv_sqrt)
    return D_inv_sqrt @ adj @ D_inv_sqrt


# ==================== Loop-Free Adjacency Matrix ====================
"""
Core idea of loop-free adjacency matrix:
    Traditional k-hop adjacency matrix A^k contains paths that pass through 
    the same node (loops), which causes:
    1) Signal backflow
    2) Repeated feature dilution
    
    Loop-free version only keeps true k-hop neighbors without passing 
    through intermediate nodes
"""

def lra_2_hop(adj):
    """
    Compute 2-hop loop-free adjacency matrix
    
    Traditional: A^2 = A @ A
    Loop-free: Only keep true 2-hop neighbors, remove self-loops
    
    Args:
        adj: Original adjacency matrix
    
    Returns:
        2-hop loop-free adjacency matrix
    """
    P2 = adj @ adj
    P2.fill_diagonal_(0)
    return P2


def lra_3_hop(adj):
    """
    Compute 3-hop loop-free adjacency matrix
    
    Traditional: A^3 = A^2 @ A
    Loop-free: Remove paths passing through 1-hop neighbors
    
    Mathematical derivation:
        P3 = P2 @ A - A * (degrees - 1)
        where (degrees - 1) subtracts paths through 1-hop neighbors
    """
    degrees = adj.sum(dim=1).unsqueeze(0)
    P2 = lra_2_hop(adj)
    Q3 = P2 @ adj
    P3 = Q3 - adj * (degrees - 1)
    P3.fill_diagonal_(0)
    return P3


def lra_4_hop(adj):
    """
    Compute 4-hop loop-free adjacency matrix
    
    More complex removal logic considering:
    - Paths through 1-hop neighbors
    - Paths through 2-hop neighbors
    """
    P3 = lra_3_hop(adj)
    P2 = lra_2_hop(adj)
    Q3 = P2 @ adj
    Q4 = P3 @ adj
    degrees = adj.sum(dim=1).unsqueeze(0)
    Q3_diag = Q3.diagonal().unsqueeze(0)

    P4_0 = Q4 - (degrees - 1) * P2
    P4_1 = Q4 - degrees * P2 - Q3_diag + 4 * P2
    
    P4 = P4_0 * (1 - adj) + P4_1 * adj
    P4.fill_diagonal_(0)
    return P4


def lra_5_hop(adj):
    """
    Compute 5-hop loop-free adjacency matrix
    
    Even more complex derivation considering all paths from 1-4 hops
    """
    P2 = lra_2_hop(adj)
    P3 = lra_3_hop(adj)
    P4 = lra_4_hop(adj)
    Q3 = P2 @ adj
    Q4 = P3 @ adj
    Q5 = P4 @ adj
    degrees = adj.sum(dim=1).unsqueeze(0)
    Q3_diag = Q3.diagonal().unsqueeze(0)
    Q4_diag = Q4.diagonal().unsqueeze(0)
    
    P5_0 = Q5.clone()
    P5_0 -= (degrees - 1) * P3 - adj @ (adj * P2)
    P5_0 -= adj @ adj * Q3_diag - 2 * adj @ (adj * P2)
    
    P5_1 = Q5.clone()
    P5_1 -= adj @ adj + (degrees - 2) * P3 - adj @ (adj * P2)
    P5_1 -= (adj @ adj) * (Q3_diag - 2 * P2 + 2) - 2 * adj @ (adj * P2)
    P5_1 -= Q4_diag - 2 * P3 - P2 * (P2 - 1)
    
    P5 = P5_0 * (1 - adj) + P5_1 * adj
    P5.fill_diagonal_(0)
    return P5


def lra_k_hop(adj, k):
    """
    Get k-hop loop-free adjacency matrix
    
    Args:
        adj: Adjacency matrix
        k: Number of hops (0-5)
    
    Returns:
        k-hop loop-free adjacency matrix
    
    Note: Returns zero matrix for k > 5 (not implemented yet)
    """
    if k == 0:
        return torch.eye(adj.shape[0], dtype=torch.float32).to(adj.device)
    elif k == 1:
        return adj
    elif k == 2:
        return lra_2_hop(adj)
    elif k == 3:
        return lra_3_hop(adj)
    elif k == 4:
        return lra_4_hop(adj)
    elif k == 5:
        return lra_5_hop(adj)
    else:
        return torch.zeros_like(adj)


def adj_k_hop(adj, k):
    """
    Traditional k-hop adjacency matrix (with loops)
    
    Args:
        adj: Adjacency matrix
        k: Number of hops
    
    Returns:
        A^k
    """
    adj_k = torch.eye(adj.shape[0], dtype=torch.float32).to(adj.device)
    for i in range(k):
        adj_k = adj_k @ adj
    return adj_k


def get_adj_list(adj, K):
    """
    Get adjacency matrix list from 0 to K hops
    
    Args:
        adj: Adjacency matrix
        K: Maximum number of hops
    
    Returns:
        adj_list: List of adjacency matrices [A_0, A_1, ..., A_K]
    """
    Ak_list = []
    adj_n_k = adj_norm(adj + torch.eye(adj.shape[0]).to(adj.device), if_self_loop=False)
    
    for k in range(K + 1):
        if k <= 5:
            adj_k_lf = lra_k_hop(adj, k)
            adj_k_lf = adj_norm(adj_k_lf)
            Ak_list.append(adj_k_lf)
        else:
            adj_k_lf = adj_k_hop(adj_n_k, k)
            Ak_list.append(adj_k_lf)
    
    return Ak_list


class Deloop_prop(MessagePassing):
    """
    Loop-free neighborhood aggregation layer
    
    Core idea:
        Use learnable coefficients theta_k to weight different hop matrices
        
        output = sum(theta_k * (A_k_loop_free @ x))
        
    where theta_k are learnable parameters controlling each hop's contribution
    """

    def __init__(self, K):
        """
        Args:
            K: Maximum number of hops
        """
        super(Deloop_prop, self).__init__(aggr='add')
        self.K = K
        
        # Initialize learnable parameters theta_k
        # Using Xavier initialization for stable gradients
        bound = np.sqrt(3 / (K + 1))
        TEMP = np.random.uniform(-bound, bound, K + 1)
        TEMP = TEMP / np.sum(np.abs(TEMP))
        TEMP[0] = 0  # k=0 not used (node's own features)
        
        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        """Reset parameters"""
        torch.nn.init.zeros_(self.temp)
        bound = np.sqrt(3 / (self.K + 1))
        TEMP = np.random.uniform(-bound, bound, self.K + 1)
        TEMP[0] = 0
        TEMP = TEMP / np.sum(np.abs(TEMP))
        self.temp.data = self.temp

    def forward(self, x, adj_list):
        """
        Forward propagation
        
        Args:
            x: Node features [N, d]
            adj_list: List of adjacency matrices
        
        Returns:
            Aggregated features
        """
        hidden = torch.zeros_like(x, dtype=x.dtype, device=x.device)
        
        for k in range(self.K + 1):
            adj_k = adj_list[k]
            hidden += self.temp[k] * (adj_k @ x)
        
        return hidden

    def message(self, x_j, norm):
        """Message passing function"""
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return f'{self.__class__.__name__}(K={self.K}, temp={self.temp})'


class DeloopSGNN(torch.nn.Module):
    """
    DeloopSGNN Model
    
    Architecture:
        1. Feature Transform: Linear(nfeat -> nhid) -> ReLU -> Dropout
        2. Loop-Free Propagation: Deloop_prop (learnable coefficient multi-hop aggregation)
        3. Output: Linear(nhid -> nclass) -> LogSoftmax
    
    Difference from traditional spectral methods:
        - Traditional: Use A^k for k-hop aggregation (contains loops)
        - Ours: Use loop-free adjacency matrices, eliminate signal backflow
    """

    def __init__(self, nfeat, nhid, nclass, K=10, dropout=0.5, lr=0.01, 
                 weight_decay=5e-4, with_relu=True, device=None):
        """
        Args:
            nfeat: Input feature dimension
            nhid: Hidden layer dimension
            nclass: Number of classes
            K: Maximum number of hops
            dropout: Dropout ratio
            lr: Learning rate
            weight_decay: Weight decay
            with_relu: Whether to use ReLU
            device: Device (cpu/cuda)
        """
        super(DeloopSGNN, self).__init__()
        assert device is not None, "Please specify 'device'!"
        
        self.device = device
        self.nfeat = nfeat
        self.nclass = nclass
        self.nhid = nhid
        
        # Feature transformation layers
        self.lin1 = Linear(nfeat, nhid)
        self.lin2 = Linear(nhid, nclass)
        
        # Loop-free propagation layer
        self.prop1 = Deloop_prop(K)
        self.K = K
        
        self.dropout = dropout
        self.with_relu = with_relu
        self.lr = lr
        self.weight_decay = weight_decay if with_relu else 0
        
        self.output = None
        self.best_model = None
        self.best_output = None
        self.features = None

    def reset_parameters(self):
        """Reset all parameters"""
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop1.reset_parameters()

    def forward(self, x, adj_list):
        """
        Forward propagation
        
        Args:
            x: Node features
            adj_list: Loop-free adjacency matrix list
        
        Returns:
            Log probabilities
        """
        # Feature transformation
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        if self.with_relu:
            x = F.relu(x)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Loop-free propagation
        x = self.prop1(x, adj_list)
        
        return F.log_softmax(x, dim=1)

    def initialize(self):
        """Initialize model parameters"""
        self.reset_parameters()

    def fit(self, features, adj, labels, idx_train, idx_val=None, train_iters=200,
            initialize=True, verbose=False, normalize=True, patience=500, **kwargs):
        """
        Train the model
        
        Args:
            features: Node features
            adj: Adjacency matrix
            labels: Node labels
            idx_train: Training set indices
            idx_val: Validation set indices (optional)
            train_iters: Number of training epochs
            initialize: Whether to initialize parameters
            verbose: Whether to print logs
            patience: Early stopping patience
        """
        if initialize:
            self.initialize()

        # Convert to tensors
        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        # Get adjacency matrix list (loop-free)
        self.adj_list = get_adj_list(adj, self.K)
        self.features = features
        self.labels = labels
        
        # Adjust K value
        if self.K > len(self.adj_list) - 1:
            self.K = len(self.adj_list) - 1
            self.prop1.K = self.K
            self.prop1.reset_parameters()
        
        # Training
        if idx_val is None:
            self._train_without_val(labels, idx_train, train_iters, verbose)
        else:
            if patience < train_iters:
                self._train_with_early_stopping(labels, idx_train, idx_val, train_iters, patience, verbose)
            else:
                self._train_with_val(labels, idx_train, idx_val, train_iters, verbose)

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        """Training without validation set"""
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_list)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            
            if verbose and i % 10 == 0:
                print(f'Epoch {i}, training loss: {loss_train.item():.4f}')

        self.eval()
        self.output = self.forward(self.features, self.adj_list)

    def _train_with_early_stopping(self, labels, idx_train, idx_val, train_iters, patience, verbose):
        """Training with early stopping"""
        if verbose:
            print('=== Training with early stopping ===')
        
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_loss_val = float('inf')
        best_output = None
        weights = None
        current_patience = patience
        
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_list)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            
            self.eval()
            output = self.forward(self.features, self.adj_list)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            
            if loss_val < best_loss_val:
                best_loss_val = loss_val
                best_output = output
                weights = deepcopy(self.state_dict())
                current_patience = patience
            else:
                current_patience -= 1
            
            if i > patience and current_patience <= 0:
                if verbose:
                    print(f'Early stopping at epoch {i}')
                break
        
        if weights is not None:
            self.load_state_dict(weights)
            self.output = best_output
        self.eval()
        self.output = self.forward(self.features, self.adj_list)

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        """Training with validation set (no early stopping)"""
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_list)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            
            if verbose and i % 10 == 0:
                self.eval()
                output = self.forward(self.features, self.adj_list)
                loss_val = F.nll_loss(output[idx_val], labels[idx_val])
                print(f'Epoch {i}, train loss: {loss_train.item():.4f}, val loss: {loss_val.item():.4f}')

        self.eval()
        self.output = self.forward(self.features, self.adj_list)

    def test(self, idx_test, verbose=False):
        """Test the model
        
        Args:
            idx_test: Test set indices
            verbose: Whether to print results
        
        Returns:
            Accuracy
        """
        self.eval()
        output = self.forward(self.features, self.adj_list)
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        
        if verbose:
            print(f'Test accuracy: {acc_test.item():.4f}')
        
        return acc_test.item()

    def predict(self):
        """Prediction"""
        self.eval()
        return self.forward(self.features, self.adj_list)
