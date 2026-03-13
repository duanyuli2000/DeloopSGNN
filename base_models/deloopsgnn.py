"""
DeloopSGNN: Revisiting Spectral GNNs Through the Lens of Spatial Aggregation

本文件实现了论文中的核心模型：DeloopSGNN
论文: AAAI 2026

核心思想：
    - 谱域 GNN 的问题：深度聚合时会出现 over-smoothing
    - 原因：Ã^k 包含计算环路，导致信号回流和重复稀释
    - 解决：使用无环邻接矩阵 (loop-free adjacency matrix)

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
    邻接矩阵对称归一化: D^(-1/2) * A * D^(-1/2)
    
    Parameters:
        adj: 邻接矩阵
        if_self_loop: 是否添加自环
    
    Returns:
        归一化后的邻接矩阵
    """
    D = adj.sum(dim=1)
    if if_self_loop:
        D = D + 1  # 添加自环后度+1
    D_inv_sqrt = torch.pow(D, -0.5)
    D_inv_sqrt[D <= 0] = 0  # 处理度为0的情况
    D_inv_sqrt = torch.diag(D_inv_sqrt)
    return D_inv_sqrt @ adj @ D_inv_sqrt


# ==================== 无环邻接矩阵 (Loop-Free Adjacency Matrix) ====================
"""
无环邻接矩阵的核心思想：
    传统的 k 跳邻接矩阵 Ã^k 包含了通过同一节点的路径（环路）
    这会导致：1) 信号回流 2) 特征重复稀释
    
    无环版本只保留真正的 k 跳邻居，不经过中间节点
"""

def lra_2_hop(adj):
    """
    计算 2 跳无环路邻接矩阵
    
    传统方法: Ã² = Ã @ Ã
    无环方法: 只保留真正的 2 跳邻居，移除自环
    
    Args:
        adj: 原始邻接矩阵
    
    Returns:
        2 跳无环路邻接矩阵
    """
    P2 = adj @ adj
    P2.fill_diagonal_(0)  # 移除自环
    return P2


def lra_3_hop(adj):
    """
    计算 3 跳无环路邻接矩阵
    
    传统方法: Ã³ = Ã² @ Ã
    无环方法: 移除经过 1 跳路径的连接
    
    数学推导:
        P3 = P2 @ Ã - Ã * (degrees - 1)
        其中 (degrees - 1) 是减去通过 1 跳路径的连接数
    """
    degrees = adj.sum(dim=1).unsqueeze(0)
    P2 = lra_2_hop(adj)
    Q3 = P2 @ adj
    P3 = Q3 - adj * (degrees - 1)  # 移除 1 跳路径
    P3.fill_diagonal_(0)
    return P3


def lra_4_hop(adj):
    """
    计算 4 跳无环路邻接矩阵
    
    更加复杂的移除逻辑，需要考虑：
    - 经过 1 跳路径的连接
    - 经过 2 跳路径的连接
    """
    P3 = lra_3_hop(adj)
    P2 = lra_2_hop(adj)
    Q3 = P2 @ adj
    Q4 = P3 @ adj
    degrees = adj.sum(dim=1).unsqueeze(0)
    Q3_diag = Q3.diagonal().unsqueeze(0)

    # 两种情况的组合
    P4_0 = Q4 - (degrees - 1) * P2  # 非邻居情况
    P4_1 = Q4 - degrees * P2 - Q3_diag + 4 * P2  # 邻居情况
    
    # 根据原始邻接矩阵选择
    P4 = P4_0 * (1 - adj) + P4_1 * adj
    P4.fill_diagonal_(0)
    return P4


def lra_5_hop(adj):
    """
    计算 5 跳无环路邻接矩阵
    
    更复杂的推导，需要考虑 1-4 跳的所有路径
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
    获取 k 跳无环路邻接矩阵
    
    Args:
        adj: 邻接矩阵
        k: 跳数 (0-5)
    
    Returns:
        k 跳无环路邻接矩阵
    
    Note: k > 5 时返回零矩阵（尚未实现）
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
        # k > 5 时返回零矩阵（简化处理）
        return torch.zeros_like(adj)


def adj_k_hop(adj, k):
    """
    传统的 k 跳邻接矩阵（带环路）
    
    Args:
        adj: 邻接矩阵
        k: 跳数
    
    Returns:
        Ã^k
    """
    adj_k = torch.eye(adj.shape[0], dtype=torch.float32).to(adj.device)
    for i in range(k):
        adj_k = adj_k @ adj
    return adj_k


def get_adj_list(adj, K):
    """
    获取 0 到 K 跳的邻接矩阵列表
    
    Args:
        adj: 邻接矩阵
        K: 最大跳数
    
    Returns:
        adj_list: 邻接矩阵列表 [Ã₀, Ã₁, ..., Ã_K]
    """
    Ak_list = []
    # 归一化带自环的邻接矩阵
    adj_n_k = adj_norm(adj + torch.eye(adj.shape[0]).to(adj.device), if_self_loop=False)
    
    for k in range(K + 1):
        if k <= 5:
            # 使用无环路邻接矩阵
            adj_k_lf = lra_k_hop(adj, k)
            adj_k_lf = adj_norm(adj_k_lf)
            Ak_list.append(adj_k_lf)
        else:
            # k > 5 使用传统邻接矩阵
            adj_k_lf = adj_k_hop(adj_n_k, k)
            Ak_list.append(adj_k_lf)
    
    return Ak_list


class Deloop_prop(MessagePassing):
    """
    无环路邻域聚合层
    
    核心思想：
        使用可学习的系数 θ_k 对不同跳数的无环邻接矩阵进行加权组合
        
        output = Σ θ_k * (Ã_k_loop_free @ x)
        
    其中 θ_k 是可学习的参数，控制 k 跳邻居信息的贡献
    """

    def __init__(self, K):
        """
        Args:
            K: 最大跳数
        """
        super(Deloop_prop, self).__init__(aggr='add')
        self.K = K
        
        # 初始化可学习参数 θ_k
        # 使用 Xavier 初始化，保证梯度稳定
        bound = np.sqrt(3 / (K + 1))
        TEMP = np.random.uniform(-bound, bound, K + 1)
        TEMP = TEMP / np.sum(np.abs(TEMP))  # 归一化
        TEMP[0] = 0  # k=0 不使用（已经是节点自身特征）
        
        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        """重置参数"""
        torch.nn.init.zeros_(self.temp)
        bound = np.sqrt(3 / (self.K + 1))
        TEMP = np.random.uniform(-bound, bound, self.K + 1)
        TEMP[0] = 0
        TEMP = TEMP / np.sum(np.abs(TEMP))
        self.temp.data = self.temp

    def forward(self, x, adj_list):
        """
        前向传播
        
        Args:
            x: 节点特征 [N, d]
            adj_list: 邻接矩阵列表
        
        Returns:
            聚合后的特征
        """
        hidden = torch.zeros_like(x, dtype=x.dtype, device=x.device)
        
        # 对每个跳数进行加权聚合
        for k in range(self.K + 1):
            adj_k = adj_list[k]
            hidden += self.temp[k] * (adj_k @ x)
        
        return hidden

    def message(self, x_j, norm):
        """消息传递函数"""
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return f'{self.__class__.__name__}(K={self.K}, temp={self.temp})'


class DeloopSGNN(torch.nn.Module):
    """
    DeloopSGNN 模型
    
    整体架构：
        1. 特征变换: Linear(nfeat -> nhid) -> ReLU -> Dropout
        2. 无环路传播: Deloop_prop (可学习系数的多跳聚合)
        3. 输出层: Linear(nhid -> nclass) -> LogSoftmax
    
    与传统谱方法的区别：
        - 传统: 使用 Ã^k 进行 k 跳聚合（包含环路）
        - Ours: 使用无环邻接矩阵，消除信号回流
    """

    def __init__(self, nfeat, nhid, nclass, K=10, dropout=0.5, lr=0.01, 
                 weight_decay=5e-4, with_relu=True, device=None):
        """
        Args:
            nfeat: 输入特征维度
            nhid: 隐藏层维度
            nclass: 类别数
            K: 最大跳数
            dropout: Dropout 比率
            lr: 学习率
            weight_decay: 权重衰减
            with_relu: 是否使用 ReLU
            device: 设备 (cpu/cuda)
        """
        super(DeloopSGNN, self).__init__()
        assert device is not None, "Please specify 'device'!"
        
        self.device = device
        self.nfeat = nfeat
        self.nclass = nclass
        self.nhid = nhid
        
        # 特征变换层
        self.lin1 = Linear(nfeat, nhid)
        self.lin2 = Linear(nhid, nclass)
        
        # 无环路传播层
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
        """重置所有参数"""
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop1.reset_parameters()

    def forward(self, x, adj_list):
        """
        前向传播
        
        Args:
            x: 节点特征
            adj_list: 无环邻接矩阵列表
        
        Returns:
            Log 概率
        """
        # 特征变换
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        if self.with_relu:
            x = F.relu(x)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 无环路传播
        x = self.prop1(x, adj_list)
        
        return F.log_softmax(x, dim=1)

    def initialize(self):
        """初始化模型参数"""
        self.reset_parameters()

    def fit(self, features, adj, labels, idx_train, idx_val=None, train_iters=200,
            initialize=True, verbose=False, normalize=True, patience=500, **kwargs):
        """
        训练模型
        
        Args:
            features: 节点特征
            adj: 邻接矩阵
            labels: 节点标签
            idx_train: 训练集索引
            idx_val: 验证集索引 (可选)
            train_iters: 训练轮数
            initialize: 是否初始化参数
            verbose: 是否打印日志
            patience: 早停耐心值
        """
        if initialize:
            self.initialize()

        # 转换为张量
        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        # 获取邻接矩阵列表（无环）
        self.adj_list = get_adj_list(adj, self.K)
        self.features = features
        self.labels = labels
        
        # 调整 K 值
        if self.K > len(self.adj_list) - 1:
            self.K = len(self.adj_list) - 1
            self.prop1.K = self.K
            self.prop1.reset_parameters()
        
        # 训练
        if idx_val is None:
            self._train_without_val(labels, idx_train, train_iters, verbose)
        else:
            if patience < train_iters:
                self._train_with_early_stopping(labels, idx_train, idx_val, train_iters, patience, verbose)
            else:
                self._train_with_val(labels, idx_train, idx_val, train_iters, verbose)

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        """无验证集的训练"""
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
        """带早停的训练"""
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
        """带验证集的训练（无早停）"""
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
        """测试模型
        
        Args:
            idx_test: 测试集索引
            verbose: 是否打印结果
        
        Returns:
            准确率
        """
        self.eval()
        output = self.forward(self.features, self.adj_list)
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        
        if verbose:
            print(f'Test accuracy: {acc_test.item():.4f}')
        
        return acc_test.item()

    def predict(self):
        """预测"""
        self.eval()
        return self.forward(self.features, self.adj_list)
