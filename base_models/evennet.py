import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, dense_to_sparse, get_laplacian
from deeprobust.graph import utils
from scipy.special import comb
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
import copy
from torch.nn import Linear
import numpy as np



class Even_prop(MessagePassing):
    def __init__(self, K, alpha=0.5, bias=True, **kwargs):
        super(Even_prop, self).__init__(aggr="add", **kwargs)
        self.K = int(K // 2)
        self.alpha = alpha
        TEMP = alpha * (1 - alpha) ** (2 * np.arange(K // 2 + 1))
        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K + 1):
            self.temp.data[k] = self.alpha * (1 - self.alpha) ** (2 * k)

    def forward(self, x, edge_index, edge_weight=None):
        # 计算对称归一化拉普拉斯矩阵的边索引和权重
        edge_index1, norm1 = get_laplacian(
            edge_index,
            edge_weight,
            normalization="sym",
            dtype=x.dtype,
            num_nodes=x.size(self.node_dim),
        )
        # 构造 I-L 的边索引（添加自环并调整权重）
        edge_index2, norm2 = add_self_loops(
            edge_index1, -norm1, fill_value=1.0, num_nodes=x.size(self.node_dim)
        )

        hidden = x * self.temp[0]  # 初始隐藏状态
        for k in range(self.K):
            # 两次传播（可能对应模型设计中的特定操作）
            x = self.propagate(edge_index2, x=x, norm=norm2)
            x = self.propagate(edge_index2, x=x, norm=norm2)
            gamma = self.temp[k + 1]  # 权重系数
            hidden = hidden + gamma * x  # 累加不同阶的传播结果
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j  # 消息传递：权重归一化后乘以邻居特征

    def __repr__(self):
        return "{}(K={}, temp={})".format(self.__class__.__name__, self.K, self.temp)


class EvenNet(nn.Module):
    def __init__(
        self, 
        nfeat, 
        nclass, 
        nhid, 
        K,
        dprate=0.0, 
        dropout=0.5, 
        lr=0.01, 
        weight_decay=5e-4,
        device=None
    ):
        super(EvenNet, self).__init__()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 线性层
        self.lin1 = Linear(nfeat, nhid)
        self.lin2 = Linear(nhid, nclass)
        # 自定义传播层
        self.prop1 = Even_prop(K)
        # 超参数
        self.dprate = dprate
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay

    def reset_parameters(self):
        """重置模型参数"""
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop1.reset_parameters()

    def forward(self, x, edge_index):
        """前向传播
        
        Parameters
        ----------
        x : torch.FloatTensor
            节点特征矩阵 [N, nfeat]
        edge_index : torch.LongTensor
            边索引 [2, E]
            
        Returns
        -------
        torch.FloatTensor
            节点类别对数概率 [N, nclass]
        """
        # 特征 dropout
        x = F.dropout(x, p=self.dropout, training=self.training)
        # 第一层线性变换 + ReLU
        x = self.lin1(x)
        x = F.relu(x)
        # 第二层线性变换 + 特征 dropout
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        
        # 应用传播层（根据dprate决定是否加dropout）
        if self.dprate > 0.0:
            x = F.dropout(x, p=self.dprate, training=self.training)
        # 传播并返回对数 softmax
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)

    def fit(
        self, 
        features, 
        adj, 
        labels, 
        idx_train, 
        idx_val=None, 
        idx_test=None, 
        train_iters=1000, 
        initialize=True, 
        verbose=False, 
        normalize=True, 
        patience=100
    ):
        """训练模型
        
        Parameters
        ----------
        features : np.ndarray/torch.Tensor
            节点特征矩阵 [N, nfeat]
        adj : np.ndarray/torch.Tensor/scipy.sparse
            邻接矩阵 [N, N]
        labels : np.ndarray/torch.Tensor
            节点标签 [N]
        idx_train : np.ndarray/torch.Tensor
            训练节点索引
        idx_val : np.ndarray/torch.Tensor, optional
            验证节点索引
        idx_test : np.ndarray/torch.Tensor, optional
            测试节点索引
        train_iters : int, optional
            训练轮次
        initialize : bool, optional
            是否初始化参数
        verbose : bool, optional
            是否打印训练日志
        normalize : bool, optional
            是否归一化邻接矩阵
        patience : int, optional
            早停耐心值
        """
        # 数据转Tensor并移动到设备
        device = self.device
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        if not isinstance(adj, torch.Tensor):
            adj = torch.tensor(adj, dtype=torch.float32)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)
        
        features, adj, labels = features.to(device), adj.to(device), labels.to(device)
        
        # 构建训练掩码
        self.data = Data(x=features, edge_index=None, y=labels)
        self.data.train_mask = torch.zeros(features.size(0), dtype=torch.bool, device=device)
        self.data.train_mask[idx_train] = True
        
        # 构建验证/测试掩码
        if idx_val is not None:
            idx_val = torch.tensor(idx_val, dtype=torch.long).to(device)
            self.data.val_mask = torch.zeros_like(self.data.train_mask)
            self.data.val_mask[idx_val] = True
        else:
            self.data.val_mask = None
            
        if idx_test is not None:
            idx_test = torch.tensor(idx_test, dtype=torch.long).to(device)
            self.data.test_mask = torch.zeros_like(self.data.train_mask)
            self.data.test_mask[idx_test] = True
        else:
            self.data.test_mask = None
        
        # 邻接矩阵预处理（归一化+转边索引）
        if normalize:
            adj = utils.normalize_adj_tensor(adj)
        edge_index, _ = dense_to_sparse(adj)  # 稀疏矩阵转边索引
        edge_index, _ = add_self_loops(edge_index, num_nodes=features.size(0))  # 添加自环
        
        self.data.edge_index = edge_index  # 存储边索引
        
        # 参数初始化
        if initialize:
            self.reset_parameters()
        
        # 优化器与早停配置
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_val_loss = float('inf')
        best_weights = None
        current_patience = patience
        
        # 早停训练循环
        for epoch in range(train_iters):
            self.train()  # 训练模式
            optimizer.zero_grad()
            output = self.forward(self.data.x, self.data.edge_index)  # 前向传播
            # 计算训练损失（仅用训练集）

            loss_train = F.nll_loss(output[self.data.train_mask], self.data.y[self.data.train_mask])
            loss_train.backward()
            optimizer.step()
            
            # 打印训练日志
            if verbose and epoch % 10 == 0:
                print(f'Epoch {epoch:03d}, Train Loss: {loss_train.item():.4f}')
            
            # 验证阶段
            self.eval()  # 评估模式
            with torch.no_grad():
                output = self.forward(self.data.x, self.data.edge_index)
                val_loss = 0.0
                if self.data.val_mask is not None:
                    val_loss = F.nll_loss(output[self.data.val_mask], self.data.y[self.data.val_mask])
            
            # 早停逻辑
            if self.data.val_mask is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = copy.deepcopy(self.state_dict())
                    current_patience = patience
                else:
                    current_patience -= 1
                    if current_patience <= 0:
                        if verbose:
                            print(f'Early stopping at epoch {epoch}, Best Val Loss: {best_val_loss:.4f}')
                        break
        
        # 加载最佳模型
        if best_weights is not None:
            self.load_state_dict(best_weights)
            self.output = self.forward(self.data.x, self.data.edge_index)
        else:
            self.output = output

    def test(self, idx_test=None):
        """评估测试集性能
        
        Parameters
        ----------
        idx_test : np.ndarray/torch.Tensor, optional
            测试节点索引
            
        Returns
        -------
        float
            测试准确率
        """
        self.eval()
        # 确定测试索引
        if idx_test is None:
            if self.data.test_mask is not None:
                idx_test = self.data.test_mask.nonzero(as_tuple=True)[0]
            else:
                raise ValueError("Test indices not provided and data has no test_mask.")
        idx_test = torch.tensor(idx_test, dtype=torch.long).to(self.data.x.device)
        
        # 前向传播并计算指标
        with torch.no_grad():
            output = self.forward(self.data.x, self.data.edge_index)
            loss_test = F.nll_loss(output[idx_test], self.data.y[idx_test])
            acc_test = utils.accuracy(output[idx_test], self.data.y[idx_test])
        return acc_test.item()

    def predict(self, x=None, edge_index=None):
        """预测节点类别概率
        
        Parameters
        ----------
        x : torch.FloatTensor, optional
            节点特征矩阵（若为None则使用训练数据）
        edge_index : torch.LongTensor, optional
            边索引（若为None则使用训练数据）
            
        Returns
        -------
        torch.FloatTensor
            节点类别对数概率 [N, nclass]
        """
        self.eval()
        # 使用存储数据或外部输入
        if x is None or edge_index is None:
            if hasattr(self, 'data'):
                x, edge_index = self.data.x, self.data.edge_index
            else:
                raise ValueError("x and edge_index must be provided if no data is stored.")
        return self.forward(x, edge_index)