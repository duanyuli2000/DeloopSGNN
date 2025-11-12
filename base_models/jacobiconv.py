import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear
from torch_sparse import SparseTensor
from deeprobust.graph import utils
import numpy as np
import torch.optim as optim
from copy import deepcopy
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, dense_to_sparse, get_laplacian

class JacobiConv_prop(nn.Module):
    '''
    A framework for polynomial graph signal filter.
    Args:
        conv_fn: the filter function, like PowerConv, LegendreConv,...
        depth (int): the order of polynomial.
        cached (bool): whether or not to cache the adjacency matrix. 
        alpha (float):  the parameter to initialize polynomial coefficients.
        fixed (bool): whether or not to fix to polynomial coefficients.
    '''
    def __init__(self,
                 depth: int = 3,
                 cached: bool = True,
                 alpha: float = 0.5,
                 fixed: float = False,
                 nclass: int = 1
                 ):
        super().__init__()
        self.depth = depth
        self.basealpha = alpha
        self.alphas = nn.ParameterList([
            nn.Parameter(torch.tensor(float(min(1 / alpha, 1))),
                         requires_grad=not fixed) for i in range(depth + 1)
        ])
        self.cached = cached
        self.adj = None
        self.fixed = fixed
        self.comb_weight = nn.Parameter(torch.ones((1, self.depth+1, nclass)))


    def jacobiconv(self, L, xs, adj, alphas, a=1.75, b=-0.5, l=-1.0, r=1.0):
        '''
        Jacobi Bases. Please refer to our paper for the form of the bases.
        '''
        if L == 0: return xs[0]
        if L == 1:
            coef1 = (a - b) / 2 - (a + b + 2) / 2 * (l + r) / (r - l)
            coef1 *= alphas[0]
            coef2 = (a + b + 2) / (r - l)
            coef2 *= alphas[0]
            return coef1 * xs[-1] + coef2 * (adj @ xs[-1])
        coef_l = 2 * L * (L + a + b) * (2 * L - 2 + a + b)
        coef_lm1_1 = (2 * L + a + b - 1) * (2 * L + a + b) * (2 * L + a + b - 2)
        coef_lm1_2 = (2 * L + a + b - 1) * (a**2 - b**2)
        coef_lm2 = 2 * (L - 1 + a) * (L - 1 + b) * (2 * L + a + b)
        tmp1 = alphas[L - 1] * (coef_lm1_1 / coef_l)
        tmp2 = alphas[L - 1] * (coef_lm1_2 / coef_l)
        tmp3 = alphas[L - 1] * alphas[L - 2] * (coef_lm2 / coef_l)
        tmp1_2 = tmp1 * (2 / (r - l))
        tmp2_2 = tmp1 * ((r + l) / (r - l)) + tmp2
        nx = tmp1_2 * (adj @ xs[-1]) - tmp2_2 * xs[-1]
        nx -= tmp3 * xs[-2]
        return nx


    def forward(self, x: Tensor, adj: Tensor):
        '''
        Args:
            x: node embeddings. of shape (number of nodes, node feature dimension)
            edge_index and edge_attr: If the adjacency is cached, they will be ignored.
        '''
        
        self.adj = adj
        alphas = [self.basealpha * torch.tanh(_) for _ in self.alphas]
        xs = [self.jacobiconv(0, [x], self.adj, alphas)]
        for L in range(1, self.depth + 1):
            tx = self.jacobiconv(L, xs, self.adj, alphas)
            xs.append(tx)
        
        xs = [x.unsqueeze(1) for x in xs]
        
        
        x = torch.cat(xs, dim=1)
        x = x * self.comb_weight
        x = torch.sum(x, dim=1)
        return x




class JacobiConvGNN(nn.Module):
    def __init__(
        self, 
        nfeat, 
        nclass, 
        nhid,
        K,  # JacobiConv_prop的深度参数
        cached=True, 
        alpha=1.0, 
        fixed=False,
        dropout=0.5, 
        lr=0.01, 
        weight_decay=5e-4,
        device=None
    ):
        super(JacobiConvGNN, self).__init__()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 线性层：输入到隐藏层
        self.lin1 = nn.Linear(nfeat, nhid)
        # 线性层：隐藏层到输出层
        self.lin2 = nn.Linear(nhid, nclass)
        # 自定义传播层：JacobiConv_prop
        self.prop1 = JacobiConv_prop(
            depth=K,
            cached=cached,
            alpha=alpha,
            fixed=fixed,
            nclass = nclass
        )
        # 超参数
        self.depth = K
        self.dropout = dropout  # 线性层后的dropout率
        self.lr = lr  # 学习率
        self.weight_decay = weight_decay  # 权重衰减
        self.comb_weight = nn.Parameter(torch.ones((1, self.depth, nclass)))

    def reset_parameters(self):
        """重置模型各层参数"""
        # 重置线性层参数
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        # 重置传播层的多项式系数参数
        with torch.no_grad():
            basealpha = self.prop1.basealpha
            for alpha_param in self.prop1.alphas:
                initial_val = min(1.0 / basealpha, 1.0)  # 初始值为min(1/alpha, 1)
                alpha_param.data.fill_(initial_val)       # 重置为初始值
                alpha_param.requires_grad = not self.prop1.fixed  # 是否可训练

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 节点特征矩阵 [N, nfeat]
            adj: 邻接矩阵 [N, N]（对称归一化后）
            
        Returns:
            节点类别对数概率 [N, nclass]
        """
        # 特征dropout
        x = F.dropout(x, p=self.dropout, training=self.training)
        # 第一层线性变换 + ReLU激活
        x = self.lin1(x)
        x = F.relu(x)
        # 第二层线性变换 + 特征dropout
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        # 应用Jacobi图卷积传播层
        x = self.prop1(x, adj)
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
        """
        训练模型
        
        Args:
            features: 节点特征矩阵 [N, nfeat]
            adj: 邻接矩阵 [N, N]（稀疏或密集）
            labels: 节点标签 [N]
            idx_train: 训练节点索引
            idx_val: 验证节点索引（可选）
            idx_test: 测试节点索引（可选）
            train_iters: 训练轮次
            initialize: 是否初始化参数
            verbose: 是否打印训练日志
            normalize: 是否归一化邻接矩阵
            patience: 早停耐心值
        """
        # 数据转换为Tensor并移动到设备
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
        self.data.adj = adj
        # 参数初始化
        # 参数初始化
        if initialize:
            self.reset_parameters()
        
        # 优化器设置（Adam优化器）
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        # 早停初始化
        best_val_loss = float('inf')
        best_weights = None
        current_patience = patience
        
        # 训练循环
        for epoch in range(train_iters):
            self.train()  # 训练模式
            optimizer.zero_grad()  # 梯度清零
            
            # 前向传播
            output = self.forward(self.data.x, self.data.adj)
            # 计算训练损失（仅训练集）
            loss_train = F.nll_loss(output[self.data.train_mask], self.data.y[self.data.train_mask])
            # 反向传播与梯度更新
            loss_train.backward()
            optimizer.step()
            
            # 打印训练日志（每10轮）
            if verbose and epoch % 10 == 0:
                print(f'Epoch {epoch:03d}, Train Loss: {loss_train.item():.4f}')
            
            # 验证阶段（如果有验证集）
            if idx_val is not None or self.data.val_mask is not None:
                self.eval()  # 评估模式
                with torch.no_grad():
                    output = self.forward(self.data.x, self.data.adj)
                    # 计算验证损失
                    val_loss = 0.0
                    if self.data.val_mask is not None:
                        val_mask = self.data.val_mask
                    else:
                        val_mask = torch.zeros_like(self.data.train_mask)
                        val_mask[idx_val] = True
                    val_loss = F.nll_loss(output[val_mask], self.data.y[val_mask])
                
                # 早停逻辑
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = deepcopy(self.state_dict())
                    current_patience = patience
                else:
                    current_patience -= 1
                    if current_patience <= 0:
                        if verbose:
                            print(f'Early stopping at epoch {epoch}, Best Val Loss: {best_val_loss:.4f}')
                        break
        
        # 加载最佳模型参数（如果有）
        if best_weights is not None:
            self.load_state_dict(best_weights)
            # 用最佳模型计算输出（可选）
            self.output = self.forward(self.data.x, self.data.adj)
        else:
            self.output = output

    def test(self, idx_test=None):
        """
        评估测试集性能
        
        Args:
            idx_test: 测试节点索引（可选）
            
        Returns:
            测试准确率
        """
        self.eval()  # 评估模式
        # 确定测试索引
        if idx_test is None:
            if hasattr(self.data, 'test_mask') and self.data.test_mask is not None:
                idx_test = self.data.test_mask.nonzero(as_tuple=True)[0]
            else:
                raise ValueError("Test indices not provided and data has no test_mask.")
        idx_test = torch.tensor(idx_test, dtype=torch.long).to(self.data.x.device)
        
        # 前向传播并计算指标
        with torch.no_grad():
            output = self.forward(self.data.x, self.data.adj)
            loss_test = F.nll_loss(output[idx_test], self.data.y[idx_test])
            acc_test = utils.accuracy(output[idx_test], self.data.y[idx_test])
        return acc_test.item()

    def predict(self, x=None, adj=None):
        """
        预测节点类别概率
        
        Args:
            x: 节点特征矩阵（若为None则使用训练数据）
            adj: 邻接矩阵（若为None则使用训练数据）
            
        Returns:
            节点类别对数概率 [N, nclass]
        """
        self.eval()
        # 使用存储数据或外部输入
        if x is None or adj is None:
            if hasattr(self, 'data'):
                x, adj = self.data.x, self.data.adj
            else:
                raise ValueError("x and adj must be provided if no data is stored.")
        return self.forward(x, adj)

    @staticmethod
    def _to_tensor(data, device):
        """辅助函数：将数据转换为指定设备的Tensor"""
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        return data.to(device)