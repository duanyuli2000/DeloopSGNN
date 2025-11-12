import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, dense_to_sparse
from deeprobust.graph import utils
import copy
from torch_geometric.nn.conv import FAConv

class FAGCN(nn.Module):
    def __init__(
        self, 
        nfeat, 
        nhid, 
        nclass, 
        dropout=0.5,
        dprate = 0.5, 
        lr=0.01, 
        weight_decay=5e-4,
        with_relu=True, 
        num_layers = 3, 
        epsilon = 0.05,
        device=None
    ):
        super(FAGCN, self).__init__()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eps = epsilon
        self.layer_num = num_layers
        self.dropout = dropout

        # 构建FAConv层堆叠
        self.layers = nn.ModuleList()
        for _ in range(self.layer_num):
            self.layers.append(FAConv(nhid, dropout = self.dropout))

        # 输入到隐藏层的线性变换
        self.t1 = nn.Linear(nfeat, nhid)
        # 隐藏层到输出层的线性变换
        self.t2 = nn.Linear(nhid, nclass)
        
        self.reset_parameters()
        self.dprate = dprate
        self.with_relu = with_relu
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None

    def reset_parameters(self):
        """重置模型参数（Xavier初始化）"""
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2.weight, gain=1.414)
        # 可选：FAConv层参数初始化（若需要）
        for layer in self.layers:
            layer.reset_parameters()  # 假设FAConv有reset_parameters方法

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
        dict
            包含'out'（对数概率）和'emb'（节点嵌入）的字典
        """
        # 初始特征处理
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = torch.relu(self.t1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        raw = h  # 保存原始中间特征用于FAConv层
        
        # 多层FAConv传播
        for i in range(self.layer_num):
            h = self.layers[i](h, raw, edge_index)  # 传入当前特征、原始特征和边索引
        
        # 输出层
        h = self.t2(h)
        return {'out': F.log_softmax(h, dim=1), 'emb': h}  # 返回对数概率和嵌入

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
        """训练FAGCN模型
        
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
            最大训练轮次
        initialize : bool, optional
            是否初始化参数
        verbose : bool, optional
            是否打印训练日志
        normalize : bool, optional
            是否归一化邻接矩阵
        patience : int, optional
            早停耐心值（验证集无提升时停止）
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
        
        # 构建训练/验证/测试掩码
        self.data = Data(x=features, edge_index=None, y=labels)
        self.data.train_mask = torch.zeros(features.size(0), dtype=torch.bool, device=device)
        self.data.train_mask[idx_train] = True
        
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
            adj = utils.normalize_adj_tensor(adj)  # 对称归一化
        edge_index, _ = dense_to_sparse(adj)       # 稀疏矩阵转边索引
        edge_index, _ = add_self_loops(edge_index, num_nodes=features.size(0))  # 添加自环
        
        self.data.edge_index = edge_index  # 存储边索引
        
        # 参数初始化
        if initialize:
            self.reset_parameters()
        
        # 优化器与损失函数
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.NLLLoss()  # 负对数似然损失（与log_softmax匹配）
        
        # 早停变量
        best_val_loss = float('inf')
        best_weights = None
        current_patience = patience
        
        # 训练循环
        for epoch in range(train_iters):
            self.train()  # 训练模式（激活Dropout）
            optimizer.zero_grad()
            
            # 前向传播
            output = self.forward(self.data.x, self.data.edge_index)
            loss_train = criterion(output['out'][self.data.train_mask], self.data.y[self.data.train_mask])
            
            # 反向传播与优化
            loss_train.backward()
            optimizer.step()
            
            # 打印训练日志
            if verbose and epoch % 10 == 0:
                print(f'Epoch {epoch:03d}, Train Loss: {loss_train.item():.4f}')
            
            # 验证阶段（评估模式）
            self.eval()
            with torch.no_grad():
                output = self.forward(self.data.x, self.data.edge_index)
                val_loss = 0.0
                if self.data.val_mask is not None:
                    val_loss = criterion(output['out'][self.data.val_mask], self.data.y[self.data.val_mask])
            
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
        
        # 加载最佳模型参数
        if best_weights is not None:
            self.load_state_dict(best_weights)
            self.output = self.forward(self.data.x, self.data.edge_index)  # 保存最终输出
        else:
            self.output = output

    def test(self, idx_test=None):
        """评估测试集准确率
        
        Parameters
        ----------
        idx_test : np.ndarray/torch.Tensor, optional
            测试节点索引（若为None则使用训练时的test_mask）
            
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
            loss_test = F.nll_loss(output['out'][idx_test], self.data.y[idx_test])
            acc_test = utils.accuracy(output['out'][idx_test], self.data.y[idx_test])
        return acc_test.item()

    def predict(self, x=None, edge_index=None):
        """模型预测（返回对数概率或嵌入）
        
        Parameters
        ----------
        x : torch.FloatTensor, optional
            节点特征矩阵（若为None则使用训练数据）
        edge_index : torch.LongTensor, optional
            边索引（若为None则使用训练数据）
            
        Returns
        -------
        dict
            包含'out'（对数概率）和'emb'（节点嵌入）的字典
        """
        self.eval()
        # 使用存储数据或外部输入
        if x is None or edge_index is None:
            if hasattr(self, 'data'):
                x, edge_index = self.data.x, self.data.edge_index
            else:
                raise ValueError("x and edge_index must be provided if no data is stored.")
        return self.forward(x, edge_index)