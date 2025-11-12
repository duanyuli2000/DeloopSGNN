import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from deeprobust.graph import utils
from copy import deepcopy
from torch_geometric.nn import ChebConv
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

class ChebNet(nn.Module):
    """ 2 Layer ChebNet based on pytorch geometric.

    Parameters
    ----------
    nfeat : int
        size of input feature dimension
    nhid : int
        number of hidden units
    nclass : int
        size of output dimension
    num_hops: int
        number of hops in ChebConv (K)
    dropout : float
        dropout rate for ChebNet
    lr : float
        learning rate for ChebNet
    weight_decay : float
        weight decay coefficient (l2 normalization)
    with_bias: bool
        whether to include bias term in ChebConv weights
    device: str
        'cpu' or 'cuda'

    Examples
    --------
    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import ChebNet
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> cheby = ChebNet(nfeat=features.shape[1],
              nhid=16, num_hops=3,
              nclass=labels.max().item() + 1,
              dropout=0.5, device='cpu')
    >>> cheby.fit(features, adj, labels, idx_train, idx_val, patience=10, verbose=True)
    >>> cheby.test(idx_test)
    """

    def __init__(self, nfeat, nhid, nclass, num_hops=3, dropout=0.5, lr=0.01,
                 weight_decay=5e-4, with_bias=True, device=None):

        super(ChebNet, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device

        # ChebConv层：第一层输入nfeat，输出nhid，使用num_hops阶切比雪夫多项式
        self.conv1 = ChebConv(
            in_channels=nfeat,
            out_channels=nhid,
            K=num_hops,
            bias=with_bias)

        # 第二层输入nhid，输出nclass（最终分类数）
        self.conv2 = ChebConv(
            in_channels=nhid,
            out_channels=nclass,
            K=num_hops,
            bias=with_bias)

        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.output = None
        self.best_model = None
        self.best_output = None

    def forward(self, x, edge_index):
        """前向传播：输入节点特征和边索引，输出log概率
        
        Parameters
        ----------
        x : torch.FloatTensor
            节点特征矩阵，形状[N, nfeat]
        edge_index : torch.LongTensor
            边索引矩阵，形状[2, E]
        
        Returns
        -------
        torch.FloatTensor
            各节点的类别log概率，形状[N, nclass]
        """
        x = F.relu(self.conv1(x, edge_index))  # 第一层ChebConv + ReLU
        x = F.dropout(x, p=self.dropout, training=self.training)  # 随机失活
        x = self.conv2(x, edge_index)  # 第二层ChebConv（无ReLU）
        return F.log_softmax(x, dim=1)  # 对类别维度取log_softmax

    def initialize(self):
        """初始化模型参数（重置卷积层权重）"""
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def fit(self, features, adj, labels, idx_train, idx_val=None, idx_test=None,
            train_iters=2000, initialize=True, verbose=False, normalize=True, patience=100):
        """训练ChebNet模型（支持早停）
        
        Parameters
        ----------
        features : np.ndarray/torch.Tensor
            节点特征矩阵，形状[N, nfeat]
        adj : np.ndarray/torch.Tensor/scipy.sparse矩阵
            邻接矩阵，形状[N, N]
        labels : np.ndarray/torch.Tensor
            节点标签，形状[N,]
        idx_train : np.ndarray/torch.Tensor
            训练节点索引
        idx_val : np.ndarray/torch.Tensor, optional
            验证节点索引（若为None则不启用早停）
        idx_test : np.ndarray/torch.Tensor, optional
            测试节点索引
        train_iters : int, optional
            训练轮次
        initialize : bool, optional
            是否初始化参数
        verbose : bool, optional
            是否打印训练日志
        normalize : bool, optional
            是否对邻接矩阵归一化（默认True）
        patience : int, optional
            早停耐心值（仅当idx_val非空时有效）
        """

        # 转换输入为torch.Tensor并移动到目标设备
        features = self._to_tensor(features, dtype=torch.float32).to(self.device)
        adj = self._to_tensor(adj, dtype=torch.float32).to(self.device)
        labels = self._to_tensor(labels, dtype=torch.long).to(self.device)

        # 处理索引（转换为布尔掩码）
        idx_train = self._to_tensor(idx_train, dtype=torch.long).to(self.device)
        self.data = self._build_pyg_data(features, adj, labels, idx_train, idx_val, idx_test)

        # 初始化参数
        if initialize:
            self.initialize()

        # 训练（带早停）
        self._train_with_early_stopping(train_iters, patience, verbose)

    def _build_pyg_data(self, features, adj, labels, idx_train, idx_val, idx_test):
        """将输入数据转换为PyG的Data对象（含mask）"""
        # 构造基础Data对象
        data = Data(x=features, edge_index=self._adj_to_edge_index(adj), y=labels)

        # 生成训练/验证/测试掩码
        data.train_mask = self._index_to_mask(idx_train, data.num_nodes)
        if idx_val is not None:
            data.val_mask = self._index_to_mask(idx_val, data.num_nodes)
        else:
            data.val_mask = None
        if idx_test is not None:
            data.test_mask = self._index_to_mask(idx_test, data.num_nodes)
        else:
            data.test_mask = None

        return data

    def _adj_to_edge_index(self, adj):
        """将邻接矩阵转换为PyG的edge_index格式（处理稀疏/密集矩阵）"""
        if utils.is_sparse_tensor(adj):
            edge_index, _ = utils.sparse_to_edge_index(adj)
        else:
            edge_index, _ = dense_to_sparse(adj)  # 来自torch_geometric.utils
        return edge_index

    @staticmethod
    def _index_to_mask(indices, num_nodes):
        """将索引转换为布尔掩码（用于PyG的Data.mask）"""
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[indices] = True
        return mask

    @staticmethod
    def _to_tensor(data, dtype):
        """通用转换函数：将输入转为指定类型的torch.Tensor"""
        if not isinstance(data, torch.Tensor):
            return torch.tensor(data, dtype=dtype)
        return data

    def _train_with_early_stopping(self, train_iters, patience, verbose):
        """早停训练循环"""
        if verbose:
            print('=== Training ChebNet model ===')

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_loss_val = float('inf')
        best_weights = None
        current_patience = patience

        for epoch in range(train_iters):
            self.train()  # 训练模式（激活Dropout）
            optimizer.zero_grad()

            # 前向传播
            output = self.forward(self.data.x, self.data.edge_index)
            loss_train = F.nll_loss(output[self.data.train_mask], self.data.y[self.data.train_mask])
            loss_train.backward()
            optimizer.step()

            # 打印训练日志
            if verbose and epoch % 10 == 0:
                print(f'Epoch {epoch}, Training Loss: {loss_train.item():.4f}')

            # 验证
            self.eval()  # 评估模式（关闭Dropout）
            output = self.forward(self.data.x, self.data.edge_index)
            loss_val = F.nll_loss(output[self.data.val_mask], self.data.y[self.data.val_mask]) if self.data.val_mask is not None else 0.0

            # 更新最佳模型
            if self.data.val_mask is not None and loss_val < best_loss_val:
                best_loss_val = loss_val
                best_weights = deepcopy(self.state_dict())
                current_patience = patience
            elif self.data.val_mask is not None:
                current_patience -= 1
                if current_patience <= 0:
                    if verbose:
                        print(f'Early stopping at epoch {epoch}, best val loss: {best_loss_val:.4f}')
                    break

        # 加载最佳模型
        if best_weights is not None:
            self.load_state_dict(best_weights)
            self.output = self.forward(self.data.x, self.data.edge_index)
        else:
            self.output = output

    def test(self, idx_test=None):
        """评估模型在测试集上的性能
        
        Parameters
        ----------
        idx_test : np.ndarray/torch.Tensor, optional
            测试节点索引（若为None则使用data.test_mask）
        
        Returns
        -------
        float
            测试准确率
        """
        self.eval()
        if idx_test is None:
            if self.data.test_mask is not None:
                idx_test = self.data.test_mask.nonzero(as_tuple=True)[0]
            else:
                raise ValueError("Test indices not provided and data has no test_mask.")
        idx_test = self._to_tensor(idx_test, dtype=torch.long).to(self.device)

        output = self.forward(self.data.x, self.data.edge_index)
        loss_test = F.nll_loss(output[idx_test], self.data.y[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.data.y[idx_test])
        # print(f"Test Results: Loss = {loss_test.item():.4f}, Accuracy = {acc_test.item():.4f}")
        return acc_test.item()

    def predict(self, x=None, edge_index=None):
        """预测节点类别概率
        
        Parameters
        ----------
        x : torch.FloatTensor, optional
            节点特征（若为None则使用存储的data.x）
        edge_index : torch.LongTensor, optional
            边索引（若为None则使用存储的data.edge_index）
        
        Returns
        -------
        torch.FloatTensor
            各节点的类别log概率（形状[N, nclass]）
        """
        self.eval()
        if x is None or edge_index is None:
            if hasattr(self, 'data'):
                x = self.data.x
                edge_index = self.data.edge_index
            else:
                raise ValueError("x and edge_index must be provided if no data is stored.")
        return self.forward(x, edge_index)