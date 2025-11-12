import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from deeprobust.graph import utils
from copy import deepcopy
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

class GAT(nn.Module):
    """ 2 Layer Graph Attention Network based on pytorch geometric.

    Parameters
    ----------
    nfeat : int
        size of input feature dimension
    nhid : int
        number of hidden units
    nclass : int
        size of output dimension
    heads: int
        number of attention heads
    output_heads: int
        number of attention output heads
    dropout : float
        dropout rate for GAT
    lr : float
        learning rate for GAT
    weight_decay : float
        weight decay coefficient (l2 normalization) for GAT.
    with_bias: bool
        whether to include bias term in GAT weights.
    device: str
        'cpu' or 'cuda'.

    Examples
    --------
    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import GAT
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> gat = GAT(nfeat=features.shape[1],
              nhid=8, heads=8,
              nclass=labels.max().item() + 1,
              dropout=0.5, device='cpu')
    >>> gat.fit(features, adj, labels, idx_train, idx_val, patience=100, verbose=True)
    >>> gat.test(idx_test)
    """

    def __init__(self, nfeat, nhid, nclass, heads=8, output_heads=1, dropout=0.5, lr=0.01,
                 weight_decay=5e-4, with_bias=True, device=None):

        super(GAT, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device

        self.conv1 = GATConv(
            in_channels=nfeat,
            out_channels=nhid,
            heads=heads,
            dropout=dropout,
            bias=with_bias,
            concat=True  # 输出维度为 nhid * heads
        )

        self.conv2 = GATConv(
            in_channels=nhid * heads,
            out_channels=nclass,
            heads=output_heads,
            concat=False,  # 输出维度为 nclass
            dropout=dropout,
            bias=with_bias
        )

        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.output = None
        self.best_model = None
        self.best_output = None

    def forward(self, x, edge_index):
        """Forward pass of GAT.

        Parameters
        ----------
        x : torch.FloatTensor
            Node feature matrix with shape [N, nfeat]
        edge_index : torch.LongTensor
            Graph edge indices with shape [2, E]

        Returns
        -------
        torch.FloatTensor
            Log probabilities of each class with shape [N, nclass]
        """
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def initialize(self):
        """Initialize parameters of GAT."""
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def fit(self, features, adj, labels, idx_train, idx_val=None, idx_test=None, train_iters=1000, initialize=True, verbose=False, normalize=True, patience=100, **kwargs):
        """Train the GAT model.

        Parameters
        ----------
        features : np.ndarray or torch.Tensor
            Node feature matrix with shape [N, nfeat]
        adj : np.ndarray or torch.Tensor or scipy.sparse matrix
            Adjacency matrix with shape [N, N]
        labels : np.ndarray or torch.Tensor
            Node labels with shape [N,]
        idx_train : np.ndarray or torch.Tensor
            Training node indices
        idx_val : np.ndarray or torch.Tensor, optional
            Validation node indices. If None, early stopping is not used.
        idx_test : np.ndarray or torch.Tensor, optional
            Test node indices
        train_iters : int, optional
            Number of training epochs
        initialize : bool, optional
            Whether to initialize parameters before training
        verbose : bool, optional
            Whether to print training progress
        normalize : bool, optional
            Whether to normalize the adjacency matrix
        patience : int, optional
            Patience for early stopping
        """

        # 转换输入为tensor并移动到设备
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        if not isinstance(adj, torch.Tensor):
            adj = torch.tensor(adj, dtype=torch.float32)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)

        features = features.to(self.device)
        adj = adj.to(self.device)
        labels = labels.to(self.device)

        # 处理索引
        idx_train = torch.tensor(idx_train, dtype=torch.long).to(self.device)
        if idx_val is not None:
            idx_val = torch.tensor(idx_val, dtype=torch.long).to(self.device)
        if idx_test is not None:
            idx_test = torch.tensor(idx_test, dtype=torch.long).to(self.device)

        # 归一化邻接矩阵
        if normalize:
            adj = utils.normalize_adj_tensor(adj)

        # 转换为边索引（处理稀疏矩阵）
        if utils.is_sparse_tensor(adj):
            edge_index, _ = utils.sparse_to_edge_index(adj)
        else:
            edge_index, _ = dense_to_sparse(adj)

        # 构造Data对象
        data = Data(x=features, edge_index=edge_index, y=labels)

        # 构造mask
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=self.device)
        data.train_mask[idx_train] = True

        if idx_val is not None:
            data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=self.device)
            data.val_mask[idx_val] = True
        else:
            data.val_mask = None

        if idx_test is not None:
            data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=self.device)
            data.test_mask[idx_test] = True
        else:
            data.test_mask = None

        # 初始化参数
        if initialize:
            self.initialize()

        # 训练
        self.data = data
        self._train_with_early_stopping(train_iters, patience, verbose)
  
    def _train_with_early_stopping(self, train_iters, patience, verbose):
        """Early stopping training loop."""
        if verbose:
            print('=== Training GAT model ===')

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_loss_val = float('inf')
        best_weights = None
        current_patience = patience

        for epoch in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.data.x, self.data.edge_index)
            loss_train = F.nll_loss(output[self.data.train_mask], self.data.y[self.data.train_mask])
            loss_train.backward()
            optimizer.step()

            if verbose and epoch % 10 == 0:
                print(f'Epoch {epoch}, Training Loss: {loss_train.item():.4f}')

            # 验证
            self.eval()
            output = self.forward(self.data.x, self.data.edge_index)
            if self.data.val_mask is not None:
                loss_val = F.nll_loss(output[self.data.val_mask], self.data.y[self.data.val_mask])
            else:
                loss_val = 0.0

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
        """Evaluate on test set.

        Parameters
        ----------
        idx_test : np.ndarray or torch.Tensor, optional
            Test node indices. If None, use data.test_mask.
        """
        self.eval()
        if idx_test is None:
            if self.data.test_mask is not None:
                idx_test = self.data.test_mask.nonzero(as_tuple=True)[0]
            else:
                raise ValueError("Test indices not provided and data has no test_mask.")
        idx_test = torch.tensor(idx_test, dtype=torch.long).to(self.device)

        output = self.forward(self.data.x, self.data.edge_index)
        loss_test = F.nll_loss(output[idx_test], self.data.y[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.data.y[idx_test])
        # print(f"Test Results: Loss = {loss_test.item():.4f}, Accuracy = {acc_test.item():.4f}")
        return acc_test.item()

    def predict(self, x=None, edge_index=None):
        """Predict using the model.

        Parameters
        ----------
        x : torch.FloatTensor, optional
            Node features. If None, use stored data.x.
        edge_index : torch.LongTensor, optional
            Edge indices. If None, use stored data.edge_index.

        Returns
        -------
        torch.FloatTensor
            Log probabilities of each class.
        """
        self.eval()
        if x is None or edge_index is None:
            if hasattr(self, 'data'):
                x = self.data.x
                edge_index = self.data.edge_index
            else:
                raise ValueError("x and edge_index must be provided if no data is stored.")
        return self.forward(x, edge_index)