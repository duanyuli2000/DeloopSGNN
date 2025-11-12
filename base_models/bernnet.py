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


class Bern_prop(MessagePassing):
    def __init__(self, K, bias=True, **kwargs):
        super(Bern_prop, self).__init__(aggr="add", **kwargs)

        self.K = K
        self.temp = Parameter(torch.Tensor(self.K + 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1)

    def forward(self, x, edge_index, edge_weight=None):
        TEMP = F.relu(self.temp)

        # L=I-D^(-0.5)AD^(-0.5)
        edge_index1, norm1 = get_laplacian(
            edge_index,
            edge_weight,
            normalization="sym",
            dtype=x.dtype,
            num_nodes=x.size(self.node_dim),
        )
        # 2I-L
        edge_index2, norm2 = add_self_loops(
            edge_index1, -norm1, fill_value=2.0, num_nodes=x.size(self.node_dim)
        )

        tmp = []
        tmp.append(x)
        for i in range(self.K):
            x = self.propagate(edge_index2, x=x, norm=norm2, size=None)
            tmp.append(x)

        out = (comb(self.K, 0) / (2**self.K)) * TEMP[0] * tmp[self.K]

        for i in range(self.K):
            x = tmp[self.K - i - 1]
            x = self.propagate(edge_index1, x=x, norm=norm1, size=None)
            for j in range(i):
                x = self.propagate(edge_index1, x=x, norm=norm1, size=None)
            out = out + (comb(self.K, i + 1) / (2**self.K)) * TEMP[i + 1] * x
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return "{}(K={}, temp={})".format(self.__class__.__name__, self.K, self.temp)
class BernNet(torch.nn.Module):
    def __init__(self, nfeat, nclass, nhid, K=10, dropout=0.5,dprate = 0.5, lr=0.01, weight_decay=5e-4,device=None):
        super(BernNet, self).__init__()
        assert device is not None, "Please specify 'device'!"
        self.device = device
        
        self.lin1 = Linear(nfeat, nhid)
        self.lin2 = Linear(nhid, nclass)
        self.prop1 = Bern_prop(K)
        self.dprate = dprate
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)

    def fit(self, features, adj, labels, idx_train, idx_val=None, idx_test=None, train_iters=1000, initialize=True, verbose=False, normalize=True, patience=100):
        # 转换输入为Tensor并移动到设备
        device = self.device if hasattr(self, 'device') else 'cpu'
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        if not isinstance(adj, torch.Tensor):
            adj = torch.tensor(adj, dtype=torch.float32)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)

        features = features.to(device)
        adj = adj.to(device)
        labels = labels.to(device)

        # 处理索引
        idx_train = torch.tensor(idx_train, dtype=torch.long).to(device)
        self.data = Data(x=features, edge_index=None, y=labels)
        self.data.train_mask = torch.zeros(features.shape[0], dtype=torch.bool, device=device)
        self.data.train_mask[idx_train] = True

        if idx_val is not None:
            idx_val = torch.tensor(idx_val, dtype=torch.long).to(device)
            self.data.val_mask = torch.zeros(features.shape[0], dtype=torch.bool, device=device)
            self.data.val_mask[idx_val] = True
        else:
            self.data.val_mask = None

        if idx_test is not None:
            idx_test = torch.tensor(idx_test, dtype=torch.long).to(device)
            self.data.test_mask = torch.zeros(features.shape[0], dtype=torch.bool, device=device)
            self.data.test_mask[idx_test] = True
        else:
            self.data.test_mask = None

        # 归一化邻接矩阵
        if normalize:
            adj = utils.normalize_adj_tensor(adj)

        # 转换为边索引并添加自环
        if utils.is_sparse_tensor(adj):
            edge_index, _ = utils.sparse_to_edge_index(adj)
        else:
            edge_index, _ = dense_to_sparse(adj)
        edge_index, _ = add_self_loops(edge_index, num_nodes=features.shape[0])

        self.data.edge_index = edge_index

        # 初始化参数
        if initialize:
            self.reset_parameters()

        # 训练参数
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_loss_val = float('inf')
        best_weights = None
        current_patience = patience

        # 早停训练
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
            loss_val = 0.0
            if self.data.val_mask is not None:
                loss_val = F.nll_loss(output[self.data.val_mask], self.data.y[self.data.val_mask])

            # 更新最佳模型
            if self.data.val_mask is not None and loss_val < best_loss_val:
                best_loss_val = loss_val
                best_weights = copy.deepcopy(self.state_dict())
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
        self.eval()
        if idx_test is None:
            if self.data.test_mask is not None:
                idx_test = self.data.test_mask.nonzero(as_tuple=True)[0]
            else:
                raise ValueError("Test indices not provided and data has no test_mask.")
        idx_test = torch.tensor(idx_test, dtype=torch.long).to(self.data.x.device)

        output = self.forward(self.data.x, self.data.edge_index)
        loss_test = F.nll_loss(output[idx_test], self.data.y[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.data.y[idx_test])
        return acc_test.item()

    def predict(self, x=None, edge_index=None):
        self.eval()
        if x is None or edge_index is None:
            if hasattr(self, 'data'):
                x = self.data.x
                edge_index = self.data.edge_index
            else:
                raise ValueError("x and edge_index must be provided if no data is stored.")
        return self.forward(x, edge_index)