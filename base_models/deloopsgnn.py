import torch
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from deeprobust.graph import utils
import torch.optim as optim
from copy import deepcopy
from deeprobust.graph.utils import *
from scipy.stats import chi2

import torch
def adj_norm(adj,if_self_loop=True):    
    D = adj.sum(dim=1)
    if if_self_loop:
        D = D +1
    D_inv_sqrt = torch.pow(D, -0.5)

    D_inv_sqrt[D <= 0] = 0
    D_inv_sqrt = torch.diag(D_inv_sqrt)
    return D_inv_sqrt@adj@D_inv_sqrt
def lra_2_hop(adj):
    P2 = adj@adj
    P2.fill_diagonal_(0)
    return P2

def lra_3_hop(adj):
    degrees = adj.sum(dim=1).unsqueeze(0)
    P2 = lra_2_hop(adj)
    Q3 = P2@adj
    P3 = Q3 - adj*(degrees-1)
    P3.fill_diagonal_(0)
    return P3


def lra_4_hop(adj):
    P3 = lra_3_hop(adj)
    P2 = lra_2_hop(adj)
    Q3 = P2@adj
    Q4 = P3@adj
    degrees = adj.sum(dim=1).unsqueeze(0)
    Q3_diag = Q3.diagonal().unsqueeze(0)

    P4_0 = Q4 - (degrees - 1) * P2
    P4_1 = Q4 - degrees* P2  - Q3_diag + 4 * P2
    P4 = P4_0*(1-adj)+P4_1*adj
    P4.fill_diagonal_(0)
    return P4


def lra_5_hop(adj):
    P2 = lra_2_hop(adj)
    P3 = lra_3_hop(adj)
    P4 = lra_4_hop(adj)
    Q3 = P2@adj
    Q4 = P3@adj
    Q5 = P4@adj

    degrees = adj.sum(dim=1).unsqueeze(0)
    Q3_diag = Q3.diagonal().unsqueeze(0)
    Q4_diag = Q4.diagonal().unsqueeze(0)
    
    P5_0 = Q5.clone()
    P5_0 -= (degrees - 1) * P3  - adj@(adj*P2)
    P5_0 -= adj@adj*Q3_diag - 2 * adj@(adj*P2) 
    P5_1 = Q5.clone()    
    P5_1 -= adj@adj + (degrees-2)*P3 -  adj@(adj*P2) #
    P5_1 -= (adj@adj)*(Q3_diag- 2*P2+2) - 2 * adj@(adj*P2)
    P5_1 -= Q4_diag - 2*P3 - P2*(P2-1)

    P5 = P5_0*(1-adj)+P5_1*adj
    P5.fill_diagonal_(0)
    return P5    
def lra_k_hop(adj,k):
    if k==0:
        return torch.eye(adj.shape[0], dtype=torch.float32).to(adj.device)
    if k==1:
        return adj
    elif k==2:
        return lra_2_hop(adj)
    elif k==3:
        return lra_3_hop(adj)
    elif k==4:
        return lra_4_hop(adj)
    elif k==5:
        return lra_5_hop(adj)
    return torch.zeros_like(adj)


def adj_k_hop(adj,k):
    adj_k  = torch.eye(adj.shape[0], dtype=torch.float32).to(adj.device)
    for i in range(k):
        adj_k = adj_k @ adj
    return adj_k

def remove_abnormal(adj_k,features):
    feature_mean  = adj_k@features
    feature_mean = feature_mean/(adj_k.sum(0).unsqueeze(1))
    diff = torch.zeros_like(adj_k)
    for d in range(features.shape[1]):
        f1_unsqueezed = feature_mean[:,d].unsqueeze(1)  # 形状: (n, 1)
        f2_unsqueezed = features[:,d].unsqueeze(0)  # 形状: (1, n)
        diff += (f1_unsqueezed - f2_unsqueezed) ** 2  # 形状 (n, n)  
    
    diff = diff** 0.5
    diff_mean = (diff*adj_k).sum(1)/adj_k.sum(1)
    diff_mean = diff_mean.unsqueeze(1)
    
    
    
    diff_std = (diff-diff_mean)**2
    diff_std = (diff_std*adj_k).sum(1)/adj_k.sum(1)
    diff_std = diff_std**0.5
    diff_std = diff_std.unsqueeze(1)
    adj_k[diff>=diff_mean+2*diff_std]=0
    adj_k = (adj_k+adj_k.T)/2
    
    return adj_k


# TODO 这是无环路版本
def get_adj_list(adj, K):
    # 初始化
    Ak_list = []
    adj_n_k = adj_norm(adj+torch.eye(adj.shape[0]).to(adj.device),if_self_loop=False)
    
    for k in range(0, K+1):  # k=1 到 k=5
        if k<=5:
            adj_k_lf = lra_k_hop(adj, k)  # k-hop 无环路邻接矩阵
            adj_k_lf = adj_norm(adj_k_lf)
            Ak_list.append(adj_k_lf)
        else:
            adj_k_lf = adj_k_hop(adj_n_k,k)
            Ak_list.append(adj_k_lf)
    return Ak_list


class Deloop_prop(MessagePassing):
    '''
    propagation class for AcyclicGNN.
    '''

    def __init__(self, K):
        super(Deloop_prop, self).__init__(aggr='add')
        self.K = K
        # random init
        bound = np.sqrt(3/(K+1))
        TEMP = np.random.uniform(-bound, bound, K+1)
        TEMP = TEMP/np.sum(np.abs(TEMP))
        TEMP[0] = 0
        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        bound = np.sqrt(3/(self.K+1))
        TEMP = np.random.uniform(-bound, bound, self.K+1)
        TEMP[0] = 0
        TEMP = TEMP/np.sum(np.abs(TEMP))
        self.temp.data = self.temp

    def forward(self, x, adj_list):
        hidden = torch.zeros_like(x, dtype=x.dtype, device=x.device)
        for k in range(self.K+1):
            adj_k = adj_list[k]
            hidden += self.temp[k]*(adj_k @ x)
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


class DeloopSGNN(torch.nn.Module):
    def __init__(self,  nfeat, nhid, nclass, K=10, dropout=0.5, lr=0.01, weight_decay=5e-4,
                  with_relu=True, device=None):
        super(DeloopSGNN, self).__init__()
        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass

        self.lin1 = Linear(nfeat, nhid)
        self.lin2 = Linear(nhid, nclass)

        # 消息传递采用DeloopSGNN_prop
        self.prop1 = Deloop_prop(K)
        self.K = K
        self.dropout = dropout
        self.with_relu = with_relu
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.output = None
        self.best_model = None
        self.best_output = None
        self.features = None

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop1.reset_parameters()

    def forward(self, x, adj_list): 
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        if self.with_relu:
            x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.prop1(x, adj_list)
        return F.log_softmax(x, dim=1)

    def initialize(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop1.reset_parameters()

    def fit(self, features, adj, labels, idx_train, idx_val=None, train_iters=200, initialize=True, verbose=False, normalize=True, patience=500, **kwargs):
        """Train the gcn model, when idx_val is not None, pick the best model according to the validation loss.

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
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        normalize : bool
            whether to normalize the input adjacency matrix.
        patience : int
            patience for early stopping, only valid when `idx_val` is given
        """

        if initialize:
            self.initialize()

        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)


        self.adj_list = get_adj_list(adj,self.K)
        self.features = features
        self.labels = labels
        
        if self.K > len(self.adj_list)-1:
            self.K = len(self.adj_list)-1
            self.prop1.K = self.K
            self.prop1.reset_parameters()
        
        if idx_val is None:
            self._train_without_val(labels, idx_train, train_iters, verbose)
        else:
            if patience < train_iters:
                self._train_with_early_stopping(labels, idx_train, idx_val, train_iters, patience, verbose)
            else:
                self._train_with_val(labels, idx_train, idx_val, train_iters, verbose)

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_list)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.adj_list)
        self.output = output

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_list)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.features, self.adj_list)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

    def _train_with_early_stopping(self, labels, idx_train, idx_val, train_iters, patience, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        early_stopping = patience
        best_loss_val = 100

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_list)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.features, self.adj_list)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break

        if verbose:
             print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val) )
        self.load_state_dict(weights)

    def test(self, idx_test,verbose=False):
        """Evaluate GCN performance on test set.

        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        output = self.forward(self.features, self.adj_list)
        # output = self.output
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        if verbose:
            print("Test set results:",
                "loss= {:.4f}".format(loss_test.item()),
                "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()


    def predict(self):
        self.eval()
        return self.forward(self.features, self.adj_list)



