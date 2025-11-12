import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse
from torch_sparse import SparseTensor, matmul
from torch import Tensor
from torch_geometric.utils import get_laplacian
from deeprobust.graph import utils
from copy import deepcopy
import numpy as np

class H2GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, k=2, dropout=0.5, 
                 with_relu=True, lr=0.01, weight_decay=5e-4, 
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(H2GCN, self).__init__()
        self.k = k
        self.dropout = dropout
        self.with_relu = with_relu
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device

        # Feature transformation
        self.w_embed = nn.Parameter(torch.zeros(nfeat, nhid).to(self.device))
        # Final classification layer
        final_dim = (2**(k+1)-1) * nhid
        self.w_classify = nn.Parameter(torch.zeros(final_dim, nclass).to(self.device))

        # Propagation matrices
        self.a1 = None
        self.a2 = None
        
        self.reset_parameters()
        
        # Training states
        self.features = None
        self.labels = None
        self.output = None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_embed)
        nn.init.xavier_uniform_(self.w_classify)

    def _prepare_prop(self, adj):
        """Precompute normalized adjacency matrices for 1-hop and 2-hop propagation"""
        
        # 1-hop matrix (A - I)
        adj_1hop = adj.fill_diagonal_(0.0)  # Remove self-loops
        
        # 2-hop matrix (A^2 - A - I)
        adj_sq = adj_1hop @ adj_1hop
        adj_2hop = adj_sq - adj_1hop - torch.eye(adj.shape[0]).to(self.device)

        
        # Convert to sparse tensors
        self.a1 = self._normalize_adj(adj_1hop).to(self.device)
        self.a2 = self._normalize_adj(adj_2hop).to(self.device)

    def _normalize_adj(self, adj):
        # adj.fill_diagonal_(1.0)
        D = adj.sum(dim=1)
        D_inv_sqrt = torch.pow(D, -0.5)
        D_inv_sqrt[D_inv_sqrt == float('inf')] = 0
        D_inv_sqrt = torch.diag(D_inv_sqrt)
        return D_inv_sqrt@adj@D_inv_sqrt

    def forward(self, x):
        # Feature transformation
        x = torch.mm(x, self.w_embed)
        if self.with_relu:
            x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        # Propagation
        rs = [x]
        for _ in range(self.k):
            r1 = torch.sparse.mm(self.a1, rs[-1])
            r2 = torch.sparse.mm(self.a2, rs[-1])
            rs.append(torch.cat([r1, r2], dim=1))
        
        # Concatenate all layers
        h_final = torch.cat(rs, dim=1)
        h_final = F.dropout(h_final, self.dropout, training=self.training)
        return F.log_softmax(torch.mm(h_final, self.w_classify), dim=1)

    def fit(self, features, adj, labels, idx_train, idx_val=None, 
            train_iters=200, verbose=True, patience=500):
        # features = utils.to_tensor(features)
        # labels = utils.to_tensor(labels)
        self.features, self.labels = features.to(self.device), labels.to(self.device)
        
        # Process adjacency matrix
        # edge_index, edge_weight = get_laplacian(adj.nonzero().T, normalization='sym')
        self._prepare_prop(adj)

        # Training setup
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, 
            weight_decay=self.weight_decay
        )

        best_loss = float('inf')
        best_acc = 0
        patience_counter = 0
        best_state = None

        for epoch in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self(self.features)
            loss = F.nll_loss(output[idx_train], labels[idx_train])
            loss.backward()
            optimizer.step()

            # Validation
            if idx_val is not None:
                self.eval()
                with torch.no_grad():
                    val_loss = F.nll_loss(output[idx_val], labels[idx_val])
                    val_acc = utils.accuracy(output[idx_val], labels[idx_val])

                # Early stopping
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_acc = val_acc
                    best_state = deepcopy(self.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

            if verbose and epoch % 10 == 0:
                log = f"Epoch {epoch}: train_loss={loss.item():.4f}"
                if idx_val is not None:
                    log += f", val_loss={val_loss.item():.4f}, val_acc={val_acc:.4f}"
                print(log)

        # Restore best model
        if best_state is not None:
            self.load_state_dict(best_state)
        self.eval()
        self.output = self(self.features)

    def test(self, idx_test):
        """Evaluate model performance on test set"""
        assert self.output is not None, "Run fit first!"
        loss_test = F.nll_loss(self.output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(self.output[idx_test], self.labels[idx_test])
        # print(f"Test set results: loss={loss_test.item():.4f}, acc={acc_test.item():.4f}")
        return acc_test.item()

    def predict(self, features=None):
        """Predict node logits"""
        self.eval()
        if features is None:
            return self.output
        else:
            return self(features.to(self.device))
