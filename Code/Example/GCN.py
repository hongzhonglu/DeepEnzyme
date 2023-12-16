import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adj(adj):
    rowsum = adj.sum(1)
    r_inv_sqrt = torch.pow(rowsum, -0.5)
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    adj = torch.mm(torch.mm(r_mat_inv_sqrt, adj), r_mat_inv_sqrt)
    return adj


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads, dropout):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.MultiheadAttention(out_features, num_heads)

    def forward(self, x, adj):
        #x = F.relu(x)
        #x = self.dropout(x)
        x = self.linear(x)
        adj = adj + torch.eye(adj.size(0)).cuda()
        adj = normalize_adj(adj)
        x = torch.spmm(adj.to(torch.float32), x.to(torch.float32))
        #x = x.transpose(0, 1)
        #x, _ = self.attention(x, x, x)
        #x = x.transpose(0, 1)

        return x


class GCN(nn.Module):
    def __init__(self, in_features, hidden_featrures1, hidden_featrures2, out_features, num_heads, dropout):
        super(GCN, self).__init__()
        self.layer1 = GCNLayer(in_features, hidden_featrures2, num_heads, dropout)
        self.layer2 = GCNLayer(hidden_featrures1, hidden_featrures2, num_heads, dropout)
        self.layer3 = GCNLayer(hidden_featrures2, out_features, num_heads, dropout)
        self.layer4 = GCNLayer(in_features, out_features, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        #x = self.layer1(x, adj)
        #x = F.relu(x)
        #x = self.dropout(x)
        #x = self.layer2(x, adj)
        #x = self.layer3(x, adj)
        x = self.layer4(x, adj)
        return x
