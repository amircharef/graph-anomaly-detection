import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class Dominant(nn.Module):
    """DOMINANT (Ding et al., SDM 2019): graph autoencoder for anomaly detection."""

    def __init__(self, in_dim, hid_dim=64, dropout=0.3, alpha=0.8):
        super().__init__()
        self.alpha, self.dropout = alpha, dropout
        self.enc1 = GCNConv(in_dim, hid_dim)
        self.enc2 = GCNConv(hid_dim, hid_dim)
        self.enc3 = GCNConv(hid_dim, hid_dim)
        self.attr_dec1 = GCNConv(hid_dim, hid_dim)
        self.attr_dec2 = GCNConv(hid_dim, in_dim)
        self.struct_dec = GCNConv(hid_dim, hid_dim)

    def _drop(self, h):
        return F.dropout(h, self.dropout, self.training)

    def forward(self, x, edge_index):
        h = self._drop(F.relu(self.enc1(x, edge_index)))
        h = self._drop(F.relu(self.enc2(h, edge_index)))
        z = self._drop(F.relu(self.enc3(h, edge_index)))
        xa = self._drop(F.relu(self.attr_dec1(z, edge_index)))
        x_hat = self.attr_dec2(xa, edge_index)
        s = self._drop(F.relu(self.struct_dec(z, edge_index)))
        a_hat = torch.sigmoid(s @ s.T)
        return x_hat, a_hat

    def node_scores(self, x, edge_index, adj):
        x_hat, a_hat = self(x, edge_index)
        attr_err = torch.sqrt(((x_hat - x) ** 2).sum(1) + 1e-12)
        struct_err = torch.sqrt(((a_hat - adj) ** 2).sum(1) + 1e-12)
        return self.alpha * attr_err + (1 - self.alpha) * struct_err