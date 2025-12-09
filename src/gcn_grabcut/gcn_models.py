import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv


class GCN_Segmenter(nn.Module):
    """
    2-layer GCN for superpixel node classification.
    Produces logits of shape [N_nodes, out_channels].
    """

    def __init__(self, in_channels, hidden_channels=32, out_channels=2, dropout=0.2):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels, add_self_loops=True)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, add_self_loops=True)
        self.conv3 = GCNConv(hidden_channels, out_channels, add_self_loops=True)

        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)

        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x.contiguous(), data.edge_index

        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        logits = self.conv3(x, edge_index)
        return logits  # shape [N, out_channels]


class GraphSAGE_Segmenter(nn.Module):
    """
    GraphSAGE-based model for superpixel node segmentation.
    """

    def __init__(self, in_channels, hidden_channels=32, out_channels=2, dropout=0.2):
        super().__init__()

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)

        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)

        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x.contiguous(), data.edge_index

        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        logits = self.conv3(x, edge_index)
        return logits
