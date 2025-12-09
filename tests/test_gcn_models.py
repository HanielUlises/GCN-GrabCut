import torch
from torch_geometric.data import Data

from src.gcn_models import GCN_Segmenter, GraphSAGE_Segmenter


def build_dummy_graph(num_nodes=5, in_features=3):
    x = torch.randn((num_nodes, in_features))
    edge_index = torch.tensor([[0, 1, 2, 3],
                               [1, 2, 3, 4]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


def test_gcn_segmenter_forward():
    data = build_dummy_graph()
    model = GCN_Segmenter(in_channels=3, hidden_channels=4, out_channels=2)

    out = model(data)
    assert out.shape == (5, 2)
    assert torch.isfinite(out).all()


def test_graphsage_segmenter_forward():
    data = build_dummy_graph()
    model = GraphSAGE_Segmenter(in_channels=3, hidden_channels=4, out_channels=2)

    out = model(data)
    assert out.shape == (5, 2)
    assert torch.isfinite(out).all()
