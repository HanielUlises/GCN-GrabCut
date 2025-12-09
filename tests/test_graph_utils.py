import torch
import numpy as np

from gcn_grabcut.graph_utils import adjacency_from_segments, build_graph


def test_adjacency_from_segments():
    seg = np.array([
        [0, 0, 1],
        [0, 2, 2],
        [3, 3, 2]
    ], dtype=np.int32)

    edges = adjacency_from_segments(seg)
    assert edges.shape[0] == 2
    assert edges.shape[1] > 0
    # Directed edges but symmetric is expected
    assert edges.min() >= 0


def test_build_graph():
    seg = np.random.randint(0, 10, (32, 32))
    feats = np.random.randn(10, 5).astype(np.float32)

    data = build_graph(feats, seg)

    assert hasattr(data, "x")
    assert hasattr(data, "edge_index")
    assert data.x.shape[1] == 5
    assert data.edge_index.shape[0] == 2
    assert torch.isfinite(data.x).all()
