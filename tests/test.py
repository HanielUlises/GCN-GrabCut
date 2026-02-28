"""
tests/test_all.py — Comprehensive test suite for GCN-GrabCut.
Run: pytest tests/ -v
"""

from __future__ import annotations
import sys, random
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# fixtures

def _img(H=64, W=64, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randint(20, 220, (H, W, 3), dtype=np.uint8)


def _circle_mask(H=64, W=64, r=20):
    m = np.zeros((H, W), dtype=np.uint8)
    cy, cx = H//2, W//2
    import cv2; cv2.circle(m, (cx, cy), r, 1, -1)
    return m


# grabcut

class TestGrabCut:

    def test_bbox_returns_binary(self):
        from gcn_grabcut.grabcut import GrabCut, GrabCutConfig
        img  = _img(100, 100)
        gc   = GrabCut(img, GrabCutConfig(n_iter=1))
        mask = gc.run_with_bbox((10, 10, 80, 80))
        assert mask.shape == (100, 100)
        assert set(np.unique(mask)).issubset({0, 1})

    def test_trimap_mode(self):
        from gcn_grabcut.grabcut import GrabCut, GrabCutConfig
        import cv2
        img    = _img(100, 100)
        trimap = np.full((100, 100), cv2.GC_PR_BGD, dtype=np.uint8)
        trimap[30:70, 30:70] = cv2.GC_PR_FGD
        gc     = GrabCut(img, GrabCutConfig(n_iter=1))
        mask   = gc.run_with_trimap(trimap)
        assert mask.shape == (100, 100)

    @pytest.mark.parametrize("cs", ["rgb", "hsv", "lab"])
    def test_color_spaces(self, cs):
        from gcn_grabcut.grabcut import GrabCut, GrabCutConfig
        img  = _img(80, 80)
        gc   = GrabCut(img, GrabCutConfig(n_iter=1, color_space=cs))
        mask = gc.run_with_bbox((5, 5, 70, 70))
        assert mask.shape == (80, 80)

    def test_history_logged(self):
        from gcn_grabcut.grabcut import GrabCut
        img = _img(80, 80)
        gc  = GrabCut(img)
        gc.run_with_bbox((5, 5, 70, 70))
        assert len(gc.history) == 1
        snap = gc.history[0]
        assert "fg_pixels" in snap.__dict__
        assert 0 <= snap.fg_ratio <= 1

    def test_overlay_shape(self):
        from gcn_grabcut.grabcut import GrabCut
        img = _img(80, 80)
        gc  = GrabCut(img)
        gc.run_with_bbox((5, 5, 70, 70))
        assert gc.overlay_mask().shape == (80, 80, 3)

    def test_crop_foreground_rgba(self):
        from gcn_grabcut.grabcut import GrabCut
        img = _img(80, 80)
        gc  = GrabCut(img)
        gc.run_with_bbox((5, 5, 70, 70))
        rgba = gc.crop_foreground()
        assert rgba.shape == (80, 80, 4)


# graph_builder

class TestGraphBuilder:

    def test_build_basic(self):
        from gcn_grabcut.graph_builder import GraphBuilder, SuperpixelGraphConfig
        img   = _img(64, 64)
        graph = GraphBuilder(img, SuperpixelGraphConfig(n_segments=50)).build()
        assert graph.n_nodes > 0
        assert graph.node_features.shape == (graph.n_nodes, 16)
        assert graph.edge_index.shape[0] == 2
        assert graph.n_edges == graph.edge_index.shape[1]

    def test_node_feature_range(self):
        from gcn_grabcut.graph_builder import GraphBuilder
        img   = _img(64, 64)
        graph = GraphBuilder(img).build()
        # Normalised colour channels
        assert graph.node_features[:, :6].min() >= -0.01
        assert graph.node_features[:, :6].max() <=  1.01

    def test_edge_attr_shape(self):
        from gcn_grabcut.graph_builder import GraphBuilder, N_EDGE_FEATS
        img   = _img(64, 64)
        graph = GraphBuilder(img).build()
        assert graph.edge_attr.shape == (graph.n_edges, N_EDGE_FEATS)

    def test_segments_cover_all(self):
        from gcn_grabcut.graph_builder import GraphBuilder
        img   = _img(64, 64)
        graph = GraphBuilder(img).build()
        assert np.unique(graph.segments).min() == 0
        assert np.unique(graph.segments).max() == graph.n_nodes - 1

    def test_encode_hints(self):
        from gcn_grabcut.graph_builder import GraphBuilder, encode_user_hints
        img   = _img(64, 64)
        graph = GraphBuilder(img).build()
        segs  = graph.segments
        hints = encode_user_hints(segs, [(32, 32)], [(2, 2)])
        assert hints.shape == (graph.n_nodes, 3)
        # FG node
        fg_nid = int(segs[32, 32])
        assert hints[fg_nid, 0] == 1.0
        # BG node
        bg_nid = int(segs[2, 2])
        assert hints[bg_nid, 1] == 1.0

    @pytest.mark.parametrize("connectivity", [4, 8])
    def test_connectivity(self, connectivity):
        from gcn_grabcut.graph_builder import GraphBuilder, SuperpixelGraphConfig
        img   = _img(64, 64)
        cfg   = SuperpixelGraphConfig(n_segments=50, connectivity=connectivity)
        graph = GraphBuilder(img, cfg).build()
        # 8-connectivity should produce at least as many edges as 4-connectivity
        assert graph.n_edges > 0

class TestDataset:

    def test_sample_clicks(self):
        from gcn_grabcut.dataset import sample_clicks
        mask = _circle_mask()
        fg, bg = sample_clicks(mask, n_fg=3, n_bg=3, erosion_radius=5)
        assert len(fg) >= 1
        assert len(bg) >= 1
        for r, c in fg:
            assert 0 <= r < mask.shape[0]

    def test_augment_preserves_shape(self):
        from gcn_grabcut.dataset import augment_sample
        img = _img(64, 64); mask = _circle_mask()
        a_img, a_mask = augment_sample(img, mask)
        assert a_img.shape == img.shape
        assert a_mask.shape == mask.shape

    def test_synthetic_dataset(self):
        from gcn_grabcut.dataset import make_synthetic_dataset
        samples = make_synthetic_dataset(n=10, size=64, seed=0)
        assert len(samples) >= 5
        s = samples[0]
        assert "image" in s and "gt_mask" in s
        assert "fg_points" in s and "bg_points" in s
        assert s["image"].dtype == np.uint8
        assert set(np.unique(s["gt_mask"])).issubset({0, 1})

    def test_split_dataset(self):
        from gcn_grabcut.dataset import make_synthetic_dataset, split_dataset
        samples = make_synthetic_dataset(n=40, size=64, seed=1)
        train, val, test = split_dataset(samples, val_ratio=0.2, test_ratio=0.1)
        total = len(train) + len(val) + len(test)
        assert total == len(samples)
        assert len(val) >= 1
        assert len(test) >= 1

    def test_derive_trimap_labels(self):
        from gcn_grabcut.dataset import derive_trimap_labels
        from gcn_grabcut.graph_builder import GraphBuilder
        img   = _img(64, 64)
        graph = GraphBuilder(img).build()
        mask  = _circle_mask()
        labels = derive_trimap_labels(graph.segments, mask)
        assert labels.shape == (graph.n_nodes,)
        assert set(np.unique(labels)).issubset({0, 1, 2})


class TestMetrics:

    def test_perfect(self):
        from gcn_grabcut.metrics import evaluate
        gt = _circle_mask()
        m  = evaluate(gt, gt)
        assert m.iou   == pytest.approx(1.0, abs=1e-4)
        assert m.dice  == pytest.approx(1.0, abs=1e-4)
        assert m.recall == pytest.approx(1.0, abs=1e-4)

    def test_zero_prediction(self):
        from gcn_grabcut.metrics import evaluate
        gt   = _circle_mask()
        pred = np.zeros_like(gt)
        m    = evaluate(pred, gt)
        assert m.iou < 0.01

    def test_iou_range(self):
        from gcn_grabcut.metrics import evaluate
        pred = (np.random.RandomState(5).rand(64, 64) > 0.5).astype(np.uint8)
        gt   = _circle_mask()
        m    = evaluate(pred, gt)
        assert 0 <= m.iou <= 1

    def test_boundary_f1_perfect(self):
        from gcn_grabcut.metrics import boundary_f1
        mask = _circle_mask()
        bf1  = boundary_f1(mask, mask)
        assert bf1 == pytest.approx(1.0, abs=1e-3)

    def test_evaluate_trimap(self):
        from gcn_grabcut.metrics import evaluate_trimap
        from gcn_grabcut.grabcut import Label
        gt     = _circle_mask()
        trimap = np.where(gt, Label.FG_DEFINITE, Label.BG_DEFINITE).astype(np.uint8)
        tm     = evaluate_trimap(trimap, gt)
        assert tm.fg_recall   > 0.95
        assert tm.bg_recall   > 0.95
        assert tm.bg_contamination < 0.01

    def test_as_dict(self):
        from gcn_grabcut.metrics import evaluate
        m = evaluate(_circle_mask(), _circle_mask())
        d = m.as_dict()
        assert "iou" in d and "dice" in d

class TestModel:

    @pytest.fixture(autouse=True)
    def require_torch(self):
        pytest.importorskip("torch")
        pytest.importorskip("torch_geometric")

    def _make_data(self, N=80, in_dim=19, edge_dim=4):
        import torch
        from torch_geometric.data import Data
        x = torch.randn(N, in_dim)
        src = torch.arange(N - 1); dst = torch.arange(1, N)
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
        edge_attr  = torch.rand(edge_index.size(1), edge_dim)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    @pytest.mark.parametrize("variant", ["gcn", "gat", "resgcn"])
    def test_forward_shape(self, variant):
        from gcn_grabcut.model import build_model
        model = build_model(variant=variant, hidden_channels=32, n_layers=2)
        data  = self._make_data()
        out   = model(data)
        assert out.shape == (80, 3)

    def test_resgcn_residual(self):
        """ResGCNNet should output different results with different inputs."""
        import torch
        from gcn_grabcut.model import ResGCNNet
        model = ResGCNNet(hidden_channels=32, n_layers=2)
        model.eval()
        d1 = self._make_data(seed=1)
        d2 = self._make_data(seed=2)

        def _make_data_seeded(seed):
            import torch
            from torch_geometric.data import Data
            rng = torch.Generator(); rng.manual_seed(seed)
            N = 80
            x = torch.randn(N, 19, generator=rng)
            src = torch.arange(N - 1); dst = torch.arange(1, N)
            ei = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
            ea = torch.rand(ei.size(1), 4, generator=rng)
            return Data(x=x, edge_index=ei, edge_attr=ea)

        d1 = _make_data_seeded(1); d2 = _make_data_seeded(2)
        with torch.no_grad():
            o1 = model(d1); o2 = model(d2)
        assert not torch.allclose(o1, o2)

    def test_predict_trimap(self):
        import torch
        from torch_geometric.data import Data
        from gcn_grabcut.model import ResGCNNet, TRIMAP_BG, TRIMAP_FG, TRIMAP_PROB_BG
        model = ResGCNNet(hidden_channels=32, n_layers=2)
        segs  = np.zeros((32, 32), dtype=np.int32)
        segs[16:, :] = 1
        N = 2
        x = torch.randn(N, 19)
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        ea = torch.rand(2, 4)
        data = Data(x=x, edge_index=ei, edge_attr=ea)
        tri  = model.predict_trimap(data, segs)
        assert tri.shape == (32, 32)
        assert set(np.unique(tri)).issubset({TRIMAP_BG, TRIMAP_FG, TRIMAP_PROB_BG, 3})

    def test_param_groups_resgcn(self):
        from gcn_grabcut.model import ResGCNNet
        model = ResGCNNet(hidden_channels=32, n_layers=3)
        groups = model.param_groups(base_lr=1e-3)
        assert len(groups) > 1
        # Layer-wise LR should decrease toward earlier layers
        lrs = [g["lr"] for g in groups if len(g["params"]) > 0]
        assert any(lr < 1e-3 for lr in lrs)


class TestTrainer:

    @pytest.fixture(autouse=True)
    def require_torch(self):
        pytest.importorskip("torch")
        pytest.importorskip("torch_geometric")

    def test_fit_synthetic(self, tmp_path):
        from gcn_grabcut.dataset import make_synthetic_dataset, split_dataset
        from gcn_grabcut.model import build_model
        from gcn_grabcut.trainer import Trainer, TrainConfig

        samples = make_synthetic_dataset(n=10, size=64, seed=0)
        train_s, val_s, _ = split_dataset(samples, val_ratio=0.2, test_ratio=0.0)

        model = build_model("gcn", hidden_channels=16, n_layers=2)
        cfg   = TrainConfig(n_epochs=3, lr=1e-3, early_stop_patience=99,
                            amp=False, verbose=False)
        trainer = Trainer(model=model, config=cfg, device="cpu",
                          save_dir=str(tmp_path))
        history = trainer.fit(train_s, val_s)
        assert len(history["train_loss"]) == 3
        assert (tmp_path / "final_model.pt").exists()

    def test_focal_loss(self):
        import torch
        from gcn_grabcut.trainer import FocalLoss
        loss_fn = FocalLoss(gamma=2.0)
        logits  = torch.randn(100, 3)
        labels  = torch.randint(0, 3, (100,))
        loss    = loss_fn(logits, labels)
        assert loss.item() > 0

    def test_label_smooth_loss(self):
        import torch
        from gcn_grabcut.trainer import LabelSmoothingCE
        loss_fn = LabelSmoothingCE(smoothing=0.1)
        logits  = torch.randn(50, 3)
        labels  = torch.randint(0, 3, (50,))
        loss    = loss_fn(logits, labels)
        assert loss.item() > 0


# -----------------------------------------------------------------------  pipeline (torch required)

class TestPipeline:

    @pytest.fixture(autouse=True)
    def require_torch(self):
        pytest.importorskip("torch")
        pytest.importorskip("torch_geometric")

    def test_segment_returns_result(self, tmp_path):
        from gcn_grabcut.model import ResGCNNet
        from gcn_grabcut.pipeline import GCNGrabCutPipeline

        img      = _img(100, 100)
        model    = ResGCNNet(hidden_channels=32, n_layers=2)
        pipeline = GCNGrabCutPipeline(model, device="cpu")
        result   = pipeline.segment(img, [(50, 50)], [(5, 5)])

        assert result.binary_mask.shape  == (100, 100)
        assert result.trimap.shape       == (100, 100)
        assert result.overlay.shape      == (100, 100, 3)
        assert result.rgba.shape         == (100, 100, 4)
        assert "gcn_inference" in result.timing

    def test_segment_bbox_fallback(self):
        from gcn_grabcut.model import GCNTrimapNet
        from gcn_grabcut.pipeline import GCNGrabCutPipeline

        img      = _img(100, 100)
        model    = GCNTrimapNet(hidden_channels=16, n_layers=2)
        pipeline = GCNGrabCutPipeline(model, device="cpu")
        result   = pipeline.segment_bbox(img, (10, 10, 80, 80))
        assert result.binary_mask.shape == (100, 100)

    def test_evaluate_against(self):
        from gcn_grabcut.model import GCNTrimapNet
        from gcn_grabcut.pipeline import GCNGrabCutPipeline

        img      = _img(100, 100)
        gt       = _circle_mask(100, 100)
        model    = GCNTrimapNet(hidden_channels=16, n_layers=2)
        pipeline = GCNGrabCutPipeline(model, device="cpu")
        result   = pipeline.segment(img, [(50, 50)], [(5, 5)])
        seg_m, tri_m = result.evaluate_against(gt)
        assert 0 <= seg_m.iou <= 1
        assert 0 <= tri_m.fg_recall <= 1
