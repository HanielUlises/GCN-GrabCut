"""
GCN-GrabCut

Interactive image segmentation using a Graph Convolutional Network for
automatic trimap prediction, followed by classical GrabCut refinement.

"""

from .grabcut import GrabCut, GrabCutConfig, Label
from .graph_builder import (
    GraphBuilder, SuperpixelGraph, SuperpixelGraphConfig,
    encode_user_hints, N_NODE_FEATS, N_EDGE_FEATS,
)
from .metrics import (
    evaluate, SegmentationMetrics,
    evaluate_trimap, TrimapMetrics,
    evaluate_batch,
)
from .dataset import (
    load_image_mask_dataset, make_synthetic_dataset,
    split_dataset, sample_clicks, prepare_sample,
    augment_sample, derive_trimap_labels,
)
from .pipeline import GCNGrabCutPipeline, SegmentationResult
try:
    from .trainer import Trainer, TrainConfig, FocalLoss, LabelSmoothingCE
except ImportError:
    pass
from .visualise import (
    plot_training_curves, plot_trimap_comparison,
    plot_superpixel_graph, plot_confusion_matrix,
    save_research_report,
)

# Model imports are guarded since torch might not be installed
try:
    from .model import (
        GCNTrimapNet, GATTrimapNet, ResGCNNet,
        build_model, _probs_to_trimap,
        TRIMAP_BG, TRIMAP_FG, TRIMAP_PROB_BG, TRIMAP_PROB_FG,
        CLASS_BG, CLASS_UNK, CLASS_FG,
    )
    _MODELS_AVAILABLE = True
except ImportError:
    _MODELS_AVAILABLE = False

__version__ = "0.2.0"
__author__  = "Haniel Ulises Vásquez Morales"

__all__ = [
    "GrabCut", "GrabCutConfig", "Label",

    "GraphBuilder", "SuperpixelGraph", "SuperpixelGraphConfig",
    "encode_user_hints", "N_NODE_FEATS", "N_EDGE_FEATS",

    "load_image_mask_dataset", "make_synthetic_dataset", "split_dataset",
    "sample_clicks", "prepare_sample", "augment_sample", "derive_trimap_labels",

    "evaluate", "SegmentationMetrics",
    "evaluate_trimap", "TrimapMetrics", "evaluate_batch",

    "GCNGrabCutPipeline", "SegmentationResult",

    "Trainer", "TrainConfig", "FocalLoss", "LabelSmoothingCE",

    "GCNTrimapNet", "GATTrimapNet", "ResGCNNet", "build_model",

    "plot_training_curves", "plot_trimap_comparison",
    "plot_superpixel_graph", "plot_confusion_matrix", "save_research_report",
]
