"""
GCN-GrabCut

Fully automatic image segmentation: a Graph Convolutional Network predicts a
trimap from an image-derived superpixel graph (no user clicks), and classical
GrabCut refines it into a pixel-level mask.

"""

from .grabcut import GrabCut, GrabCutConfig, Label
from .graph_builder import (
    GraphBuilder, SuperpixelGraph, SuperpixelGraphConfig,
    compute_auto_prior, encode_user_hints,
    N_NODE_FEATS, N_EDGE_FEATS, N_PRIOR_FEATS,
)
from .metrics import (
    evaluate, SegmentationMetrics,
    evaluate_trimap, TrimapMetrics,
    evaluate_batch,
)
from .dataset import (
    load_image_mask_dataset, make_synthetic_dataset,
    split_dataset, sample_clicks, prepare_sample, prepare_dataset,
    augment_sample, derive_trimap_labels,
)
from .pipeline import (
    GCNGrabCutPipeline, SegmentationResult, clean_mask,
    guided_filter, refine_trimap,
)
try:
    from .losses import FocalLoss, LabelSmoothingCE, TrimapLoss
    from .trainer import Trainer, TrainConfig
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
        probs_to_node_trimap, project_to_pixels,
        TRIMAP_BG, TRIMAP_FG, TRIMAP_PROB_BG, TRIMAP_PROB_FG,
        CLASS_BG, CLASS_UNK, CLASS_FG,
    )
    _MODELS_AVAILABLE = True
except ImportError:
    _MODELS_AVAILABLE = False

__version__ = "0.3.0"
__author__  = "Haniel Ulises Vásquez Morales"

__all__ = [
    "GrabCut", "GrabCutConfig", "Label",

    "GraphBuilder", "SuperpixelGraph", "SuperpixelGraphConfig",
    "compute_auto_prior", "encode_user_hints",
    "N_NODE_FEATS", "N_EDGE_FEATS", "N_PRIOR_FEATS",

    "load_image_mask_dataset", "make_synthetic_dataset", "split_dataset",
    "sample_clicks", "prepare_sample", "prepare_dataset",
    "augment_sample", "derive_trimap_labels",

    "evaluate", "SegmentationMetrics",
    "evaluate_trimap", "TrimapMetrics", "evaluate_batch",

    "GCNGrabCutPipeline", "SegmentationResult", "clean_mask",
    "guided_filter", "refine_trimap",

    "Trainer", "TrainConfig", "FocalLoss", "LabelSmoothingCE", "TrimapLoss",

    "GCNTrimapNet", "GATTrimapNet", "ResGCNNet", "build_model",
    "probs_to_node_trimap", "project_to_pixels",

    "plot_training_curves", "plot_trimap_comparison",
    "plot_superpixel_graph", "plot_confusion_matrix", "save_research_report",
]
