import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from src.grabcut_ops import refine_with_grabcut


def test_refine_with_grabcut_runs():
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    logits = torch.randn((64, 64, 2))

    mask = refine_with_grabcut(img, logits)

    assert mask.shape == (64, 64)
    assert mask.dtype == np.uint8
    assert mask.max() <= 1
    assert mask.min() >= 0
