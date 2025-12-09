import torch
import numpy as np

from src.superpixel import SuperpixelExtractor


def test_superpixel_shapes():
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    sp = SuperpixelExtractor(num_segments=50, compactness=10)

    segments = sp.compute(img)
    assert segments.shape == (64, 64)
    assert segments.max() < 50
    assert segments.min() >= 0


def test_superpixel_features():
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    sp = SuperpixelExtractor(num_segments=20, compactness=10)
    segments = sp.compute(img)

    feats = sp.features(img, segments)
    assert isinstance(feats, np.ndarray)
    assert feats.shape[0] <= 20
    assert feats.shape[1] > 0
