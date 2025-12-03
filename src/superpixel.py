import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.color import rgb2lab

def compute_superpixels(img_bgr, num_segments=200, compactness=10):
    seg = slic(img_bgr, n_segments=num_segments, compactness=compactness, start_label=0)
    return seg

def compute_superpixel_features(img_bgr, segments):
    lab = rgb2lab(img_bgr)
    num_sp = segments.max() + 1

    feats = np.zeros((num_sp, 3), dtype=np.float32)
    for sp in range(num_sp):
        mask = segments == sp
        feats[sp] = lab[mask].mean(axis=0)
    return feats

def build_superpixel_adjacency(segments):
    h, w = segments.shape
    edges = set()

    for y in range(h - 1):
        for x in range(w - 1):
            s = segments[y, x]
            for ny, nx in [(y + 1, x), (y, x + 1)]:
                t = segments[ny, nx]
                if s != t:
                    edges.add(tuple(sorted((s, t))))

    edges = np.array(list(edges), dtype=np.int32)
    return edges