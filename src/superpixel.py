import numpy as np
from skimage.segmentation import slic
from skimage.color import rgb2lab


class SuperpixelExtractor:
    def __init__(self, num_segments=200, compactness=10):
        self.num_segments = num_segments
        self.compactness = compactness

    def compute(self, img_bgr):
        seg = slic(
            img_bgr,
            n_segments=self.num_segments,
            compactness=self.compactness,
            start_label=0
        )
        return seg

    def features(self, img_bgr, segments):
        lab = rgb2lab(img_bgr)
        n = segments.max() + 1
        feats = np.zeros((n, 3), dtype=np.float32)
        for k in range(n):
            mk = segments == k
            feats[k] = lab[mk].mean(0)
        return feats

    def adjacency(self, segments):
        h, w = segments.shape
        edges = set()
        for y in range(h - 1):
            for x in range(w - 1):
                s = segments[y, x]
                t1 = segments[y, x + 1]
                t2 = segments[y + 1, x]
                if s != t1:
                    edges.add((min(s, t1), max(s, t1)))
                if s != t2:
                    edges.add((min(s, t2), max(s, t2)))
        return np.array(list(edges), dtype=np.int32)
