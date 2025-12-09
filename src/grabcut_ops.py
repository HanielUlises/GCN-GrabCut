import numpy as np
import cv2


class GrabCutRefiner:
    """
    Wrapper around OpenCV GrabCut that accepts:
    - a predicted trimap from the GNN
    - user clicks / rectangles
    """

    def __init__(self, iterations=5):
        self.iterations = iterations

    def run(self, img_bgr, init_mask):
        if init_mask is None:
            raise ValueError("init_mask cannot be None when using GC_INIT_WITH_MASK")

        mask = init_mask.copy()
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)

        cv2.grabCut(
            img_bgr,
            mask,
            None,
            bgd,
            fgd,
            self.iterations,
            cv2.GC_INIT_WITH_MASK
        )

        return mask

    @staticmethod
    def to_binary(mask):
        """Return binary foreground mask."""
        fg = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
        return fg.astype(np.uint8)

    @staticmethod
    def init_from_rect(img_bgr, rect):
        x, y, w, h = rect
        mask = np.full(img_bgr.shape[:2], cv2.GC_PR_BGD, np.uint8)
        mask[y:y + h, x:x + w] = cv2.GC_PR_FGD
        return mask

    @staticmethod
    def init_from_superpixels(segments, fg_ids, bg_ids):
        mask = np.full(segments.shape, cv2.GC_PR_BGD, np.uint8)

        for sp in fg_ids:
            mask[segments == sp] = cv2.GC_PR_FGD
        for sp in bg_ids:
            mask[segments == sp] = cv2.GC_BGD

        return mask

    @staticmethod
    def refine(mask):
        """Smooths mask before thresholding to remove noise."""
        m = mask.astype(np.float32)
        m = cv2.GaussianBlur(m, (5, 5), 0)
        return (m > 0.5).astype(np.uint8)
