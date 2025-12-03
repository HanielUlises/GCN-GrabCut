import cv2
import numpy as np


class GrabCutRefiner:
    def __init__(self, iterations=5):
        self.iterations = iterations

    def run(self, img_bgr, init_mask):
        mask = init_mask.copy()
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)

        cv2.grabCut(
            img_bgr, mask, None, bgd, fgd,
            self.iterations, cv2.GC_INIT_WITH_MASK
        )
        return mask

    @staticmethod
    def to_binary(mask):
        fg = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
        return fg.astype(np.uint8)

    @staticmethod
    def init_from_rect(img_bgr, rect):
        x, y, w, h = rect
        mask = np.full(img_bgr.shape[:2], cv2.GC_PR_BGD, np.uint8)
        mask[y:y+h, x:x+w] = cv2.GC_PR_FGD
        return mask

    @staticmethod
    def init_from_superpixels(segments, fg_ids, bg_ids):
        mask = np.full(segments.shape, cv2.GC_PR_BGD, np.uint8)
        for sp in fg_ids:
            mask[segments == sp] = cv2.GC_PR_FGD
        for sp in bg_ids:
            mask[segments == sp] = cv2.GC_PR_BGD
        return mask

    @staticmethod
    def refine(mask):
        m = mask.astype(np.float32)
        m = cv2.GaussianBlur(m, (3, 3), 0)
        return (m > 0.5).astype(np.uint8)
