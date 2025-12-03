import cv2
import numpy as np

def run_grabcut(img_bgr, init_mask, iters=5):
    """init_mask: 0=bg, 1=fg, 2=prob-bg, 3=prob-fg"""
    h, w = init_mask.shape
    mask = init_mask.copy()

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        img_bgr, mask, None, bgd, fgd,
        iters, cv2.GC_INIT_WITH_MASK
    )

    return mask

def mask_to_binary(mask):
    fg = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    return fg.astype(np.uint8)

def rect_init_mask(img_bgr, rect):
    x, y, w, h = rect
    mask = np.full(img_bgr.shape[:2], cv2.GC_PR_BGD, np.uint8)
    mask[y:y+h, x:x+w] = cv2.GC_PR_FGD
    return mask

def superpixel_prior_mask(segments, fg_ids, bg_ids):
    mask = np.full(segments.shape, cv2.GC_PR_BGD, np.uint8)
    for sp in fg_ids:
        mask[segments == sp] = cv2.GC_PR_FGD
    for sp in bg_ids:
        mask[segments == sp] = cv2.GC_PR_BGD
    return mask

def refine_mask_with_crf(mask, unary_fg, unary_bg, weight=3.0):
    m = mask.copy().astype(np.float32)
    m = cv2.GaussianBlur(m, (3, 3), 0)
    m = m > 0.5
    return m.astype(np.uint8)