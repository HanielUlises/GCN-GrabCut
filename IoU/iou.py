import cv2
import numpy as np
import matplotlib.pyplot as plt


def synthetic_circle(size=(256, 256), radius=60) -> np.ndarray:
    """Create a white background with a black circle."""
    img = np.full((size[0], size[1], 3), 255, np.uint8)
    center = (size[1] // 2, size[0] // 2)
    cv2.circle(img, center, radius, (0, 0, 0), -1)
    return img


def grabcut_segment(image: np.ndarray, rect: tuple, iterations=5) -> np.ndarray:
    """Run GrabCut given a rectangle; return binary mask (1=fg, 0=bg)."""
    mask = np.zeros(image.shape[:2], np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, rect, bg_model, fg_model, iterations, cv2.GC_INIT_WITH_RECT)
    return np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype(np.uint8)


def intersection_over_union(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0


def experiment_stability():
    """Quantify GrabCut stability under slight rectangle shifts."""
    image = synthetic_circle()

    base_rect = (60, 60, 130, 130)
    variations = [
        base_rect,
        (base_rect[0]-10, base_rect[1], base_rect[2], base_rect[3]),
        (base_rect[0]+10, base_rect[1], base_rect[2], base_rect[3]),
        (base_rect[0], base_rect[1]-10, base_rect[2], base_rect[3]),
        (base_rect[0], base_rect[1]+10, base_rect[2], base_rect[3])
    ]

    masks = [grabcut_segment(image, rect) for rect in variations]
    base_mask = masks[0]

    # Compute IoU with base mask
    ious = [intersection_over_union(base_mask, m) for m in masks]

    # Visualization
    plt.figure(figsize=(12, 5))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for i, (mask, rect) in enumerate(zip(masks, variations)):
        segmented = rgb.copy()
        segmented[mask == 0] = (255, 255, 255)
        plt.subplot(2, len(masks), i+1)
        plt.imshow(segmented)
        plt.title(f"Run {i+1}\nRect {rect}")
        plt.axis('off')

    plt.subplot(2, 1, 2)
    plt.plot(range(len(ious)), ious, marker='o')
    plt.title("GrabCut Stability — IoU vs. Rectangle Shift")
    plt.xlabel("Experiment Index")
    plt.ylabel("IoU with Base Segmentation")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("IoU scores:", np.round(ious, 3))


if __name__ == "__main__":
    experiment_stability()
