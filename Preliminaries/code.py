import cv2
import numpy as np
import matplotlib.pyplot as plt


def create_synthetic_image(shape: str, size: tuple = (256, 256)) -> np.ndarray:
    """
    Creates a synthetic RGB image containing a single geometric shape centered on a uniform background.

    Parameters:
        shape (str): The geometric shape to draw. Supported options: 'circle', 'square', 'triangle'.
        size (tuple): The dimensions of the output image (height, width).

    Returns:
        np.ndarray: The resulting image as a NumPy array in BGR format (as used by OpenCV).
    """
    
    # Blank RGB canvas with a uniform white background.
    # This isolates the shape visually, ensuring that GrabCut is influenced only by geometry—not texture or color variance.
    image = np.full((size[0], size[1], 3), 255, dtype=np.uint8)
    # This is the image center, so we can place all shapes symmetrically for fair comparison.
    center = (size[1] // 2, size[0] // 2)

    if shape == 'circle':
        cv2.circle(image, center, 60, (0, 0, 0), thickness=-1)

    elif shape == 'square':
        top_left = (center[0] - 40, center[1] - 40)
        bottom_right = (center[0] + 40, center[1] + 40)
        cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), thickness=-1)

    elif shape == 'triangle':
        vertices = np.array([
            [center[0], center[1] - 60],
            [center[0] - 50, center[1] + 40],
            [center[0] + 50, center[1] + 40]
        ])
        cv2.drawContours(image, [vertices], contourIdx=0, color=(0, 0, 0), thickness=-1)

    else:
        raise ValueError(f"Unsupported shape: {shape}. Choose from 'circle', 'square', or 'triangle'.")

    return image


def apply_grabcut(image: np.ndarray, rect: tuple = (80, 80, 96, 96)) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies the GrabCut segmentation algorithm to an image using a rectangular initialization.

    Parameters:
        image (np.ndarray): Input image in BGR format.
        rect (tuple): Bounding box (x, y, width, height) to initialize the segmentation.

    Returns:
        tuple:
            - segmented_image (np.ndarray): Image with the segmented foreground preserved, background whitened.
            - binary_mask (np.ndarray): Binary mask indicating foreground (1) and background (0).
    """
    height, width = image.shape[:2]

    # Mask: 0=bg, 1=fg, 2=probable bg, 3=probable fg
    mask = np.zeros((height, width), dtype=np.uint8)

    # Models used internally by GrabCut to represent the GMM distributions
    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)

    # Grabcut with the given rectangle as initialization
    cv2.grabCut(image, mask, rect, bg_model, fg_model, iterCount=5, mode=cv2.GC_INIT_WITH_RECT)

    # Binary mask considering: foreground = 1, background = 0
    binary_mask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype(np.uint8)

    segmented_image = image.copy()
    background_color = (255, 255, 255)

    segmented_image[binary_mask == 0] = background_color

    return segmented_image, binary_mask


def plot_results(originals: list[np.ndarray], masks: list[np.ndarray], titles: list[str]) -> None:
    """
    Displays the original images and their corresponding segmentation masks side by side.

    Parameters:
        originals (list[np.ndarray]): List of original BGR images.
        masks (list[np.ndarray]): List of binary masks for segmentation.
        titles (list[str]): List of shape names corresponding to each image.
    """
    num_images = len(originals)
    plt.figure(figsize=(12, 4))

    for i in range(num_images):
        plt.subplot(2, num_images, i + 1)
        rgb_image = cv2.cvtColor(originals[i], cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_image)
        plt.title(f"{titles[i].capitalize()} Image")
        plt.axis('off')
        plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(masks[i], cmap='gray', vmin=0, vmax=1)
        plt.title("Segmentation Mask")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def main() -> None:
    """
    Main function to run the geometry-based GrabCut segmentation experiment.
    Generates synthetic images, applies GrabCut, and visualizes the results.
    """
    geometric_shapes = ['circle', 'square', 'triangle']
    original_images = []
    binary_masks = []

    # Goals are (for this run):
    for shape in geometric_shapes:
        # Step 1: Create the synthetic image with a geometric shape
        image = create_synthetic_image(shape)

        # Step 2: Segment the object using GrabCut
        segmented_image, mask = apply_grabcut(image)

        # Store results for visualization
        original_images.append(image)
        binary_masks.append(mask)

    # Step 3: Visualize the segmentation results
    plot_results(original_images, binary_masks, geometric_shapes)


if __name__ == "__main__":
    main()
