import numpy as np
import cv2
import random
from skimage.segmentation import slic
from skimage.util import img_as_float

class GeometricDataset:
    def __init__(self, image_size=256):
        self.H = self.W = image_size

    def random_color(self):
        return tuple(int(random.uniform(50, 255)) for _ in range(3))

    def generate_circle(self, center, radius, color):
        img = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        mask = np.zeros((self.H, self.W), dtype=np.uint8)
        cv2.circle(img, center, radius, color, -1)
        cv2.circle(mask, center, radius, 1, -1)
        return img, mask

    def generate_polygon(self, center, num_vertices, radius, rotation, color):
        img = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        mask = np.zeros((self.H, self.W), dtype=np.uint8)

        angles = np.linspace(0, 2*np.pi, num_vertices, endpoint=False) + rotation
        points = np.vstack([
            center[0] + radius * np.cos(angles),
            center[1] + radius * np.sin(angles)
        ]).T.astype(np.int32)

        cv2.fillPoly(img, [points], color)
        cv2.fillPoly(mask, [points], 1)
        return img, mask

    def add_noise(self, img, noise_level=10):
        noise = np.random.randint(-noise_level, noise_level, img.shape, dtype=np.int16)
        noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return noisy

    def build_superpixel_graph(self, img, segments=200):
        sp = slic(img_as_float(img), n_segments=segments, compactness=10, start_label=0)
        H, W = sp.shape
        edges = set()
        for y in range(H - 1):
            for x in range(W - 1):
                a = sp[y, x]
                b1 = sp[y, x+1]
                b2 = sp[y+1, x]
                if a != b1: edges.add(tuple(sorted((a, b1))))
                if a != b2: edges.add(tuple(sorted((a, b2))))

        return sp, list(edges)

    def sample(self):
        shape_type = random.choice(["circle", "polygon"])
        center = (random.randint(60, self.W-60), random.randint(60, self.H-60))
        radius = random.randint(20, 70)
        color = self.random_color()

        if shape_type == "circle":
            img, mask = self.generate_circle(center, radius, color)
        else:
            img, mask = self.generate_polygon(center,
                                              num_vertices=random.randint(3, 8),
                                              radius=radius,
                                              rotation=random.uniform(0, np.pi),
                                              color=color)

        img = self.add_noise(img, noise_level=random.randint(3, 20))

        sp_labels, edges = self.build_superpixel_graph(img)

        return {
            "image": img,
            "mask": mask,
            "superpixels": sp_labels,
            "edges": edges,
            "params": {
                "shape": shape_type,
                "center": center,
                "radius": radius,
                "color": color
            }
        }
