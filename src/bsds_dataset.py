from pathlib import Path
import scipy.io as sio
from PIL import Image
import numpy as np

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "BSDS500"

class BSDSSample:
    def __init__(self, split="train"):
        self.images_dir = DATA_ROOT / "images" / split
        self.labels_dir = DATA_ROOT / "groundTruth" / split

        self.files = sorted([p.stem for p in self.images_dir.glob("*.jpg")])

    def __getitem__(self, idx):
        name = self.files[idx]

        img = Image.open(self.images_dir / f"{name}.jpg").convert("RGB")
        mat = sio.loadmat(self.labels_dir / f"{name}.mat")
        gt = mat["groundTruth"][0][0]["Boundaries"]

        return np.array(img), gt

    def __len__(self):
        return len(self.files)
