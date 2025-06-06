import os
import random
import shutil

VOC_ROOT = "VOCdevkit/VOC2012"
DEST = "data"

os.makedirs(f"{DEST}/JPEGImages", exist_ok=True)
os.makedirs(f"{DEST}/SegmentationClass", exist_ok=True)

image_ids = random.sample(os.listdir(f"{VOC_ROOT}/SegmentationClass"), 20)
for fname in image_ids:
    base = fname.replace(".png", "")
    shutil.copy(f"{VOC_ROOT}/JPEGImages/{base}.jpg", f"{DEST}/JPEGImages/")
    shutil.copy(f"{VOC_ROOT}/SegmentationClass/{base}.png", f"{DEST}/SegmentationClass/")