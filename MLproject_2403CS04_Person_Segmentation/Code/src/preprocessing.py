"""Loading, resizing and normalising image/mask pairs."""

from pathlib import Path
import numpy as np
from PIL import Image

IMAGE_SIZE = 256
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_pair(img_path: Path, mask_path: Path, size: int = IMAGE_SIZE):
    """Return image uint8 [H,W,3] and binary mask uint8 [H,W]."""

    img = Image.open(img_path).convert("RGB").resize(
        (size, size), Image.Resampling.BILINEAR
    )

    mask = Image.open(mask_path).resize(
        (size, size), Image.Resampling.NEAREST
    )

    img = np.asarray(img, dtype=np.uint8)
    mask = (np.asarray(mask) > 0).astype(np.uint8)

    return img, mask


def normalise(img_uint8: np.ndarray) -> np.ndarray:
    """Convert uint8 HWC image to normalized float32 CHW."""

    x = img_uint8.astype(np.float32) / 255.0
    x = (x - MEAN) / STD

    return x.transpose(2, 0, 1)


def denormalise(x_chw: np.ndarray) -> np.ndarray:
    """Convert normalized CHW image back to HWC."""

    x = x_chw.transpose(1, 2, 0) * STD + MEAN

    return np.clip(x, 0, 1)
