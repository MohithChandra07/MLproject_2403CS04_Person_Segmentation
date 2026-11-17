"""Data augmentation transforms for binary person segmentation.

Applies identical geometric transformations to image and mask pairs:
1. Random horizontal flip with probability 0.5.
2. Random zoom with scale in [0.9, 1.1] using Bilinear interpolation
   for images and Nearest-Neighbor interpolation for masks.
"""

import random
from typing import Callable, List, Optional, Tuple, Union
import numpy as np
from PIL import Image


class RandomHorizontalFlip:
    """Randomly flips image and mask horizontally with probability p."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self,
        img: Union[np.ndarray, Image.Image],
        mask: Union[np.ndarray, Image.Image],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply horizontal flip with probability p."""
        # Convert to numpy if needed
        is_pil_img = isinstance(img, Image.Image)
        is_pil_mask = isinstance(mask, Image.Image)

        img_np = np.asarray(img) if is_pil_img else img
        mask_np = np.asarray(mask) if is_pil_mask else mask

        if random.random() < self.p:
            img_np = np.fliplr(img_np).copy()
            mask_np = np.fliplr(mask_np).copy()

        return img_np, mask_np


class RandomZoom:
    """Randomly zooms image and mask by a scale factor in [min_scale, max_scale].

    - Uses Bilinear interpolation for image.
    - Uses Nearest-neighbor interpolation for mask.
    - Preserves binary mask values (0 background, 1 person).
    """

    def __init__(self, min_scale: float = 0.9, max_scale: float = 1.1):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(
        self,
        img: Union[np.ndarray, Image.Image],
        mask: Union[np.ndarray, Image.Image],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random zoom to image and mask pair."""
        scale = random.uniform(self.min_scale, self.max_scale)

        # Convert numpy to PIL for high quality interpolation
        if isinstance(img, np.ndarray):
            pil_img = Image.fromarray(img)
        else:
            pil_img = img

        if isinstance(mask, np.ndarray):
            pil_mask = Image.fromarray(mask.astype(np.uint8))
        else:
            pil_mask = mask

        orig_w, orig_h = pil_img.size
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))

        # Bilinear for image, Nearest-neighbor for mask
        resized_img = pil_img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
        resized_mask = pil_mask.resize((new_w, new_h), resample=Image.Resampling.NEAREST)

        if scale > 1.0:
            # Zoom in: crop center back to (orig_w, orig_h)
            left = (new_w - orig_w) // 2
            top = (new_h - orig_h) // 2
            right = left + orig_w
            bottom = top + orig_h

            out_img = resized_img.crop((left, top, right, bottom))
            out_mask = resized_mask.crop((left, top, right, bottom))
        elif scale < 1.0:
            # Zoom out: pad canvas back to (orig_w, orig_h) with zeros
            out_img = Image.new("RGB", (orig_w, orig_h), (0, 0, 0))
            out_mask = Image.new("L", (orig_w, orig_h), 0)

            left = (orig_w - new_w) // 2
            top = (orig_h - new_h) // 2

            out_img.paste(resized_img, (left, top))
            out_mask.paste(resized_mask, (left, top))
        else:
            out_img = resized_img
            out_mask = resized_mask

        out_img_np = np.asarray(out_img, dtype=np.uint8)
        out_mask_np = (np.asarray(out_mask) > 0).astype(np.uint8)

        return out_img_np, out_mask_np


class ComposeDualTransform:
    """Sequential composition of transforms that operate on (image, mask) pairs."""

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(
        self,
        img: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


def get_augmentation(name: str = "none") -> Optional[ComposeDualTransform]:
    """Factory to get augmentation pipeline by name.

    Supported options:
    - 'none': No augmentation (returns None).
    - 'flip_zoom': Random horizontal flip (p=0.5) + Random zoom (scale 0.9-1.1).
    """
    if name is None or name.lower() == "none":
        return None

    if name.lower() in ["flip_zoom", "augmented"]:
        return ComposeDualTransform([
            RandomHorizontalFlip(p=0.5),
            RandomZoom(min_scale=0.9, max_scale=1.1),
        ])

    raise ValueError(f"Unknown augmentation: '{name}'. Supported: 'none', 'flip_zoom'.")

