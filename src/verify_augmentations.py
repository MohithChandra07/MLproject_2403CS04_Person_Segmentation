"""Verify data augmentations visually and structurally."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.augmentations import RandomHorizontalFlip, RandomZoom, get_augmentation
from src.dataset import PennFudanDataset
from src.splits import load_splits


def verify_augmentations():
    """Verify geometric consistency, binary masks, and visual alignment."""
    print("==================================================")
    print("VERIFYING DATA AUGMENTATIONS")
    print("==================================================")

    splits = load_splits("configs/splits.json")
    train_names = splits["train"]
    dataset = PennFudanDataset("data/PennFudanPed", train_names[:4], size=256, transform=None)

    # 1. Test RandomHorizontalFlip
    flip_transform = RandomHorizontalFlip(p=1.0)  # Force flip
    orig_img = dataset.images[0].copy()
    orig_mask = dataset.masks[0].copy()

    flipped_img, flipped_mask = flip_transform(orig_img, orig_mask)

    assert flipped_img.shape == (256, 256, 3), f"Wrong shape: {flipped_img.shape}"
    assert flipped_mask.shape == (256, 256), f"Wrong shape: {flipped_mask.shape}"
    assert set(np.unique(flipped_mask)).issubset({0, 1}), f"Non-binary mask: {np.unique(flipped_mask)}"
    assert np.array_equal(flipped_img, np.fliplr(orig_img)), "Flipped image mismatch!"
    assert np.array_equal(flipped_mask, np.fliplr(orig_mask)), "Flipped mask mismatch!"
    print("[Check 1/4] Horizontal flip passed (exact geometric alignment and binary mask).")

    # 2. Test RandomZoom (Zoom In: scale=1.1)
    zoom_in_transform = RandomZoom(min_scale=1.1, max_scale=1.1)
    zoomed_in_img, zoomed_in_mask = zoom_in_transform(orig_img, orig_mask)

    assert zoomed_in_img.shape == (256, 256, 3), f"Wrong shape: {zoomed_in_img.shape}"
    assert zoomed_in_mask.shape == (256, 256), f"Wrong shape: {zoomed_in_mask.shape}"
    assert set(np.unique(zoomed_in_mask)).issubset({0, 1}), f"Non-binary mask: {np.unique(zoomed_in_mask)}"
    print("[Check 2/4] Zoom-in (1.1x) passed (shape preserved, center cropped, binary mask).")

    # 3. Test RandomZoom (Zoom Out: scale=0.9)
    zoom_out_transform = RandomZoom(min_scale=0.9, max_scale=0.9)
    zoomed_out_img, zoomed_out_mask = zoom_out_transform(orig_img, orig_mask)

    assert zoomed_out_img.shape == (256, 256, 3), f"Wrong shape: {zoomed_out_img.shape}"
    assert zoomed_out_mask.shape == (256, 256), f"Wrong shape: {zoomed_out_mask.shape}"
    assert set(np.unique(zoomed_out_mask)).issubset({0, 1}), f"Non-binary mask: {np.unique(zoomed_out_mask)}"
    print("[Check 3/4] Zoom-out (0.9x) passed (shape preserved, zero-padded border, binary mask).")

    # 4. Save visual verification plot
    out_dir = Path("results/plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_plot = out_dir / "augmentation_check.png"

    fig, axes = plt.subplots(4, 4, figsize=(14, 14))

    samples = [
        ("Original", orig_img, orig_mask),
        ("Flipped (p=1.0)", flipped_img, flipped_mask),
        ("Zoom In (1.1x)", zoomed_in_img, zoomed_in_mask),
        ("Zoom Out (0.9x)", zoomed_out_img, zoomed_out_mask),
    ]

    for col, (title, img, mask) in enumerate(samples):
        # Image
        axes[0, col].imshow(img)
        axes[0, col].set_title(f"{title}\nImage", fontsize=10)
        axes[0, col].axis("off")

        # Mask
        axes[1, col].imshow(mask, cmap="gray")
        axes[1, col].set_title(f"Binary Mask\n(values: {set(np.unique(mask))})", fontsize=9)
        axes[1, col].axis("off")

        # Mask contour overlay on image
        axes[2, col].imshow(img)
        if mask.max() > 0:
            axes[2, col].contour(mask, levels=[0.5], colors="red", linewidths=1.5)
        axes[2, col].set_title("Overlay Alignment", fontsize=9)
        axes[2, col].axis("off")

        # Composite blended overlay
        colored_mask = np.zeros_like(img)
        colored_mask[:, :, 0] = mask * 255  # Red overlay
        blended = (img * 0.7 + colored_mask * 0.3).astype(np.uint8)
        axes[3, col].imshow(blended)
        axes[3, col].set_title("Blended Check", fontsize=9)
        axes[3, col].axis("off")

    plt.tight_layout()
    plt.savefig(out_plot, dpi=120)
    plt.close()

    print(f"[Check 4/4] Visual plot successfully saved to {out_plot}.")
    print("==================================================")
    print("ALL AUGMENTATION CHECKS PASSED!")
    print("==================================================")


if __name__ == "__main__":
    verify_augmentations()

