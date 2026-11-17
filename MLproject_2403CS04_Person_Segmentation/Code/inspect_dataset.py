from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path("data/PennFudanPed")
IMG_DIR = ROOT / "PNGImages"
MASK_DIR = ROOT / "PedMasks"

images = sorted(IMG_DIR.glob("*.png"))
masks = sorted(MASK_DIR.glob("*_mask.png"))

print(f"Images: {len(images)} | Masks: {len(masks)}")

missing = [
    p.name for p in images
    if not (MASK_DIR / f"{p.stem}_mask.png").exists()
]
print("Images without a mask:", missing if missing else "none")

total_instances = 0
sizes_match = True

for img_p in images:
    mask_p = MASK_DIR / f"{img_p.stem}_mask.png"

    img = Image.open(img_p)
    m = Image.open(mask_p)

    if img.size != m.size:
        sizes_match = False

    arr = np.array(m)
    total_instances += len(np.unique(arr)) - 1

if not masks:
    raise RuntimeError("No segmentation masks found. Check dataset path.")

first = np.array(Image.open(masks[0]))

print("\nFirst mask:")
print("Mode:", Image.open(masks[0]).mode)
print("Dtype:", first.dtype)
print("Shape:", first.shape)
print("Unique values:", np.unique(first))

print("Image/mask sizes all match:", sizes_match)
print("Total pedestrian instances:", total_instances)

fg_ratios = []

for mask_p in masks:
    binary = (np.array(Image.open(mask_p)) > 0).astype(np.uint8)
    fg_ratios.append(binary.mean())

fg_ratios = np.array(fg_ratios)

print(
    f"\nForeground pixel fraction: "
    f"mean={fg_ratios.mean():.3f}, "
    f"min={fg_ratios.min():.3f}, "
    f"max={fg_ratios.max():.3f}"
)

out_dir = Path("results/plots")
out_dir.mkdir(parents=True, exist_ok=True)

idx = [i for i in [0, 20, 60, 100, 140] if i < len(images)]

fig, axes = plt.subplots(len(idx), 3, figsize=(9, 3 * len(idx)))

if len(idx) == 1:
    axes = np.expand_dims(axes, axis=0)

for r, i in enumerate(idx):
    img = Image.open(images[i]).convert("RGB")
    raw = np.array(
        Image.open(MASK_DIR / f"{images[i].stem}_mask.png")
    )

    axes[r, 0].imshow(img)
    axes[r, 0].set_title(images[i].name)

    axes[r, 1].imshow(raw, cmap="nipy_spectral")
    axes[r, 1].set_title("Original instance mask")

    axes[r, 2].imshow(raw > 0, cmap="gray")
    axes[r, 2].set_title("Merged binary mask")

    for a in axes[r]:
        a.axis("off")

plt.tight_layout()
plt.savefig(out_dir / "dataset_check.png", dpi=120)
plt.close()

print("\nSaved: results/plots/dataset_check.png")
