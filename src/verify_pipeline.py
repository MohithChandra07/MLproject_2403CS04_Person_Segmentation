"""Verify preprocessing, tensor shapes and data splits."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.dataset import get_dataloaders, get_datasets
from src.preprocessing import denormalise
from src.splits import load_splits


splits = load_splits()

sets = {
    k: set(splits[k])
    for k in ("train", "val", "test")
}

print("Split sizes:", {k: len(v) for k, v in sets.items()})

print(
    "Overlap train/val/test:",
    sets["train"] & sets["val"],
    sets["train"] & sets["test"],
    sets["val"] & sets["test"]
)

assert not (sets["train"] & sets["val"])
assert not (sets["train"] & sets["test"])
assert not (sets["val"] & sets["test"])

train_dl, val_dl, test_dl = get_dataloaders()

x, y = next(iter(train_dl))

print("Image batch:", tuple(x.shape), x.dtype,
      "| mean=%.3f std=%.3f" % (x.mean().item(), x.std().item()))

print("Mask batch:", tuple(y.shape), y.dtype,
      "| unique values:", torch.unique(y).tolist())

assert not torch.isnan(x).any()
assert not torch.isnan(y).any()
assert set(torch.unique(y).tolist()).issubset({0.0, 1.0})

fg = torch.cat([b[1] for b in test_dl]).mean().item()

print("Test foreground fraction: %.3f" % fg)

_, _, test_ds = get_datasets()

fig, axes = plt.subplots(2, 4, figsize=(14, 7))

for i in range(4):

    xi, yi = test_ds[i]

    img = denormalise(xi.numpy())

    axes[0, i].imshow(img)
    axes[0, i].set_title(test_ds.names[i], fontsize=8)
    axes[0, i].axis("off")

    axes[1, i].imshow(img)
    axes[1, i].contour(
        yi[0].numpy(),
        levels=[0.5],
        colors="red",
        linewidths=1.5
    )

    axes[1, i].set_title("Mask outline", fontsize=8)
    axes[1, i].axis("off")

Path("results/plots").mkdir(parents=True, exist_ok=True)

plt.tight_layout()
plt.savefig("results/plots/pipeline_check.png", dpi=110)
plt.close()

print("Saved results/plots/pipeline_check.png")
