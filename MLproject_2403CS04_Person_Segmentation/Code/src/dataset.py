"""PyTorch Dataset and DataLoaders for Penn-Fudan."""

from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from src.preprocessing import IMAGE_SIZE, load_pair, normalise
from src.splits import load_splits


class PennFudanDataset(Dataset):

    def __init__(self, root, names, size=IMAGE_SIZE, transform=None):

        root = Path(root)

        self.names = list(names)
        self.transform = transform

        self.images = []
        self.masks = []

        for n in self.names:

            img, mask = load_pair(
                root / "PNGImages" / f"{n}.png",
                root / "PedMasks" / f"{n}_mask.png",
                size
            )

            self.images.append(img)
            self.masks.append(mask)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):

        img = self.images[idx].copy()
        mask = self.masks[idx].copy()

        if self.transform is not None:
            img, mask = self.transform(img, mask)

        x = torch.from_numpy(normalise(img))

        y = torch.from_numpy(
            mask.astype("float32")
        ).unsqueeze(0)

        return x, y


def get_datasets(config_path="configs/config.yaml", train_transform=None):

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    splits = load_splits(cfg["data"]["splits_file"])

    root = cfg["data"]["root"]
    size = cfg["image_size"]

    return (
        PennFudanDataset(root, splits["train"], size, train_transform),
        PennFudanDataset(root, splits["val"], size),
        PennFudanDataset(root, splits["test"], size)
    )


def get_dataloaders(config_path="configs/config.yaml", train_transform=None):

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    train_ds, val_ds, test_ds = get_datasets(
        config_path, train_transform
    )

    bs = cfg["loader"]["batch_size"]
    nw = cfg["loader"]["num_workers"]

    generator = torch.Generator().manual_seed(cfg["seed"])

    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True,
                   num_workers=nw, generator=generator),

        DataLoader(val_ds, batch_size=bs, shuffle=False,
                   num_workers=nw),

        DataLoader(test_ds, batch_size=bs, shuffle=False,
                   num_workers=nw)
    )
