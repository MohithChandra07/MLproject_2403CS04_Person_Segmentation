"""Create and load reproducible train/validation/test splits."""

import json
from pathlib import Path

import numpy as np
import yaml


def make_splits(config_path="configs/config.yaml"):

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["data"]["root"])

    names = sorted(
        p.stem for p in (root / "PNGImages").glob("*.png")
    )

    n = len(names)

    if n == 0:
        raise RuntimeError(f"No images found in {root / 'PNGImages'}")

    rng = np.random.default_rng(cfg["seed"])
    perm = rng.permutation(n)

    n_train = int(cfg["data"]["train_ratio"] * n)
    n_val = int(cfg["data"]["val_ratio"] * n)

    splits = {
        "seed": cfg["seed"],
        "train": [names[i] for i in perm[:n_train]],
        "val": [names[i] for i in perm[n_train:n_train+n_val]],
        "test": [names[i] for i in perm[n_train+n_val:]]
    }

    tr = set(splits["train"])
    va = set(splits["val"])
    te = set(splits["test"])

    assert not (tr & va)
    assert not (tr & te)
    assert not (va & te)
    assert len(tr) + len(va) + len(te) == n

    out = Path(cfg["data"]["splits_file"])
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Saved {out}: train={len(tr)} val={len(va)} test={len(te)}")

    return splits


def load_splits(path="configs/splits.json"):

    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    make_splits()
