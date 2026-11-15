"""Phase 5: Baseline U-Net Training and Validation Pipeline.

Trains the custom U-Net architecture on the Penn-Fudan pedestrian dataset
using combined BCE + Dice loss and standard hyperparameters.
"""

import argparse
import csv
import os
from pathlib import Path
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import yaml

from src.augmentations import get_augmentation
from src.dataset import PennFudanDataset
from src.losses import build_loss
from src.metrics import MetricTracker
from src.model import build_model, count_parameters
from src.splits import load_splits
from src.utils import get_device, set_seed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train baseline U-Net on Penn-Fudan")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="baseline",
        help="Experiment name (used for logging and checkpoints)",
    )
    parser.add_argument(
        "--augmentation",
        type=str,
        default="none",
        choices=["none", "flip_zoom"],
        help="Data augmentation strategy ('none' or 'flip_zoom')",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate in configuration",
    )
    parser.add_argument(
        "--bce-weight",
        type=float,
        default=None,
        help="Override BCE loss weight in configuration",
    )
    parser.add_argument(
        "--dice-weight",
        type=float,
        default=None,
        help="Override Dice loss weight in configuration",
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=None,
        help="Override U-Net base_channels in configuration",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run short pre-flight checks and exit",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_dataloaders(
    cfg: dict,
    augmentation: str = "none",
) -> Tuple[DataLoader, DataLoader]:
    """Build training and validation DataLoaders.

    Guarantees that test split is NEVER loaded or accessed.
    Augmentations are only applied to the training set.
    """
    splits = load_splits(cfg["data"]["splits_file"])
    root = cfg["data"]["root"]
    size = cfg["image_size"]
    bs = cfg["loader"]["batch_size"]
    nw = cfg["loader"]["num_workers"]
    seed = cfg["seed"]

    train_transform = get_augmentation(augmentation)
    train_ds = PennFudanDataset(root, splits["train"], size=size, transform=train_transform)
    val_ds = PennFudanDataset(root, splits["val"], size=size, transform=None)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
    )

    return train_loader, val_loader


def run_preflight_checks(
    cfg: dict,
    device: torch.device,
    augmentation: str = "none",
) -> None:
    """Run verification checks before full training starts."""
    print("==================================================")
    print("RUNNING PRE-FLIGHT VERIFICATION CHECKS")
    print("==================================================")

    # 1. Verify split integrity
    splits = load_splits(cfg["data"]["splits_file"])
    train_names = set(splits["train"])
    val_names = set(splits["val"])
    test_names = set(splits["test"])

    assert len(train_names) == 118, f"Expected 118 train samples, got {len(train_names)}"
    assert len(val_names) == 25, f"Expected 25 val samples, got {len(val_names)}"
    assert not (train_names & val_names), "Data leakage between train and val!"
    assert not (train_names & test_names), "Data leakage between train and test!"
    print(f"[Check 1/5] Splits verified: train={len(train_names)}, val={len(val_names)}, zero overlap.")

    # 2. Verify DataLoaders & tensor shapes
    train_loader, val_loader = build_dataloaders(cfg, augmentation=augmentation)
    images, masks = next(iter(train_loader))

    assert images.shape == (cfg["loader"]["batch_size"], 3, cfg["image_size"], cfg["image_size"]), (
        f"Unexpected image batch shape: {images.shape}"
    )
    assert masks.shape == (cfg["loader"]["batch_size"], 1, cfg["image_size"], cfg["image_size"]), (
        f"Unexpected mask batch shape: {masks.shape}"
    )
    assert not torch.isnan(images).any() and not torch.isnan(masks).any(), "NaN in batch tensors!"
    assert set(torch.unique(masks).tolist()).issubset({0.0, 1.0}), "Masks are not binary {0, 1}!"
    print(f"[Check 2/5] Batch tensor shapes verified: images={tuple(images.shape)}, masks={tuple(masks.shape)}.")

    # 3. Verify forward pass
    model = build_model(cfg).to(device)
    images = images.to(device)
    masks = masks.to(device)

    model.eval()
    with torch.no_grad():
        logits = model(images)

    assert logits.shape == masks.shape, f"Logits shape mismatch: {logits.shape} vs {masks.shape}"
    print(f"[Check 3/5] Forward pass verified: logits shape={tuple(logits.shape)}.")

    # 4. Verify loss computation & backward pass
    criterion = build_loss(cfg)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    model.train()
    optimizer.zero_grad()
    logits_train = model(images)
    total_loss, bce_l, dice_l = criterion(logits_train, masks, return_components=True)

    assert not torch.isnan(total_loss) and not torch.isinf(total_loss), "Loss is NaN or Inf!"
    total_loss.backward()

    first_param = next(model.parameters())
    assert first_param.grad is not None, "Gradients not computed!"
    assert not torch.isnan(first_param.grad).any(), "NaN detected in gradients!"
    print(f"[Check 4/5] Backward pass verified: loss={total_loss.item():.4f}, grad_norm={first_param.grad.norm().item():.4f}.")

    # 5. Verify validation metric evaluation
    threshold = cfg.get("metrics", {}).get("threshold", 0.5)
    tracker = MetricTracker(threshold=threshold)
    tracker.update(logits, masks)
    summary = tracker.compute()

    assert 0.0 <= summary["global_iou"] <= 1.0, f"Invalid IoU: {summary['global_iou']}"
    assert 0.0 <= summary["global_f1"] <= 1.0, f"Invalid F1: {summary['global_f1']}"
    print(f"[Check 5/5] Metrics tracker verified: IoU={summary['global_iou']:.4f}, F1={summary['global_f1']:.4f}.")

    print("==================================================")
    print("ALL PRE-FLIGHT CHECKS PASSED SUCCESSFULLY!")
    print("==================================================\n")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Train the model for one epoch.

    Returns:
        avg_total_loss, avg_bce_loss, avg_dice_loss
    """
    model.train()
    total_loss_accum = 0.0
    bce_loss_accum = 0.0
    dice_loss_accum = 0.0
    num_batches = len(loader)

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)

        loss, bce_loss, dice_loss = criterion(logits, masks, return_components=True)

        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError(f"NaN or Inf loss encountered during training: loss={loss.item()}")

        loss.backward()
        optimizer.step()

        total_loss_accum += loss.item()
        bce_loss_accum += bce_loss.item()
        dice_loss_accum += dice_loss.item()

    return (
        total_loss_accum / num_batches,
        bce_loss_accum / num_batches,
        dice_loss_accum / num_batches,
    )


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    threshold: float,
    device: torch.device,
) -> Tuple[float, float, float, Dict[str, float]]:
    """Evaluate the model on the validation set.

    Returns:
        avg_total_loss, avg_bce_loss, avg_dice_loss, metrics_dict
    """
    model.eval()
    total_loss_accum = 0.0
    bce_loss_accum = 0.0
    dice_loss_accum = 0.0
    num_batches = len(loader)

    tracker = MetricTracker(threshold=threshold)

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss, bce_loss, dice_loss = criterion(logits, masks, return_components=True)

            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"NaN or Inf loss encountered during validation: loss={loss.item()}")

            total_loss_accum += loss.item()
            bce_loss_accum += bce_loss.item()
            dice_loss_accum += dice_loss.item()

            tracker.update(logits, masks)

    metrics = tracker.compute()
    return (
        total_loss_accum / num_batches,
        bce_loss_accum / num_batches,
        dice_loss_accum / num_batches,
        metrics,
    )


def train_pipeline(
    cfg: dict,
    name: str = "baseline",
    augmentation: str = "none",
) -> Dict[str, float]:
    """Execute the full training pipeline with validation, checkpointing, and logging."""
    seed = cfg["seed"]
    set_seed(seed)
    device = get_device()

    print(f"Starting Training Experiment: {name}")
    print(f"Device: {device} | Augmentation: {augmentation} | Seed: {seed}")

    # Output directories
    ckpt_dir = Path("checkpoints")
    exp_dir = Path(f"experiments/{name}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / f"{name}_best.pt"
    log_path = exp_dir / "log.csv"
    cfg_used_path = exp_dir / "config_used.yaml"

    # Save exact configuration used
    cfg_copy = dict(cfg)
    cfg_copy["experiment_name"] = name
    cfg_copy["augmentation"] = augmentation
    with open(cfg_used_path, "w") as f:
        yaml.safe_dump(cfg_copy, f, sort_keys=False)
    print(f"Saved configuration to {cfg_used_path}")

    # Build components
    train_loader, val_loader = build_dataloaders(cfg, augmentation=augmentation)
    model = build_model(cfg).to(device)
    criterion = build_loss(cfg)

    lr = float(cfg["train"]["lr"])
    weight_decay = float(cfg["train"]["weight_decay"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    factor = float(cfg["train"]["scheduler_factor"])
    scheduler_patience = int(cfg["train"]["scheduler_patience"])
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=factor,
        patience=scheduler_patience,
    )

    threshold = float(cfg.get("metrics", {}).get("threshold", 0.5))
    max_epochs = int(cfg["train"]["epochs"])
    early_stop_patience = int(cfg["train"]["early_stop_patience"])

    num_params = count_parameters(model)
    print(f"Model base_channels: {cfg['model']['base_channels']} | Parameters: {num_params:,} ({num_params / 1e6:.2f}M)")
    print(f"Loss weights: BCE={cfg['loss']['bce_weight']}, Dice={cfg['loss']['dice_weight']} | LR={lr}")
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")
    print(f"Max epochs: {max_epochs} | Early stopping patience: {early_stop_patience}\n")

    # CSV Logging initialization
    csv_headers = [
        "epoch",
        "train_loss",
        "train_bce",
        "train_dice",
        "val_loss",
        "val_bce",
        "val_dice",
        "val_precision",
        "val_recall",
        "val_f1",
        "val_iou",
        "lr",
        "epoch_time_sec",
    ]

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    best_val_iou = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    start_total_time = time.time()

    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'Val IoU':>8} | {'Val F1':>8} | {'Val Prec':>8} | {'Val Rec':>8} | {'LR':>9} | {'Time':>6}")
    print("-" * 92)

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()

        train_loss, train_bce, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_bce, val_dice, val_metrics = validate(
            model, val_loader, criterion, threshold, device
        )

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]
        val_iou = val_metrics["global_iou"]
        val_f1 = val_metrics["global_f1"]
        val_prec = val_metrics["global_precision"]
        val_rec = val_metrics["global_recall"]

        # Step LR scheduler based on validation IoU (mode='max')
        scheduler.step(val_iou)

        # Log to CSV
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{train_bce:.6f}",
                f"{train_dice:.6f}",
                f"{val_loss:.6f}",
                f"{val_bce:.6f}",
                f"{val_dice:.6f}",
                f"{val_prec:.6f}",
                f"{val_rec:.6f}",
                f"{val_f1:.6f}",
                f"{val_iou:.6f}",
                f"{current_lr:.6e}",
                f"{epoch_time:.2f}",
            ])

        # Check for best checkpoint
        is_best = val_iou > best_val_iou
        if is_best:
            best_val_iou = val_iou
            best_epoch = epoch
            epochs_no_improve = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_iou": best_val_iou,
                "val_f1": val_f1,
                "val_loss": val_loss,
                "config": cfg,
            }
            torch.save(checkpoint, ckpt_path)
            indicator = " * BEST"
        else:
            epochs_no_improve += 1
            indicator = ""

        print(
            f"{epoch:5d} | "
            f"{train_loss:10.4f} | "
            f"{val_loss:10.4f} | "
            f"{val_iou:8.4f} | "
            f"{val_f1:8.4f} | "
            f"{val_prec:8.4f} | "
            f"{val_rec:8.4f} | "
            f"{current_lr:9.1e} | "
            f"{epoch_time:5.1f}s"
            f"{indicator}"
        )

        # Early stopping check
        if epochs_no_improve >= early_stop_patience:
            print(f"\n[Early Stopping] No improvement in validation IoU for {early_stop_patience} consecutive epochs.")
            print(f"Stopping training at epoch {epoch}. Best epoch was {best_epoch} with Val IoU: {best_val_iou:.4f}.")
            break

    total_training_duration = time.time() - start_total_time

    print("==================================================")
    print("TRAINING RUN COMPLETE")
    print("==================================================")
    print(f"Best Validation IoU : {best_val_iou:.4f} (Epoch {best_epoch})")
    print(f"Total Duration      : {total_training_duration / 60:.2f} minutes ({total_training_duration:.1f}s)")
    print(f"Best Checkpoint     : {ckpt_path}")
    print(f"Training Log CSV    : {log_path}")
    print(f"Configuration File  : {cfg_used_path}")
    print("==================================================")

    return {
        "best_val_iou": best_val_iou,
        "best_epoch": best_epoch,
        "total_duration": total_training_duration,
        "checkpoint_path": str(ckpt_path),
        "log_path": str(log_path),
        "config_path": str(cfg_used_path),
    }


def main():
    """Main execution entrypoint."""
    args = parse_args()
    cfg = load_config(args.config)
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.bce_weight is not None:
        cfg["loss"]["bce_weight"] = args.bce_weight
    if args.dice_weight is not None:
        cfg["loss"]["dice_weight"] = args.dice_weight
    if args.base_channels is not None:
        cfg["model"]["base_channels"] = args.base_channels
    device = get_device()

    # Pre-flight checks before training
    run_preflight_checks(cfg, device, augmentation=args.augmentation)

    if args.smoke_test:
        print("Smoke test completed successfully. Exiting as requested by --smoke-test.")
        return

    # Execute training
    train_pipeline(
        cfg=cfg,
        name=args.name,
        augmentation=args.augmentation,
    )


if __name__ == "__main__":
    main()

