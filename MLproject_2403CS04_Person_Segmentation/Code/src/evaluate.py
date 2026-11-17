"""Phase 10: Final Test Set Evaluation for Penn-Fudan Binary Person Segmentation.

Evaluates the selected best model checkpoint (checkpoints/loss_15_05_best.pt)
exactly once on the untouched test split. Computes global and per-image metrics,
saves JSON/CSV results, builds the cross-phase comparison table, and exports
visual prediction overlays.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from src.dataset import PennFudanDataset
from src.losses import build_loss
from src.metrics import MetricTracker, calculate_per_image_metrics
from src.model import build_model, count_parameters
from src.preprocessing import denormalise
from src.splits import load_splits
from src.utils import get_device, set_seed


def verify_checkpoint_and_config(
    ckpt_path: str = "checkpoints/loss_15_05_best.pt",
    config_path: str = "experiments/loss_15_05/config_used.yaml",
) -> tuple:
    """Verify that the checkpoint matches the selected best configuration."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    with open(config_path, "r") as f:
        cfg_used = yaml.safe_load(f)

    ckpt_cfg = ckpt["config"]

    assert ckpt_cfg["model"]["base_channels"] == 32, "Expected base_channels == 32"
    assert float(ckpt_cfg["train"]["lr"]) == 0.0005, "Expected lr == 0.0005"
    assert float(ckpt_cfg["loss"]["bce_weight"]) == 1.5, "Expected bce_weight == 1.5"
    assert float(ckpt_cfg["loss"]["dice_weight"]) == 0.5, "Expected dice_weight == 0.5"
    assert int(ckpt_cfg["image_size"]) == 256, "Expected image_size == 256"
    assert cfg_used.get("augmentation") == "flip_zoom", "Expected augmentation == flip_zoom"

    print("==================================================")
    print("CHECKPOINT & CONFIGURATION VERIFIED")
    print("==================================================")
    print(f"  Checkpoint      : {ckpt_path}")
    print(f"  Best Val Epoch  : {ckpt['epoch']}")
    print(f"  Saved Val IoU   : {ckpt['val_iou']:.4f}")
    print(f"  Saved Val F1    : {ckpt['val_f1']:.4f}")
    print(f"  Saved Val Loss  : {ckpt['val_loss']:.4f}")
    print(f"  base_channels   : {ckpt_cfg['model']['base_channels']}")
    print(f"  Learning Rate   : {ckpt_cfg['train']['lr']}")
    print(f"  Loss Weights    : BCE={ckpt_cfg['loss']['bce_weight']}, Dice={ckpt_cfg['loss']['dice_weight']}")
    print(f"  Augmentation    : {cfg_used['augmentation']}")
    print(f"  Resolution      : {ckpt_cfg['image_size']}x{ckpt_cfg['image_size']}")
    print("==================================================\n")

    return ckpt, cfg_used


def create_error_overlay(img_hwc: np.ndarray, gt_hw: np.ndarray, pred_hw: np.ndarray) -> np.ndarray:
    """Create RGB overlay distinguishing TP (Green), FP (Red), FN (Blue)."""
    overlay = img_hwc.copy()
    alpha = 0.45

    tp = (pred_hw == 1) & (gt_hw == 1)
    fp = (pred_hw == 1) & (gt_hw == 0)
    fn = (pred_hw == 0) & (gt_hw == 1)

    color_mask = np.zeros_like(overlay)
    color_mask[tp] = [0.0, 0.9, 0.2]   # Green: True Positive
    color_mask[fp] = [1.0, 0.15, 0.15] # Red: False Positive
    color_mask[fn] = [0.15, 0.45, 1.0] # Blue: False Negative

    active = tp | fp | fn
    overlay[active] = (1 - alpha) * overlay[active] + alpha * color_mask[active]
    return np.clip(overlay, 0, 1)


def save_sample_prediction_plot(
    img_hwc: np.ndarray,
    gt_hw: np.ndarray,
    pred_hw: np.ndarray,
    name: str,
    iou: float,
    f1: float,
    precision: float,
    recall: float,
    save_path: Path,
):
    """Save 4-panel visualization for a single test image."""
    overlay = create_error_overlay(img_hwc, gt_hw, pred_hw)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.3))

    axes[0].imshow(img_hwc)
    axes[0].set_title(f"Original Image\n({name})", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(gt_hw, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Ground-Truth Mask\n(FG: {gt_hw.mean()*100:.1f}%)", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(pred_hw, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Predicted Mask\n(FG: {pred_hw.mean()*100:.1f}%)", fontsize=10)
    axes[2].axis("off")

    axes[3].imshow(overlay)
    axes[3].set_title(
        f"Overlay (TP/FP/FN)\nIoU: {iou:.3f} | F1: {f1:.3f}",
        fontsize=10,
    )
    axes[3].axis("off")

    # Add legend for overlay
    tp_patch = mpatches.Patch(color=(0.0, 0.9, 0.2), label="TP (Correct Person)")
    fp_patch = mpatches.Patch(color=(1.0, 0.15, 0.15), label="FP (False Alarm)")
    fn_patch = mpatches.Patch(color=(0.15, 0.45, 1.0), label="FN (Missed Person)")
    axes[3].legend(
        handles=[tp_patch, fp_patch, fn_patch],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        fontsize=7.5,
        frameon=True,
    )

    fig.suptitle(
        f"Test Sample: {name} | Precision: {precision:.3f} | Recall: {recall:.3f}",
        fontsize=11,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()


def save_grid_plot(records: List[dict], title: str, save_path: Path):
    """Save a multi-row grid visualization for 4 test samples."""
    n = len(records)
    fig, axes = plt.subplots(n, 4, figsize=(15, 3.6 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = ["Original Image", "Ground-Truth Mask", "Predicted Mask", "Overlay (Green=TP, Red=FP, Blue=FN)"]

    for r, rec in enumerate(records):
        overlay = create_error_overlay(rec["img"], rec["gt"], rec["pred"])

        axes[r, 0].imshow(rec["img"])
        axes[r, 0].set_ylabel(f"{rec['name']}\nIoU={rec['iou']:.3f}", fontsize=9, fontweight="bold")
        axes[r, 1].imshow(rec["gt"], cmap="gray", vmin=0, vmax=1)
        axes[r, 2].imshow(rec["pred"], cmap="gray", vmin=0, vmax=1)
        axes[r, 3].imshow(overlay)

        for c in range(4):
            if r == 0:
                axes[r, c].set_title(col_titles[c], fontsize=10, fontweight="bold")
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()


def build_final_comparison_table(test_metrics: dict):
    """Compile validation results across all project phases alongside final test metrics."""
    experiments = [
        ("Phase 5: Baseline (No Aug)", "baseline", "none", 32, 0.001, "1.0 / 1.0"),
        ("Phase 6: Augmented", "augmented", "flip_zoom", 32, 0.001, "1.0 / 1.0"),
        ("Phase 7: LR Opt (Best LR)", "lr_0005", "flip_zoom", 32, 0.0005, "1.0 / 1.0"),
        ("Phase 7: LR Opt (Low LR)", "lr_0001", "flip_zoom", 32, 0.0001, "1.0 / 1.0"),
        ("Phase 8: Loss Opt (0.5/1.5)", "loss_05_15", "flip_zoom", 32, 0.0005, "0.5 / 1.5"),
        ("Phase 8: Loss Opt (1.5/0.5) [SELECTED]", "loss_15_05", "flip_zoom", 32, 0.0005, "1.5 / 0.5"),
        ("Phase 9: Capacity (16 ch)", "architecture_16", "flip_zoom", 16, 0.0005, "1.5 / 0.5"),
        ("Phase 9: Capacity (48 ch)", "architecture_48", "flip_zoom", 48, 0.0005, "1.5 / 0.5"),
    ]

    rows = []
    for label, exp_name, aug, ch, lr, lw in experiments:
        log_path = Path(f"experiments/{exp_name}/log.csv")
        ckpt_path = Path(f"checkpoints/{exp_name}_best.pt")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        best_epoch = ckpt["epoch"]

        with open(log_path, "r") as f:
            reader = list(csv.DictReader(f))
        best_row = [r for r in reader if int(r["epoch"]) == best_epoch][0]
        total_time = sum(float(r["epoch_time_sec"]) for r in reader)

        rows.append({
            "stage": label,
            "split_evaluated": "Validation (25 imgs)",
            "augmentation": aug,
            "base_channels": ch,
            "lr": lr,
            "loss_weights_bce_dice": lw,
            "best_epoch": best_epoch,
            "iou": round(float(best_row["val_iou"]), 4),
            "f1": round(float(best_row["val_f1"]), 4),
            "precision": round(float(best_row["val_precision"]), 4),
            "recall": round(float(best_row["val_recall"]), 4),
            "loss": round(float(best_row["val_loss"]), 4),
            "duration_min": round(total_time / 60.0, 2),
        })

    # Add Final Test Evaluation row separately
    rows.append({
        "stage": "Phase 10: Final Selected Model (TEST SET)",
        "split_evaluated": f"Test ({test_metrics['num_test_images']} imgs)",
        "augmentation": "flip_zoom (train only)",
        "base_channels": 32,
        "lr": 0.0005,
        "loss_weights_bce_dice": "1.5 / 0.5",
        "best_epoch": 57,
        "iou": round(test_metrics["global_iou"], 4),
        "f1": round(test_metrics["global_f1"], 4),
        "precision": round(test_metrics["global_precision"], 4),
        "recall": round(test_metrics["global_recall"], 4),
        "loss": round(test_metrics["test_combined_loss"], 4),
        "duration_min": "-",
    })

    for out_csv in [
        Path("results/metrics/final_comparison_table.csv"),
        Path("experiments/final_comparison.csv"),
    ]:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print("Saved cross-phase comparison table to results/metrics/final_comparison_table.csv")


def run_final_test_evaluation():
    """Execute one-time evaluation on the untouched test split."""
    ckpt, cfg = verify_checkpoint_and_config()

    set_seed(cfg["seed"])
    device = get_device()
    print(f"Evaluation Device: {device}")

    # Load splits and verify test split isolation
    splits = load_splits(cfg["data"]["splits_file"])
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_names = splits["test"]
    test_set = set(test_names)

    assert not (test_set & train_set), "Test/Train overlap detected!"
    assert not (test_set & val_set), "Test/Val overlap detected!"
    print(f"Loaded untouched test split: {len(test_names)} images (zero overlap with train/val).")

    # Build test dataset & loader (NO augmentation)
    test_ds = PennFudanDataset(
        root=cfg["data"]["root"],
        names=test_names,
        size=cfg["image_size"],
        transform=None,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["loader"]["batch_size"],
        shuffle=False,
        num_workers=cfg["loader"]["num_workers"],
    )

    # Load model weights
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded U-Net ({count_parameters(model):,} parameters) weights from epoch {ckpt['epoch']}.")

    # Build loss criterion (BCE 1.5, Dice 0.5)
    criterion = build_loss(cfg)
    threshold = float(cfg.get("metrics", {}).get("threshold", 0.5))

    tracker = MetricTracker(threshold=threshold)

    total_samples = 0
    total_comb_loss = 0.0
    total_bce_loss = 0.0
    total_dice_loss = 0.0

    per_image_details = []
    idx_cursor = 0

    with torch.no_grad():
        for images, masks in test_loader:
            bs = images.size(0)
            images_dev = images.to(device)
            masks_dev = masks.to(device)

            logits = model(images_dev)
            comb_l, bce_l, dice_l = criterion(logits, masks_dev, return_components=True)

            # Sample-weighted loss accumulation
            total_comb_loss += comb_l.item() * bs
            total_bce_loss += bce_l.item() * bs
            total_dice_loss += dice_l.item() * bs
            total_samples += bs

            # Update global tracker
            tracker.update(logits, masks_dev)

            # Per-image analysis and visualization payload
            per_img_list, _ = calculate_per_image_metrics(logits, masks_dev, threshold=threshold)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds_bin = (probs >= threshold).astype(np.uint8)

            for i in range(bs):
                img_name = test_ds.names[idx_cursor]
                img_hwc = denormalise(images[i].numpy())
                gt_hw = masks[i, 0].numpy().astype(np.uint8)
                pred_hw = preds_bin[i, 0]
                m = per_img_list[i]

                per_image_details.append({
                    "name": img_name,
                    "iou": m["iou"],
                    "f1": m["f1"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "tp": m["tp"],
                    "fp": m["fp"],
                    "fn": m["fn"],
                    "tn": m["tn"],
                    "gt_fg_fraction": float(gt_hw.mean()),
                    "pred_fg_fraction": float(pred_hw.mean()),
                    "img": img_hwc,
                    "gt": gt_hw,
                    "pred": pred_hw,
                })
                idx_cursor += 1

    assert total_samples == len(test_ds) == len(test_names), "Did not evaluate entire test split!"

    summary = tracker.compute()

    avg_comb_loss = total_comb_loss / total_samples
    avg_bce_loss = total_bce_loss / total_samples
    avg_dice_loss = total_dice_loss / total_samples

    ious = [r["iou"] for r in per_image_details]
    f1s = [r["f1"] for r in per_image_details]

    final_metrics = {
        "checkpoint": "checkpoints/loss_15_05_best.pt",
        "best_val_epoch": int(ckpt["epoch"]),
        "num_test_images": total_samples,
        "threshold": threshold,
        "global_iou": round(summary["global_iou"], 6),
        "global_f1": round(summary["global_f1"], 6),
        "global_precision": round(summary["global_precision"], 6),
        "global_recall": round(summary["global_recall"], 6),
        "mean_per_image_iou": round(summary["mean_iou"], 6),
        "mean_per_image_f1": round(summary["mean_f1"], 6),
        "mean_per_image_precision": round(summary["mean_precision"], 6),
        "mean_per_image_recall": round(summary["mean_recall"], 6),
        "std_per_image_iou": round(float(np.std(ious)), 6),
        "min_per_image_iou": round(float(np.min(ious)), 6),
        "median_per_image_iou": round(float(np.median(ious)), 6),
        "max_per_image_iou": round(float(np.max(ious)), 6),
        "test_bce_loss": round(avg_bce_loss, 6),
        "test_dice_loss": round(avg_dice_loss, 6),
        "test_combined_loss": round(avg_comb_loss, 6),
        "total_tp": summary["tp"],
        "total_fp": summary["fp"],
        "total_fn": summary["fn"],
        "total_tn": summary["tn"],
        "val_iou_reference": round(float(ckpt["val_iou"]), 6),
        "val_f1_reference": round(float(ckpt["val_f1"]), 6),
        "val_loss_reference": round(float(ckpt["val_loss"]), 6),
    }

    # Save metrics JSON & CSV
    metrics_dir = Path("results/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    json_path = metrics_dir / "final_test_metrics.json"
    with open(json_path, "w") as f:
        json.dump(final_metrics, f, indent=2)

    csv_path = metrics_dir / "final_test_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(final_metrics.keys()))
        writer.writeheader()
        writer.writerow(final_metrics)

    # Save per-image CSV
    per_img_csv = metrics_dir / "per_image_test_metrics.csv"
    per_img_fields = [
        "name", "iou", "f1", "precision", "recall",
        "tp", "fp", "fn", "tn", "gt_fg_fraction", "pred_fg_fraction"
    ]
    sorted_details = sorted(per_image_details, key=lambda x: x["iou"], reverse=True)
    with open(per_img_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_img_fields)
        writer.writeheader()
        for r in sorted_details:
            writer.writerow({k: r[k] for k in per_img_fields})

    # Generate visual predictions (12 images: Top 4, Median 4, Worst 4)
    pred_dir = Path("results/predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(sorted_details)
    mid_start = max(0, n_total // 2 - 2)

    top_4 = sorted_details[:4]
    med_4 = sorted_details[mid_start:mid_start + 4]
    bot_4 = sorted_details[-4:]

    selected_12 = top_4 + med_4 + bot_4
    for rank_idx, rec in enumerate(selected_12, start=1):
        out_img = pred_dir / f"sample_{rank_idx:02d}_{rec['name']}_iou_{rec['iou']:.3f}.png"
        save_sample_prediction_plot(
            rec["img"],
            rec["gt"],
            rec["pred"],
            rec["name"],
            rec["iou"],
            rec["f1"],
            rec["precision"],
            rec["recall"],
            out_img,
        )

    save_grid_plot(top_4, "Top 4 Highest IoU Test Predictions", pred_dir / "grid_top_4.png")
    save_grid_plot(med_4, "Median 4 Test Predictions", pred_dir / "grid_median_4.png")
    save_grid_plot(bot_4, "Bottom 4 Lowest IoU Test Predictions", pred_dir / "grid_worst_4.png")

    # Build final comparison table
    build_final_comparison_table(final_metrics)

    print("\n==================================================")
    print("FINAL TEST EVALUATION RESULTS (27 TEST IMAGES)")
    print("==================================================")
    print(f"  Global Pixel IoU       : {final_metrics['global_iou']:.4f} (Val Ref: {final_metrics['val_iou_reference']:.4f})")
    print(f"  Global Pixel F1        : {final_metrics['global_f1']:.4f} (Val Ref: {final_metrics['val_f1_reference']:.4f})")
    print(f"  Global Precision       : {final_metrics['global_precision']:.4f}")
    print(f"  Global Recall          : {final_metrics['global_recall']:.4f}")
    print(f"  Mean Per-Image IoU     : {final_metrics['mean_per_image_iou']:.4f} ± {final_metrics['std_per_image_iou']:.4f}")
    print(f"  Mean Per-Image F1      : {final_metrics['mean_per_image_f1']:.4f}")
    print(f"  Min / Med / Max IoU    : {final_metrics['min_per_image_iou']:.4f} / {final_metrics['median_per_image_iou']:.4f} / {final_metrics['max_per_image_iou']:.4f}")
    print(f"  Test BCE Loss          : {final_metrics['test_bce_loss']:.4f}")
    print(f"  Test Dice Loss         : {final_metrics['test_dice_loss']:.4f}")
    print(f"  Test Combined Loss     : {final_metrics['test_combined_loss']:.4f} (Val Ref: {final_metrics['val_loss_reference']:.4f})")
    print("==================================================")
    print(f"Saved metrics to: {json_path} and {csv_path}")
    print(f"Saved per-image breakdown to: {per_img_csv}")
    print(f"Saved 12 individual predictions + 3 grid summaries to: {pred_dir}/")


if __name__ == "__main__":
    run_final_test_evaluation()

