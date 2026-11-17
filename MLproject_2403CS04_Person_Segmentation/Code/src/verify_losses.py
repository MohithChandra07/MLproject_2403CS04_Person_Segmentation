"""Phase 4: Verify loss functions, evaluation metrics, gradients, and validation batch integration."""

import yaml
import torch
import torch.nn as nn

from src.dataset import get_dataloaders
from src.losses import BCEDiceLoss, BCELoss, DiceLoss, build_loss
from src.metrics import (
    MetricTracker,
    calculate_global_metrics,
    calculate_per_image_metrics,
    compute_metrics_from_counts,
    compute_pixel_counts,
)
from src.model import build_model
from src.utils import get_device, set_seed


def test_manual_metrics():
    """1. Test metric calculations against manually calculated expected results."""
    print("--- Test 1: Manual Calculation Check ---")

    # Manually constructed scenario:
    # Target:      [1, 1, 0, 0, 1]
    # Prediction:  [1, 0, 0, 1, 1]
    #
    # Pixel 0: T=1, P=1 -> TP
    # Pixel 1: T=1, P=0 -> FN
    # Pixel 2: T=0, P=0 -> TN
    # Pixel 3: T=0, P=1 -> FP
    # Pixel 4: T=1, P=1 -> TP
    #
    # Expected counts: TP=2, FP=1, FN=1, TN=1
    # Expected metrics:
    #   Precision = 2 / (2 + 1) = 2/3 = 0.666667
    #   Recall    = 2 / (2 + 1) = 2/3 = 0.666667
    #   F1 Score  = 2*2 / (2*2 + 1 + 1) = 4/6 = 0.666667
    #   IoU       = 2 / (2 + 1 + 1) = 2/4 = 0.500000

    targets = torch.tensor([1, 1, 0, 0, 1], dtype=torch.float32)
    preds = torch.tensor([1, 0, 0, 1, 1], dtype=torch.int64)

    tp, fp, fn, tn = compute_pixel_counts(preds, targets)
    assert (tp, fp, fn, tn) == (2, 1, 1, 1), f"Unexpected counts: {(tp, fp, fn, tn)}"

    metrics = compute_metrics_from_counts(tp, fp, fn, tn)
    assert abs(metrics["precision"] - (2 / 3)) < 1e-5, f"Precision mismatch: {metrics['precision']}"
    assert abs(metrics["recall"] - (2 / 3)) < 1e-5, f"Recall mismatch: {metrics['recall']}"
    assert abs(metrics["f1"] - (2 / 3)) < 1e-5, f"F1 mismatch: {metrics['f1']}"
    assert abs(metrics["iou"] - 0.5) < 1e-5, f"IoU mismatch: {metrics['iou']}"

    print(f"  Manual counts : TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"  Precision     : {metrics['precision']:.4f} (expected {2/3:.4f})")
    print(f"  Recall        : {metrics['recall']:.4f} (expected {2/3:.4f})")
    print(f"  F1 Score      : {metrics['f1']:.4f} (expected {2/3:.4f})")
    print(f"  IoU           : {metrics['iou']:.4f} (expected 0.5000)")
    print("  PASS: Manual metric verification matches exactly.\n")


def test_perfect_predictions():
    """2. Test perfect predictions (loss ~ 0, metrics = 1.0)."""
    print("--- Test 2: Perfect Predictions ---")

    targets = torch.tensor([
        [[1.0, 1.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.0]]  # All-background sample
    ]).unsqueeze(1)  # Shape (2, 1, 2, 2)

    # Perfect raw logits: high positive for 1, high negative for 0
    logits = torch.where(targets == 1.0, 15.0, -15.0)

    bce_fn = BCELoss()
    dice_fn = DiceLoss(smooth=1.0)
    comb_fn = BCEDiceLoss(bce_weight=1.0, dice_weight=1.0, smooth=1.0)

    bce = bce_fn(logits, targets).item()
    dice = dice_fn(logits, targets).item()
    comb = comb_fn(logits, targets).item()

    assert bce < 1e-4, f"BCE should be ~0, got {bce}"
    assert dice < 1e-4, f"Dice loss should be ~0, got {dice}"
    assert comb < 2e-4, f"Combined loss should be ~0, got {comb}"

    m_global = calculate_global_metrics(logits, targets, threshold=0.5)
    assert m_global["precision"] == 1.0, f"Expected precision 1.0, got {m_global['precision']}"
    assert m_global["recall"] == 1.0, f"Expected recall 1.0, got {m_global['recall']}"
    assert m_global["f1"] == 1.0, f"Expected f1 1.0, got {m_global['f1']}"
    assert m_global["iou"] == 1.0, f"Expected iou 1.0, got {m_global['iou']}"

    print(f"  BCE Loss      : {bce:.6f} (~0)")
    print(f"  Dice Loss     : {dice:.6f} (~0)")
    print(f"  Combined Loss : {comb:.6f} (~0)")
    print(f"  Global IoU    : {m_global['iou']:.4f} (=1.0)")
    print(f"  Global F1     : {m_global['f1']:.4f} (=1.0)")
    print("  PASS: Perfect prediction verification passed.\n")


def test_all_background_predictions():
    """3. Test all-background predictions and zero denominator handling."""
    print("--- Test 3: All-Background Predictions & Zero Denominators ---")

    # Case A: Target has persons, model predicts all-background (logits = -15.0)
    targets_with_fg = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])  # 2 positive pixels
    all_bg_logits = torch.full_like(targets_with_fg, -15.0)

    mA = calculate_global_metrics(all_bg_logits, targets_with_fg, threshold=0.5)
    assert mA["tp"] == 0
    assert mA["fp"] == 0
    assert mA["fn"] == 2
    assert mA["precision"] == 0.0, "Precision should be 0.0 when model predicts no positives"
    assert mA["recall"] == 0.0
    assert mA["f1"] == 0.0
    assert mA["iou"] == 0.0

    comb_fn = BCEDiceLoss(1.0, 1.0, smooth=1.0)
    lossA = comb_fn(all_bg_logits, targets_with_fg).item()
    assert lossA > 0.5, f"Loss should be substantial on complete miss, got {lossA}"

    print("  Case A (Target has persons, Pred is all-background):")
    print(f"    TP={mA['tp']}, FP={mA['fp']}, FN={mA['fn']}, TN={mA['tn']}")
    print(f"    Precision={mA['precision']:.1f}, Recall={mA['recall']:.1f}, F1={mA['f1']:.1f}, IoU={mA['iou']:.1f}")
    print(f"    Loss={lossA:.4f}")

    # Case B: Target is also all-background, model predicts all-background
    targets_pure_bg = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    logits_pure_bg = torch.full_like(targets_pure_bg, -15.0)

    mB = calculate_global_metrics(logits_pure_bg, targets_pure_bg, threshold=0.5)
    assert mB["tp"] == 0 and mB["fp"] == 0 and mB["fn"] == 0 and mB["tn"] == 16
    assert mB["precision"] == 1.0, "All-bg target + all-bg pred should yield precision 1.0"
    assert mB["recall"] == 1.0
    assert mB["f1"] == 1.0
    assert mB["iou"] == 1.0

    lossB = comb_fn(logits_pure_bg, targets_pure_bg).item()
    assert lossB < 1e-3, f"Loss should be near 0 for all-bg match, got {lossB}"

    print("  Case B (Both target & pred are all-background):")
    print(f"    TP={mB['tp']}, FP={mB['fp']}, FN={mB['fn']}, TN={mB['tn']}")
    print(f"    Precision={mB['precision']:.1f}, Recall={mB['recall']:.1f}, F1={mB['f1']:.1f}, IoU={mB['iou']:.1f}")
    print(f"    Loss={lossB:.6f}")
    print("  PASS: Zero denominator handling verified.\n")


def test_random_predictions():
    """4. Test random predictions."""
    print("--- Test 4: Random Predictions ---")

    torch.manual_seed(42)
    logits = torch.randn(4, 1, 64, 64)
    targets = torch.randint(0, 2, (4, 1, 64, 64)).float()

    comb_fn = BCEDiceLoss(1.0, 1.0, smooth=1.0)
    loss, bce, dice = comb_fn(logits, targets, return_components=True)

    assert not torch.isnan(loss) and not torch.isinf(loss)
    assert not torch.isnan(bce) and not torch.isnan(dice)
    assert 0.0 < loss.item() < 10.0

    metrics = calculate_global_metrics(logits, targets, threshold=0.5)
    for k in ["precision", "recall", "f1", "iou"]:
        assert 0.0 <= metrics[k] <= 1.0, f"{k} out of range: {metrics[k]}"

    print(f"  Random BCE Loss  : {bce.item():.4f}")
    print(f"  Random Dice Loss : {dice.item():.4f}")
    print(f"  Combined Loss    : {loss.item():.4f}")
    print(f"  Random IoU       : {metrics['iou']:.4f}")
    print(f"  Random F1        : {metrics['f1']:.4f}")
    print("  PASS: Random prediction losses and metrics are bounded and finite.\n")


def test_loss_gradients():
    """5. Test loss values and gradients."""
    print("--- Test 5: Loss Values and Gradient Flow ---")

    torch.manual_seed(42)
    logits = torch.randn(2, 1, 32, 32, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 32, 32)).float()

    criterion = BCEDiceLoss(bce_weight=1.0, dice_weight=1.0, smooth=1.0)
    loss = criterion(logits, targets)

    loss.backward()

    assert logits.grad is not None, "Gradient was not computed!"
    assert not torch.isnan(logits.grad).any(), "NaN found in gradients!"
    assert not torch.isinf(logits.grad).any(), "Inf found in gradients!"
    assert logits.grad.abs().sum().item() > 0.0, "Gradients are all zero!"

    print(f"  Computed loss    : {loss.item():.4f}")
    print(f"  Gradient norm    : {logits.grad.norm().item():.4f}")
    print(f"  Gradient min/max : [{logits.grad.min().item():.4f}, {logits.grad.max().item():.4f}]")
    print("  PASS: Gradients flow cleanly without NaNs or Infs.\n")


def test_global_vs_per_image_aggregation():
    """6. Test global aggregation vs per-image metrics."""
    print("--- Test 6: Global Aggregation vs Per-Image Metrics ---")

    # Image 0: 4 pixels, perfect match (2 TP, 2 TN) -> IoU = 1.0
    # Image 1: 4 pixels, complete miss (2 FN, 2 FP)  -> IoU = 0.0
    targets = torch.tensor([
        [[[1.0, 1.0], [0.0, 0.0]]],
        [[[1.0, 1.0], [0.0, 0.0]]]
    ])
    logits = torch.tensor([
        [[[10.0, 10.0], [-10.0, -10.0]]],  # Matches targets
        [[[-10.0, -10.0], [10.0, 10.0]]]   # Inverts targets
    ])

    per_img, mean_m = calculate_per_image_metrics(logits, targets, threshold=0.5)
    assert len(per_img) == 2
    assert per_img[0]["iou"] == 1.0, f"Image 0 IoU should be 1.0, got {per_img[0]['iou']}"
    assert per_img[1]["iou"] == 0.0, f"Image 1 IoU should be 0.0, got {per_img[1]['iou']}"
    assert mean_m["iou"] == 0.5, f"Mean IoU should be 0.5, got {mean_m['iou']}"

    global_m = calculate_global_metrics(logits, targets, threshold=0.5)
    # Total TP=2, FP=2, FN=2, TN=2 -> Global IoU = 2 / (2 + 2 + 2) = 2/6 = 0.3333
    expected_global_iou = 2.0 / 6.0
    assert abs(global_m["iou"] - expected_global_iou) < 1e-5, f"Global IoU mismatch: {global_m['iou']}"

    # Test MetricTracker
    tracker = MetricTracker(threshold=0.5)
    tracker.update(logits, targets)
    summary = tracker.compute()

    assert abs(summary["mean_iou"] - 0.5) < 1e-5
    assert abs(summary["global_iou"] - expected_global_iou) < 1e-5
    assert summary["num_samples"] == 2

    print(f"  Per-image IoU: Img 0 = {per_img[0]['iou']:.2f}, Img 1 = {per_img[1]['iou']:.2f}")
    print(f"  Mean per-image IoU : {mean_m['iou']:.4f} (expected 0.5000)")
    print(f"  Global pooled IoU  : {global_m['iou']:.4f} (expected {expected_global_iou:.4f})")
    print(f"  MetricTracker check: mean_iou={summary['mean_iou']:.4f}, global_iou={summary['global_iou']:.4f}")
    print("  PASS: Global and per-image aggregations verified.\n")


def test_real_validation_batch():
    """7. Integration test on one real validation batch (no test set!)."""
    print("--- Test 7: Integration with Real Validation Batch ---")

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = get_device()
    print("  Using device:", device)

    # Note: Only load val_dl. Strictly DO NOT touch test_dl.
    _, val_dl, _ = get_dataloaders("configs/config.yaml")
    images, masks = next(iter(val_dl))

    print(f"  Batch size       : {images.size(0)}")
    print(f"  Images tensor    : {tuple(images.shape)}, dtype={images.dtype}")
    print(f"  Masks tensor     : {tuple(masks.shape)}, dtype={masks.dtype}")
    print(f"  Foreground ratio : {masks.mean().item():.3f}")

    images = images.to(device)
    masks = masks.to(device)

    # Build model & loss
    model = build_model(cfg).to(device)
    criterion = build_loss(cfg)
    threshold = cfg.get("metrics", {}).get("threshold", 0.5)

    model.eval()
    with torch.no_grad():
        logits = model(images)

    assert logits.shape == masks.shape, f"Shape mismatch: {logits.shape} vs {masks.shape}"

    # Compute loss components
    total_loss, bce_l, dice_l = criterion(logits, masks, return_components=True)

    # Compute metrics
    global_metrics = calculate_global_metrics(logits, masks, threshold=threshold)
    per_img_records, mean_metrics = calculate_per_image_metrics(logits, masks, threshold=threshold)

    print(f"\n  [Untrained U-Net on Validation Batch]")
    print(f"  BCE Loss         : {bce_l.item():.4f}")
    print(f"  Dice Loss        : {dice_l.item():.4f}")
    print(f"  Total Loss       : {total_loss.item():.4f}")
    print(f"  Global IoU       : {global_metrics['iou']:.4f}")
    print(f"  Global F1        : {global_metrics['f1']:.4f}")
    print(f"  Global Precision : {global_metrics['precision']:.4f}")
    print(f"  Global Recall    : {global_metrics['recall']:.4f}")
    print(f"  Mean per-img IoU : {mean_metrics['iou']:.4f}")
    print(f"  Mean per-img F1  : {mean_metrics['f1']:.4f}")

    # End-to-end model gradient check
    model.train()
    logits_train = model(images)
    loss_train = criterion(logits_train, masks)
    loss_train.backward()

    first_param = next(model.parameters())
    assert first_param.grad is not None, "Model parameters did not receive gradients!"
    assert not torch.isnan(first_param.grad).any(), "NaN detected in model gradients!"
    print(f"  Model grad check : Verified ({first_param.grad.norm().item():.4f} norm on first conv layer)")
    print("  PASS: Real validation batch integration completed successfully!\n")


if __name__ == "__main__":
    print("==================================================")
    print("PHASE 4: LOSS FUNCTIONS & METRICS VERIFICATION")
    print("==================================================\n")

    test_manual_metrics()
    test_perfect_predictions()
    test_all_background_predictions()
    test_random_predictions()
    test_loss_gradients()
    test_global_vs_per_image_aggregation()
    test_real_validation_batch()

    print("==================================================")
    print("ALL PHASE 4 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("==================================================")

