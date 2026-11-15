"""Evaluation metrics for binary person segmentation.

Computes:
- Pixel-level confusion matrix: TP, FP, FN, TN
- Classification & segmentation metrics: Precision, Recall, F1 Score, IoU
- Safe handling of zero denominators
- Global (dataset-level) aggregation and per-image metric tracking
"""

from typing import Dict, List, Tuple, Union
import numpy as np
import torch


def compute_metrics_from_counts(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    zero_division: float = 0.0,
) -> Dict[str, Union[float, int]]:
    """Compute precision, recall, f1, and iou from confusion counts safely.

    Handles zero denominators:
    - If both ground truth and prediction have no positive pixels (tp + fp + fn == 0),
      the prediction is a perfect all-background match, so all metrics return 1.0.
    - Otherwise, if a denominator is zero, returns zero_division (default 0.0).

    Args:
        tp: True positives count.
        fp: False positives count.
        fn: False negatives count.
        tn: True negatives count.
        zero_division: Value to return when denominator is zero.

    Returns:
        Dictionary containing precision, recall, f1, iou, tp, fp, fn, tn.
    """
    tp = int(tp)
    fp = int(fp)
    fn = int(fn)
    tn = int(tn)

    # Empty target and empty prediction: perfect agreement
    if tp + fp + fn == 0:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "iou": 1.0,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    precision = tp / (tp + fp) if (tp + fp) > 0 else zero_division
    recall = tp / (tp + fn) if (tp + fn) > 0 else zero_division
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else zero_division
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else zero_division

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def compute_pixel_counts(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[int, int, int, int]:
    """Compute pixel-level TP, FP, FN, TN between binary prediction and target tensors.

    Args:
        preds: Binary predictions (0 or 1), any shape.
        targets: Binary ground truth (0 or 1), matching shape.

    Returns:
        Tuple of (tp, fp, fn, tn) as integers.
    """
    preds_bool = preds.bool()
    targets_bool = targets.bool()

    tp = torch.sum(preds_bool & targets_bool).item()
    fp = torch.sum(preds_bool & (~targets_bool)).item()
    fn = torch.sum((~preds_bool) & targets_bool).item()
    tn = torch.sum((~preds_bool) & (~targets_bool)).item()

    return tp, fp, fn, tn


def binarize_predictions(
    logits: torch.Tensor,
    threshold: float = 0.5,
    is_logits: bool = True,
) -> torch.Tensor:
    """Convert raw logits or probabilities to binary masks (0 or 1)."""
    if is_logits:
        probs = torch.sigmoid(logits)
    else:
        probs = logits
    return (probs >= threshold).to(torch.int64)


def calculate_per_image_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    is_logits: bool = True,
    zero_division: float = 0.0,
) -> Tuple[List[Dict[str, Union[float, int]]], Dict[str, float]]:
    """Compute metrics for each image individually in a batch.

    Args:
        logits: Raw logits or probabilities of shape (B, 1, H, W) or (B, H, W).
        targets: Binary targets of matching or broadcastable shape.
        threshold: Classification threshold (default 0.5).
        is_logits: Whether logits need sigmoid applied.
        zero_division: Value for zero denominators.

    Returns:
        per_image_list: List of metric dicts for each image.
        mean_metrics: Mean precision, recall, f1, iou across images.
    """
    if targets.shape != logits.shape:
        targets = targets.view_as(logits)

    preds = binarize_predictions(logits, threshold=threshold, is_logits=is_logits)
    targets_bin = (targets >= 0.5).to(torch.int64)

    # Batch dimension
    b = preds.size(0) if preds.dim() > 1 else 1
    per_image_list = []

    if preds.dim() > 1:
        for i in range(b):
            p_i = preds[i]
            t_i = targets_bin[i]
            tp, fp, fn, tn = compute_pixel_counts(p_i, t_i)
            m = compute_metrics_from_counts(tp, fp, fn, tn, zero_division=zero_division)
            per_image_list.append(m)
    else:
        tp, fp, fn, tn = compute_pixel_counts(preds, targets_bin)
        m = compute_metrics_from_counts(tp, fp, fn, tn, zero_division=zero_division)
        per_image_list.append(m)

    mean_metrics = {
        "precision": float(np.mean([m["precision"] for m in per_image_list])),
        "recall": float(np.mean([m["recall"] for m in per_image_list])),
        "f1": float(np.mean([m["f1"] for m in per_image_list])),
        "iou": float(np.mean([m["iou"] for m in per_image_list])),
    }

    return per_image_list, mean_metrics


def calculate_global_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    is_logits: bool = True,
    zero_division: float = 0.0,
) -> Dict[str, Union[float, int]]:
    """Compute globally aggregated (pooled) metrics across all pixels in the batch.

    Args:
        logits: Raw logits or probabilities of shape (B, 1, H, W) or (B, H, W).
        targets: Binary targets of matching or broadcastable shape.
        threshold: Classification threshold (default 0.5).
        is_logits: Whether logits need sigmoid applied.
        zero_division: Value for zero denominators.

    Returns:
        Dictionary with pooled precision, recall, f1, iou, and total tp, fp, fn, tn.
    """
    if targets.shape != logits.shape:
        targets = targets.view_as(logits)

    preds = binarize_predictions(logits, threshold=threshold, is_logits=is_logits)
    targets_bin = (targets >= 0.5).to(torch.int64)

    tp, fp, fn, tn = compute_pixel_counts(preds, targets_bin)
    return compute_metrics_from_counts(tp, fp, fn, tn, zero_division=zero_division)


class MetricTracker:
    """Accumulator for evaluation metrics across batches."""

    def __init__(
        self,
        threshold: float = 0.5,
        is_logits: bool = True,
        zero_division: float = 0.0,
    ):
        self.threshold = threshold
        self.is_logits = is_logits
        self.zero_division = zero_division
        self.reset()

    def reset(self):
        """Reset all accumulated counts and records."""
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_tn = 0
        self.per_image_records: List[Dict[str, Union[float, int]]] = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """Update tracker with batch predictions and targets."""
        with torch.no_grad():
            per_img_list, _ = calculate_per_image_metrics(
                logits,
                targets,
                threshold=self.threshold,
                is_logits=self.is_logits,
                zero_division=self.zero_division,
            )
            self.per_image_records.extend(per_img_list)

            for m in per_img_list:
                self.total_tp += m["tp"]
                self.total_fp += m["fp"]
                self.total_fn += m["fn"]
                self.total_tn += m["tn"]

    def compute(self) -> Dict[str, float]:
        """Compute both global pooled metrics and mean per-image metrics."""
        global_m = compute_metrics_from_counts(
            self.total_tp,
            self.total_fp,
            self.total_fn,
            self.total_tn,
            zero_division=self.zero_division,
        )

        if self.per_image_records:
            mean_prec = float(np.mean([m["precision"] for m in self.per_image_records]))
            mean_rec = float(np.mean([m["recall"] for m in self.per_image_records]))
            mean_f1 = float(np.mean([m["f1"] for m in self.per_image_records]))
            mean_iou = float(np.mean([m["iou"] for m in self.per_image_records]))
        else:
            mean_prec = mean_rec = mean_f1 = mean_iou = 0.0

        return {
            "global_precision": global_m["precision"],
            "global_recall": global_m["recall"],
            "global_f1": global_m["f1"],
            "global_iou": global_m["iou"],
            "mean_precision": mean_prec,
            "mean_recall": mean_rec,
            "mean_f1": mean_f1,
            "mean_iou": mean_iou,
            "tp": self.total_tp,
            "fp": self.total_fp,
            "fn": self.total_fn,
            "tn": self.total_tn,
            "num_samples": len(self.per_image_records),
        }

