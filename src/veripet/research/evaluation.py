"""Threshold calibration and evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from .metrics import (
    accuracy_from_confusion,
    binary_confusion,
    eer,
    pair_type_accuracy,
    precision_recall_f1,
    roc_auc,
    roc_points,
    tar_at_far,
)


@dataclass(frozen=True)
class CalibrationResult:
    strategy: str
    threshold: float
    auc: float
    eer: float


@dataclass(frozen=True)
class VerificationEvaluationResult:
    threshold: float
    metrics: Dict[str, float]


def calibrate_threshold(
    labels: Sequence[int], scores: Sequence[float], strategy: str = "youden"
) -> CalibrationResult:
    points = roc_points(labels, scores)
    if strategy == "eer":
        best = min(points, key=lambda point: abs(point["fpr"] - (1.0 - point["tpr"])))
    elif strategy.startswith("far@"):
        target = float(strategy.split("@", 1)[1])
        eligible = [point for point in points if point["fpr"] <= target]
        best = max(eligible or points, key=lambda point: point["tpr"])
    else:
        best = max(points, key=lambda point: point["tpr"] - point["fpr"])
    return CalibrationResult(
        strategy=strategy,
        threshold=float(best["threshold"]),
        auc=roc_auc(labels, scores),
        eer=eer(labels, scores),
    )


def evaluate_verification_predictions(
    pairs: Iterable[Dict[str, object]],
    threshold: float,
) -> VerificationEvaluationResult:
    rows = list(pairs)
    labels = [int(row["label"]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    confusion = binary_confusion(labels, scores, threshold)
    metrics = {
        "auc": roc_auc(labels, scores),
        "eer": eer(labels, scores),
        "accuracy": accuracy_from_confusion(confusion),
        "tar_at_far_1pct": tar_at_far(labels, scores, 0.01),
        "tar_at_far_0_1pct": tar_at_far(labels, scores, 0.001),
    }
    metrics.update(precision_recall_f1(confusion))
    metrics.update(pair_type_accuracy(rows, threshold))
    return VerificationEvaluationResult(threshold=threshold, metrics=metrics)
