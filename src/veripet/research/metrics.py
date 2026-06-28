"""Metrics for classification and verification tasks."""

from __future__ import annotations

from collections import Counter
from math import sqrt
from typing import Dict, Iterable, List, Sequence


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def binary_confusion(
    labels: Sequence[int], scores: Sequence[float], threshold: float
) -> Dict[str, int]:
    tp = fp = tn = fn = 0
    for label, score in zip(labels, scores):
        pred = 1 if score >= threshold else 0
        if label == 1 and pred == 1:
            tp += 1
        elif label == 0 and pred == 1:
            fp += 1
        elif label == 0 and pred == 0:
            tn += 1
        else:
            fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def accuracy_from_confusion(confusion: Dict[str, int]) -> float:
    total = sum(confusion.values())
    return _safe_div(confusion["tp"] + confusion["tn"], total)


def precision_recall_f1(confusion: Dict[str, int]) -> Dict[str, float]:
    precision = _safe_div(confusion["tp"], confusion["tp"] + confusion["fp"])
    recall = _safe_div(confusion["tp"], confusion["tp"] + confusion["fn"])
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def roc_points(labels: Sequence[int], scores: Sequence[float]) -> List[Dict[str, float]]:
    thresholds = sorted(set(scores), reverse=True)
    thresholds = [max(thresholds) + 1e-9] + thresholds + [min(thresholds) - 1e-9]
    points: List[Dict[str, float]] = []
    positives = sum(labels)
    negatives = len(labels) - positives
    for threshold in thresholds:
        confusion = binary_confusion(labels, scores, threshold)
        tpr = _safe_div(confusion["tp"], positives)
        fpr = _safe_div(confusion["fp"], negatives)
        points.append({"threshold": threshold, "tpr": tpr, "fpr": fpr})
    points.sort(key=lambda row: row["fpr"])
    return points


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    total = 0
    for pos in positives:
        for neg in negatives:
            total += 1
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return _safe_div(wins, total)


def eer(labels: Sequence[int], scores: Sequence[float]) -> float:
    best_gap = None
    best_eer = 1.0
    for point in roc_points(labels, scores):
        fnr = 1.0 - point["tpr"]
        gap = abs(point["fpr"] - fnr)
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_eer = (point["fpr"] + fnr) / 2.0
    return best_eer


def tar_at_far(labels: Sequence[int], scores: Sequence[float], far_target: float) -> float:
    best_tar = 0.0
    for point in roc_points(labels, scores):
        if point["fpr"] <= far_target:
            best_tar = max(best_tar, point["tpr"])
    return best_tar


def classification_metrics(
    y_true: Sequence[str], y_pred: Sequence[str]
) -> Dict[str, float]:
    correct = sum(1 for truth, pred in zip(y_true, y_pred) if truth == pred)
    accuracy = _safe_div(correct, len(y_true))
    labels = sorted(set(y_true) | set(y_pred))
    precision_sum = recall_sum = f1_sum = 0.0
    for label in labels:
        tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred == label)
        fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth != label and pred == label)
        fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred != label)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1
    count = max(len(labels), 1)
    return {
        "accuracy": accuracy,
        "precision_macro": precision_sum / count,
        "recall_macro": recall_sum / count,
        "f1_macro": f1_sum / count,
    }


def pair_type_accuracy(rows: Iterable[Dict[str, object]], threshold: float) -> Dict[str, float]:
    grouped: Dict[str, List[tuple[int, float]]] = {}
    for row in rows:
        pair_type = str(row["pair_type"])
        grouped.setdefault(pair_type, []).append((int(row["label"]), float(row["score"])))
    metrics: Dict[str, float] = {}
    for pair_type, pairs in grouped.items():
        confusion = binary_confusion(
            [label for label, _ in pairs],
            [score for _, score in pairs],
            threshold,
        )
        metrics[f"{pair_type}_accuracy"] = accuracy_from_confusion(confusion)
    return metrics
