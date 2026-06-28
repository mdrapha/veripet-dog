"""Tabular plot payload builders for notebook visualization."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

from .metrics import roc_points


def build_roc_rows(labels: Sequence[int], scores: Sequence[float]) -> List[Dict[str, float]]:
    return roc_points(labels, scores)


def build_score_histogram_rows(scores: Iterable[float], bins: int = 10) -> List[Dict[str, float]]:
    scores = list(scores)
    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    if low == high:
        return [{"bin_start": low, "bin_end": high, "count": len(scores)}]
    width = (high - low) / bins
    counts = [0 for _ in range(bins)]
    for score in scores:
        index = min(int((score - low) / width), bins - 1)
        counts[index] += 1
    rows = []
    for idx, count in enumerate(counts):
        rows.append(
            {
                "bin_start": low + idx * width,
                "bin_end": low + (idx + 1) * width,
                "count": count,
            }
        )
    return rows
