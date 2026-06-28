"""Controlled identity sampling for large verification datasets."""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class SampleProfile:
    name: str
    min_images_per_identity: int
    identity_fraction: Optional[float] = None
    max_identities: Optional[int] = None
    seed: int = 42

    def resolve_target_identities(self, total_eligible: int) -> int:
        if total_eligible <= 0:
            return 0
        if self.identity_fraction is not None:
            return min(total_eligible, max(1, math.ceil(total_eligible * self.identity_fraction)))
        if self.max_identities is not None:
            return min(total_eligible, self.max_identities)
        return total_eligible


@dataclass(frozen=True)
class SampleBundle:
    rows: List[Dict[str, str]]
    identity_count: int
    image_count: int
    split_counts: Dict[str, int]
    identity_fraction: float


DEFAULT_SAMPLE_PROFILES = {
    "debug_sample": SampleProfile("debug_sample", min_images_per_identity=2, max_identities=4),
    "dev_sample": SampleProfile("dev_sample", min_images_per_identity=3, max_identities=16),
    "local_10pct": SampleProfile("local_10pct", min_images_per_identity=2, identity_fraction=0.10),
    "full_dataset": SampleProfile("full_dataset", min_images_per_identity=1),
}


def _group_by_identity(rows: Sequence[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["label"], []).append(row)
    return grouped


def build_identity_sample(
    rows: Sequence[Dict[str, str]], profile: SampleProfile, split_column: Optional[str] = None
) -> List[Dict[str, str]]:
    if split_column and rows and split_column in rows[0]:
        return _build_split_aware_identity_sample(rows, profile, split_column)

    grouped = _group_by_identity(rows)
    eligible = [
        identity
        for identity, identity_rows in grouped.items()
        if len(identity_rows) >= profile.min_images_per_identity
    ]
    rng = random.Random(profile.seed)
    eligible = sorted(eligible)
    rng.shuffle(eligible)
    chosen = eligible[: profile.resolve_target_identities(len(eligible))]

    sampled_rows: List[Dict[str, str]] = []
    for identity in chosen:
        identity_rows = list(grouped[identity])
        rng.shuffle(identity_rows)
        sampled_rows.extend(identity_rows)
    sampled_rows.sort(key=lambda row: (row["label"], row["filename"]))
    return sampled_rows


def _build_split_aware_identity_sample(
    rows: Sequence[Dict[str, str]], profile: SampleProfile, split_column: str
) -> List[Dict[str, str]]:
    by_split: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        by_split.setdefault(row[split_column], []).append(row)

    sampled_rows: List[Dict[str, str]] = []
    for split_name, split_rows in sorted(by_split.items()):
        sampled_rows.extend(build_identity_sample(split_rows, profile, split_column=None))
    sampled_rows.sort(key=lambda row: (row.get(split_column, ""), row["label"], row["filename"]))
    return sampled_rows


def build_experiment_subset(
    rows: Sequence[Dict[str, str]],
    identity_fraction: float,
    min_images_per_identity: int = 2,
    seed: int = 42,
    split_column: Optional[str] = None,
) -> List[Dict[str, str]]:
    profile = SampleProfile(
        name=f"subset_{identity_fraction:.4f}",
        min_images_per_identity=min_images_per_identity,
        identity_fraction=identity_fraction,
        seed=seed,
    )
    return build_identity_sample(rows, profile, split_column=split_column)


def build_local_sample_bundle(
    rows: Sequence[Dict[str, str]],
    local_identity_fraction: float = 0.10,
    experiment_identity_fraction: float = 1.0,
    min_images_per_identity: int = 2,
    seed: int = 42,
    split_column: Optional[str] = None,
) -> SampleBundle:
    local_profile = SampleProfile(
        name="local_sample",
        min_images_per_identity=min_images_per_identity,
        identity_fraction=local_identity_fraction,
        seed=seed,
    )
    local_rows = build_identity_sample(rows, local_profile, split_column=split_column)
    experiment_rows = build_experiment_subset(
        local_rows,
        identity_fraction=experiment_identity_fraction,
        min_images_per_identity=min_images_per_identity,
        seed=seed,
        split_column=split_column,
    )
    grouped = _group_by_identity(experiment_rows)
    split_counts: Dict[str, int] = {}
    if split_column:
        for row in experiment_rows:
            split_counts[row[split_column]] = split_counts.get(row[split_column], 0) + 1
    return SampleBundle(
        rows=experiment_rows,
        identity_count=len(grouped),
        image_count=len(experiment_rows),
        split_counts=split_counts,
        identity_fraction=local_identity_fraction * experiment_identity_fraction,
    )


def save_sample_manifest(rows: Sequence[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
