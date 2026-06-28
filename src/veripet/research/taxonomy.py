"""Breed taxonomy helpers and alias normalization."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[_/]+", " ", value)
    value = re.sub(r"[^a-z0-9 ]+", "", value)
    value = re.sub(r"\s+", " ", value)
    return value


@dataclass(frozen=True)
class BreedTaxonomy:
    canonical_by_alias: Dict[str, str]

    @classmethod
    def from_csv(cls, aliases_csv: Path) -> "BreedTaxonomy":
        canonical_by_alias: Dict[str, str] = {}
        with aliases_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                canonical = row["canonical_name"].strip()
                alias = row["alias"].strip()
                canonical_by_alias[_slugify(canonical)] = canonical
                canonical_by_alias[_slugify(alias)] = canonical
        return cls(canonical_by_alias=canonical_by_alias)

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[str, str]]) -> "BreedTaxonomy":
        canonical_by_alias: Dict[str, str] = {}
        for alias, canonical in pairs:
            canonical_by_alias[_slugify(alias)] = canonical
            canonical_by_alias[_slugify(canonical)] = canonical
        return cls(canonical_by_alias=canonical_by_alias)

    def normalize(self, label: Optional[str], default: str = "breed_unknown") -> str:
        if not label:
            return default
        return self.canonical_by_alias.get(_slugify(label), default)


def normalize_breed_label(
    label: Optional[str], taxonomy: BreedTaxonomy, default: str = "breed_unknown"
) -> str:
    return taxonomy.normalize(label, default=default)
