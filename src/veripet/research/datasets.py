"""Manifest-backed datasets for dog classification and verification experiments."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional


def _read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@dataclass
class ClassificationSample:
    filename: str
    breed: str
    split: str


class ClassificationManifestDataset:
    required_columns = ("filename", "breed", "split")

    def __init__(
        self,
        manifest_path: Path,
        image_root: Optional[Path] = None,
        transform: Optional[Callable[[Dict[str, str]], Dict[str, str]]] = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.image_root = image_root
        self.transform = transform
        self.rows = _read_csv_rows(manifest_path)
        self._validate()

    def _validate(self) -> None:
        if not self.rows:
            return
        missing = set(self.required_columns) - set(self.rows[0].keys())
        if missing:
            raise ValueError(f"Classification manifest missing columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, str]:
        sample = dict(self.rows[index])
        if self.image_root:
            sample["image_path"] = str((self.image_root / sample["filename"]).resolve())
        if self.transform:
            return self.transform(sample)
        return sample


class VerificationPairsDataset:
    required_columns = (
        "filename1",
        "filename2",
        "label",
        "id1",
        "id2",
        "breed1",
        "breed2",
        "pair_type",
    )

    def __init__(
        self,
        pairs_csv: Path,
        image_root: Optional[Path] = None,
        transform: Optional[Callable[[Dict[str, str]], Dict[str, str]]] = None,
    ) -> None:
        self.pairs_csv = pairs_csv
        self.image_root = image_root
        self.transform = transform
        self.rows = _read_csv_rows(pairs_csv)
        self._validate()

    def _validate(self) -> None:
        if not self.rows:
            return
        missing = set(self.required_columns) - set(self.rows[0].keys())
        if missing:
            raise ValueError(f"Verification pairs missing columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, str]:
        sample = dict(self.rows[index])
        if self.image_root:
            sample["image_path1"] = str((self.image_root / sample["filename1"]).resolve())
            sample["image_path2"] = str((self.image_root / sample["filename2"]).resolve())
        if self.transform:
            return self.transform(sample)
        return sample
