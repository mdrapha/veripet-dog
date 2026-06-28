"""Dataset-specific raw ingest helpers for local preparation before Colab."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .pair_mining import PairRecord, pair_records_to_rows
from .taxonomy import BreedTaxonomy, normalize_breed_label

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def petface_identity_from_filename(filename: str) -> str:
    parts = Path(filename).parts
    if len(parts) < 2:
        raise ValueError(f"Unexpected PetFace filename format: {filename}")
    return parts[-2]


def load_petface_breed_map(annotations_csv: Path, taxonomy: Optional[BreedTaxonomy] = None) -> Dict[str, str]:
    rows = _read_csv_rows(annotations_csv)
    breed_map: Dict[str, str] = {}
    for row in rows:
        identity = row["Name"].strip()
        breed = row.get("Breed", "").strip() or "breed_unknown"
        if taxonomy:
            breed = normalize_breed_label(breed, taxonomy)
        breed_map[identity] = breed
    return breed_map


def build_petface_identity_manifest(
    split_csv: Path,
    annotations_csv: Path,
    taxonomy: Optional[BreedTaxonomy] = None,
    split_name: str = "train",
) -> List[Dict[str, str]]:
    breed_map = load_petface_breed_map(annotations_csv, taxonomy=taxonomy)
    manifest: List[Dict[str, str]] = []
    for row in _read_csv_rows(split_csv):
        filename = row["filename"].strip()
        identity = petface_identity_from_filename(filename)
        manifest.append(
            {
                "filename": filename,
                "label": row["label"].strip(),
                "id": identity,
                "breed": breed_map.get(identity, "breed_unknown"),
                "split": split_name,
            }
        )
    return manifest


def build_petface_identity_manifest_from_txt(
    split_txt: Path,
    annotations_csv: Path,
    taxonomy: Optional[BreedTaxonomy] = None,
    split_name: str = "verification_val",
) -> List[Dict[str, str]]:
    breed_map = load_petface_breed_map(annotations_csv, taxonomy=taxonomy)
    manifest: List[Dict[str, str]] = []
    with split_txt.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            filename = raw_line.strip()
            if not filename:
                continue
            identity = petface_identity_from_filename(filename)
            manifest.append(
                {
                    "filename": filename,
                    "label": identity,
                    "id": identity,
                    "breed": breed_map.get(identity, "breed_unknown"),
                    "split": split_name,
                }
            )
    return manifest


def build_petface_verification_manifest(
    verification_csv: Path,
    annotations_csv: Path,
    taxonomy: Optional[BreedTaxonomy] = None,
) -> List[Dict[str, object]]:
    breed_map = load_petface_breed_map(annotations_csv, taxonomy=taxonomy)
    pairs: List[PairRecord] = []
    for row in _read_csv_rows(verification_csv):
        filename1 = row["filename1"].strip()
        filename2 = row["filename2"].strip()
        id1 = petface_identity_from_filename(filename1)
        id2 = petface_identity_from_filename(filename2)
        label = int(row["label"])
        breed1 = breed_map.get(id1, "breed_unknown")
        breed2 = breed_map.get(id2, "breed_unknown")
        if label == 1:
            pair_type = "positive"
        elif "breed_unknown" in (breed1, breed2):
            pair_type = "breed_unknown"
        elif breed1 == breed2:
            pair_type = "negative_same_breed"
        else:
            pair_type = "negative_diff_breed"
        pairs.append(
            PairRecord(
                filename1=filename1,
                filename2=filename2,
                label=label,
                id1=id1,
                id2=id2,
                breed1=breed1,
                breed2=breed2,
                pair_type=pair_type,
            )
        )
    return pair_records_to_rows(pairs)


def discover_breed_folder_dataset(
    dataset_root: Path,
    taxonomy: Optional[BreedTaxonomy] = None,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for breed_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        raw_breed = breed_dir.name
        breed = normalize_breed_label(raw_breed, taxonomy) if taxonomy else raw_breed
        for image_path in sorted(path for path in breed_dir.iterdir() if _is_image_file(path)):
            rows.append(
                {
                    "filename": str(image_path.relative_to(dataset_root)).replace("\\", "/"),
                    "breed": breed,
                }
            )
    return rows


def stratified_classification_split(
    rows: Sequence[Dict[str, str]],
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> List[Dict[str, str]]:
    if round(train_fraction + val_fraction + test_fraction, 6) != 1.0:
        raise ValueError("train_fraction + val_fraction + test_fraction must equal 1.0")

    by_breed: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        by_breed.setdefault(row["breed"], []).append(dict(row))

    rng = random.Random(seed)
    manifest: List[Dict[str, str]] = []
    for breed, breed_rows in sorted(by_breed.items()):
        breed_rows = list(breed_rows)
        rng.shuffle(breed_rows)
        total = len(breed_rows)
        train_end = max(1, int(total * train_fraction))
        val_end = min(total, train_end + max(1, int(total * val_fraction)))
        for idx, row in enumerate(breed_rows):
            if idx < train_end:
                split = "train"
            elif idx < val_end:
                split = "val"
            else:
                split = "test"
            manifest.append({"filename": row["filename"], "breed": row["breed"], "split": split})
    manifest.sort(key=lambda row: (row["split"], row["breed"], row["filename"]))
    return manifest
