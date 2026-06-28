import csv
from pathlib import Path

from veripet.research.ingest import (
    build_petface_identity_manifest,
    build_petface_verification_manifest,
    discover_breed_folder_dataset,
    stratified_classification_split,
)
from veripet.research.taxonomy import BreedTaxonomy


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def test_petface_ingest_adds_id_and_breed_metadata(tmp_path: Path):
    annotations = tmp_path / "dog.csv"
    split_csv = tmp_path / "train.csv"
    verification_csv = tmp_path / "verification.csv"
    _write_csv(
        annotations,
        [
            ["Name", "Breed", "Color1", "Color2", "Gender"],
            ["000000", "Border collie", "Unknown", "Unknown", "Unknown"],
            ["000001", "Siberian husky", "Unknown", "Unknown", "Unknown"],
        ],
    )
    _write_csv(split_csv, [["filename", "label"], ["dog/000000/00.png", "0"], ["dog/000001/00.png", "1"]])
    _write_csv(
        verification_csv,
        [["filename1", "filename2", "label"], ["dog/000000/00.png", "dog/000001/00.png", "0"]],
    )
    taxonomy = BreedTaxonomy.from_pairs(
        [("Border collie", "Border collie"), ("Siberian husky", "Siberian husky")]
    )

    identity_manifest = build_petface_identity_manifest(split_csv, annotations, taxonomy=taxonomy)
    verification_manifest = build_petface_verification_manifest(
        verification_csv, annotations, taxonomy=taxonomy
    )

    assert identity_manifest[0]["id"] == "000000"
    assert identity_manifest[0]["breed"] == "Scottish Fold"
    assert verification_manifest[0]["pair_type"] == "negative_diff_breed"
    assert verification_manifest[0]["id1"] == "000000"


def test_breed_folder_ingest_and_split(tmp_path: Path):
    dataset_root = tmp_path / "stanford_dogs_sample"
    for breed in ("Border collie", "Siberian husky"):
        breed_dir = dataset_root / breed
        breed_dir.mkdir(parents=True)
        for idx in range(4):
            (breed_dir / f"{breed}_{idx}.jpg").write_bytes(b"img")

    taxonomy = BreedTaxonomy.from_pairs(
        [("Border collie", "Border collie"), ("Siberian husky", "Siberian husky")]
    )
    rows = discover_breed_folder_dataset(dataset_root, taxonomy=taxonomy)
    manifest = stratified_classification_split(rows, seed=7)

    assert len(rows) == 8
    assert rows[0]["breed"] in {"Border collie", "Siberian husky"}
    assert {"train", "val", "test"} <= {row["split"] for row in manifest}
