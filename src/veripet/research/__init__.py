"""Reusable research core for VeriPet experiments."""

from .config import ExperimentConfig, PathsConfig
from .datasets import ClassificationManifestDataset, VerificationPairsDataset
from .evaluation import (
    CalibrationResult,
    VerificationEvaluationResult,
    calibrate_threshold,
    evaluate_verification_predictions,
)
from .ingest import (
    build_petface_identity_manifest,
    build_petface_verification_manifest,
    discover_breed_folder_dataset,
    load_petface_breed_map,
    petface_identity_from_filename,
    stratified_classification_split,
)
from .pair_mining import (
    PairRecord,
    PairSplitResult,
    build_balanced_pairs_for_split_rows,
    build_pairs_for_split_rows,
    build_verification_pairs,
)
from .sampling import (
    SampleBundle,
    SampleProfile,
    build_experiment_subset,
    build_identity_sample,
    build_local_sample_bundle,
)
from .taxonomy import BreedTaxonomy, normalize_breed_label

__all__ = [
    "BreedTaxonomy",
    "CalibrationResult",
    "ClassificationManifestDataset",
    "ExperimentConfig",
    "PairRecord",
    "PairSplitResult",
    "PathsConfig",
    "SampleProfile",
    "SampleBundle",
    "VerificationEvaluationResult",
    "VerificationPairsDataset",
    "build_petface_identity_manifest",
    "build_petface_verification_manifest",
    "build_balanced_pairs_for_split_rows",
    "build_pairs_for_split_rows",
    "build_experiment_subset",
    "build_identity_sample",
    "build_local_sample_bundle",
    "build_verification_pairs",
    "calibrate_threshold",
    "discover_breed_folder_dataset",
    "evaluate_verification_predictions",
    "load_petface_breed_map",
    "normalize_breed_label",
    "petface_identity_from_filename",
    "stratified_classification_split",
]
