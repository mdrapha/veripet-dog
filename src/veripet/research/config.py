"""Experiment configuration primitives for Colab-friendly research flows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Literal, Optional

Species = Literal["dog"]
Task = Literal["classification", "verification"]
SampleProfileName = Literal["debug_sample", "dev_sample", "full_dataset"]


@dataclass(frozen=True)
class ColabConfig:
    mount_root: Path = Path("/content/drive/MyDrive/veripet")
    use_drive: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    precision: str = "16-mixed"
    grad_accum_steps: int = 1


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    batch_size: int = 64
    epochs: int = 10
    embedding_dim: int = 512
    input_size: int = 224
    optimizer: str = "AdamW"
    scheduler: str = "CosineAnnealingLR"
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4


@dataclass(frozen=True)
class EvalConfig:
    checkpoint_metric: str = "auc"
    threshold_strategy: str = "youden"
    far_targets: tuple[float, ...] = (0.01, 0.001)


@dataclass(frozen=True)
class PathsConfig:
    project_root: Path = Path(".")
    drive_root: Path = Path("drive/MyDrive/veripet")
    raw_ingest_root: Path = Path("data/raw_ingest")
    species: Species = "dog"
    task: Task = "verification"
    experiment_name: str = "baseline"

    def species_dirname(self) -> str:
        return "dogs"

    def research_root(self) -> Path:
        return self.project_root / self.drive_root / self.species_dirname() / self.task

    def images_dir(self) -> Path:
        return self.research_root() / "images"

    def split_dir(self) -> Path:
        return self.research_root() / "split"

    def annotations_dir(self) -> Path:
        return self.research_root() / "annotations"

    def artifacts_dir(self) -> Path:
        return self.research_root() / "artifacts" / self.experiment_name

    def results_dir(self) -> Path:
        return self.research_root() / "results" / self.experiment_name

    def manifests_dir(self) -> Path:
        return self.research_root() / "manifests"

    def sample_dirs(self) -> Dict[SampleProfileName, Path]:
        base = self.research_root()
        return {
            "debug_sample": base / "images_sample" / "debug_sample",
            "dev_sample": base / "images_sample" / "dev_sample",
            "full_dataset": base / "images",
        }

    def ensure_tree(self) -> None:
        for path in (
            self.raw_ingest_root,
            self.images_dir(),
            self.split_dir(),
            self.annotations_dir(),
            self.artifacts_dir(),
            self.results_dir(),
            self.manifests_dir(),
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ExperimentConfig:
    species: Species = "dog"
    task: Task = "verification"
    backbone: str = "convnext_small.fb_in22k_ft_in1k"
    head: str = "ArcFace"
    loss: str = "ArcFace"
    sample_profile: SampleProfileName = "debug_sample"
    paths: PathsConfig = field(default_factory=PathsConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    colab: ColabConfig = field(default_factory=ColabConfig)
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
