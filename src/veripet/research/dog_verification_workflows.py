"""Dog identity verification workflows for PetFace Dog experiments."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .evaluation import calibrate_threshold, evaluate_verification_predictions
from .io import ensure_dir, save_csv, save_json

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
FINAL_SEEDS = [7, 13, 42, 101, 2026]
TRIAGE_SEEDS = [42, 101, 2026]
DEFAULT_LOSSES = ["Softmax", "ArcFace", "TripletMarginLoss", "CosFace", "AdaFace", "MagFace"]


@dataclass(frozen=True)
class DogVerificationPaths:
    drive_base: Path = Path("/content/drive/MyDrive/dogs")
    work_dir: Path = Path("/content/dogs")
    results_dir: Path = Path("/content/drive/MyDrive/dogs/verification_experiments")

    @property
    def image_root(self) -> Path:
        return self.work_dir

    @property
    def split_dir(self) -> Path:
        return self.drive_base / "split"

    @property
    def annotations_csv(self) -> Path:
        return self.drive_base / "dog.csv"


@dataclass(frozen=True)
class DogVerificationSweepConfig:
    paths: DogVerificationPaths = field(default_factory=DogVerificationPaths)
    run_name: str = "triage_loss_sweep"
    train_identity_fraction: float = 0.25
    pair_fraction: float = 0.25
    split_sample_pct: float = 10.0
    losses: Sequence[str] = field(default_factory=lambda: list(DEFAULT_LOSSES))
    seeds: Sequence[int] = field(default_factory=lambda: list(TRIAGE_SEEDS))
    epochs: int = 8
    batch_size: int = 128
    eval_batch_size: int = 256
    num_workers: int = 4
    use_amp: bool = True
    embedding_dim: int = 512
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    input_size: int = 224
    train_csv_name: str = "train_identification_60pct_ids.csv"
    val_pairs_name: str = "verification_val_pairs.csv"
    test_pairs_name: str = "verification_test_pairs.csv"
    dry_run: bool = False


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def petface_identity_from_filename(filename: str) -> str:
    parts = Path(filename).parts
    if len(parts) < 2:
        return Path(filename).stem
    return parts[-2]


def load_identity_breed_map(annotations_csv: Path) -> Dict[str, str]:
    if not annotations_csv.exists():
        return {}
    rows = read_csv_rows(annotations_csv)
    identity_col = next((col for col in ["Name", "name", "identity", "id"] if col in rows[0]), None) if rows else None
    breed_col = next((col for col in ["Breed", "breed", "race"] if col in rows[0]), None) if rows else None
    if identity_col is None or breed_col is None:
        return {}
    return {row[identity_col].strip(): row.get(breed_col, "breed_unknown").strip() for row in rows}


def enrich_pair_rows(pair_rows: Sequence[Dict[str, str]], identity_to_breed: Dict[str, str]) -> List[Dict[str, object]]:
    enriched: List[Dict[str, object]] = []
    for row in pair_rows:
        payload: Dict[str, object] = dict(row)
        filename1 = str(row["filename1"])
        filename2 = str(row["filename2"])
        id1 = petface_identity_from_filename(filename1)
        id2 = petface_identity_from_filename(filename2)
        breed1 = identity_to_breed.get(id1, "breed_unknown")
        breed2 = identity_to_breed.get(id2, "breed_unknown")
        label = int(row["label"])
        if "pair_type" in row and row["pair_type"]:
            pair_type = row["pair_type"]
        elif label == 1:
            pair_type = "positive"
        elif "breed_unknown" in {breed1, breed2}:
            pair_type = "breed_unknown"
        elif breed1 == breed2:
            pair_type = "negative_same_breed"
        else:
            pair_type = "negative_diff_breed"
        payload.update({"label": label, "id1": id1, "id2": id2, "breed1": breed1, "breed2": breed2, "pair_type": pair_type})
        enriched.append(payload)
    return enriched


def sample_rows_by_identity(rows: Sequence[Dict[str, str]], fraction: float, seed: int) -> List[Dict[str, str]]:
    if fraction >= 1.0:
        return [dict(row) for row in rows]
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(str(row["label"]), []).append(dict(row))
    keys = sorted(grouped)
    rng = random.Random(seed)
    rng.shuffle(keys)
    keep = set(keys[: max(1, int(round(len(keys) * fraction)))])
    return [row for key in sorted(keep) for row in grouped[key]]


def sample_pair_rows(rows: Sequence[Dict[str, object]], fraction: float, seed: int) -> List[Dict[str, object]]:
    if fraction >= 1.0:
        return [dict(row) for row in rows]
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        key = str(row.get("pair_type", row.get("label", "unknown")))
        grouped.setdefault(key, []).append(dict(row))
    rng = random.Random(seed)
    sampled: List[Dict[str, object]] = []
    for key, group in grouped.items():
        group = list(group)
        rng.shuffle(group)
        sampled.extend(group[: max(1, int(round(len(group) * fraction)))])
    rng.shuffle(sampled)
    return sampled


def build_verification_transforms(input_size: int) -> Dict[str, object]:
    from torchvision import transforms

    return {
        "train": transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        ),
        "eval": transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        ),
    }


class DogVerificationTrainDataset:
    def __init__(self, rows: Sequence[Dict[str, str]], image_root: Path, transform: object) -> None:
        self.rows = list(rows)
        self.image_root = image_root
        self.transform = transform
        labels = sorted({str(row["label"]) for row in self.rows})
        self.label_to_idx = {label: index for index, label in enumerate(labels)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        from PIL import Image

        row = self.rows[index]
        image = Image.open(self.image_root / row["filename"]).convert("RGB")
        return self.transform(image), self.label_to_idx[str(row["label"])]


class DogVerificationPairsDataset:
    def __init__(self, rows: Sequence[Dict[str, object]], image_root: Path, transform: object) -> None:
        self.rows = list(rows)
        self.image_root = image_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        from PIL import Image

        row = self.rows[index]
        image1 = Image.open(self.image_root / str(row["filename1"])).convert("RGB")
        image2 = Image.open(self.image_root / str(row["filename2"])).convert("RGB")
        return self.transform(image1), self.transform(image2), int(row["label"]), dict(row)


def _collate_pairs(batch):
    import torch

    images1, images2, labels, rows = zip(*batch)
    return torch.stack(images1), torch.stack(images2), torch.tensor(labels, dtype=torch.long), list(rows)


def _loader_kwargs(batch_size: int, num_workers: int, shuffle: bool, drop_last: bool = False):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": num_workers > 0,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        kwargs.update({"prefetch_factor": 2, "persistent_workers": True})
    return kwargs


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DogEmbeddingModel:
    def __init__(self, embedding_dim: int = 512, trainable_projection: bool = True) -> None:
        import torch.nn as nn
        from torchvision import models

        super().__init__()
        self.module = nn.Module()
        backbone = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
        in_features = backbone.classifier[2].in_features
        backbone.classifier[2] = nn.Identity()
        self.module.backbone = backbone
        if trainable_projection:
            self.module.embedding = nn.Sequential(
                nn.Linear(in_features, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
            )
        else:
            self.module.embedding = nn.Identity()
        if not trainable_projection:
            for parameter in self.module.parameters():
                parameter.requires_grad = False

    def to(self, device):
        self.module = self.module.to(device)
        return self

    def train(self):
        self.module.train()

    def eval(self):
        self.module.eval()

    def parameters(self):
        return self.module.parameters()

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        return self.module.load_state_dict(state_dict)

    def __call__(self, images):
        import torch.nn.functional as F

        features = self.module.backbone(images)
        raw = self.module.embedding(features)
        norms = raw.norm(dim=1, keepdim=True).clamp_min(1e-6)
        return F.normalize(raw, p=2, dim=1), norms


def batch_hard_triplet_loss(embeddings, labels, margin: float = 0.2):
    import torch
    import torch.nn.functional as F

    distances = torch.cdist(embeddings, embeddings, p=2)
    losses = []
    for index in range(labels.numel()):
        positive = labels.eq(labels[index])
        negative = ~positive
        positive[index] = False
        if positive.any() and negative.any():
            hardest_positive = distances[index][positive].max()
            hardest_negative = distances[index][negative].min()
            losses.append(F.relu(hardest_positive - hardest_negative + margin))
    if not losses:
        return embeddings.new_tensor(0.0)
    return torch.stack(losses).mean()


def train_one_epoch(model, head, loader, optimizer, scaler, device, loss_name: str, use_amp: bool):
    import torch
    import torch.nn as nn
    from torch.amp import autocast

    model.train()
    if head is not None:
        head.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_count = 0
    correct = 0
    classified = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            embeddings, norms = model(images)
            if loss_name.lower() == "tripletmarginloss":
                loss = batch_hard_triplet_loss(embeddings, labels, margin=0.2)
                logits = None
            else:
                output = head(embeddings, labels, norms)
                logits = output.logits
                loss = criterion(logits, labels) + output.regularization
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + (list(head.parameters()) if head is not None else []),
            max_norm=5.0,
        )
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item()) * images.size(0)
        total_count += int(images.size(0))
        if logits is not None:
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            classified += int(labels.numel())
    return {
        "train_loss": total_loss / max(total_count, 1),
        "train_accuracy": correct / max(classified, 1) if classified else 0.0,
    }


def score_pairs(model, loader, device, use_amp: bool) -> List[Dict[str, object]]:
    import torch
    import torch.nn.functional as F
    from torch.amp import autocast

    model.eval()
    scored: List[Dict[str, object]] = []
    with torch.no_grad():
        for image1, image2, labels, rows in loader:
            image1 = image1.to(device, non_blocking=True)
            image2 = image2.to(device, non_blocking=True)
            with autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
                emb1, _norm1 = model(image1)
                emb2, _norm2 = model(image2)
                scores = F.cosine_similarity(emb1, emb2, dim=1)
            for row, score, label in zip(rows, scores.detach().cpu().tolist(), labels.cpu().tolist()):
                payload = dict(row)
                payload["score"] = float(score)
                payload["label"] = int(label)
                scored.append(payload)
    return scored


def _prepare_data(config: DogVerificationSweepConfig):
    train_rows = read_csv_rows(config.paths.split_dir / config.train_csv_name)
    val_pairs = read_csv_rows(config.paths.split_dir / config.val_pairs_name)
    test_pairs = read_csv_rows(config.paths.split_dir / config.test_pairs_name)
    identity_to_breed = load_identity_breed_map(config.paths.annotations_csv)
    train_rows = sample_rows_by_identity(train_rows, config.train_identity_fraction, seed=42)
    val_pairs_enriched = sample_pair_rows(enrich_pair_rows(val_pairs, identity_to_breed), config.pair_fraction, seed=42)
    test_pairs_enriched = sample_pair_rows(enrich_pair_rows(test_pairs, identity_to_breed), config.pair_fraction, seed=43)
    return train_rows, val_pairs_enriched, test_pairs_enriched


def run_frozen_embedding_baseline(config: DogVerificationSweepConfig) -> Dict[str, object]:
    import torch
    from torch.utils.data import DataLoader

    set_seed(42)
    train_rows, val_pairs, test_pairs = _prepare_data(config)
    transforms_by_split = build_verification_transforms(config.input_size)
    device = _device()
    model = DogEmbeddingModel(embedding_dim=config.embedding_dim, trainable_projection=False).to(device)
    val_loader = DataLoader(
        DogVerificationPairsDataset(val_pairs, config.paths.image_root, transforms_by_split["eval"]),
        collate_fn=_collate_pairs,
        **_loader_kwargs(config.eval_batch_size, config.num_workers, shuffle=False),
    )
    test_loader = DataLoader(
        DogVerificationPairsDataset(test_pairs, config.paths.image_root, transforms_by_split["eval"]),
        collate_fn=_collate_pairs,
        **_loader_kwargs(config.eval_batch_size, config.num_workers, shuffle=False),
    )
    val_scored = score_pairs(model, val_loader, device, config.use_amp)
    calibration = calibrate_threshold([int(row["label"]) for row in val_scored], [float(row["score"]) for row in val_scored])
    test_scored = score_pairs(model, test_loader, device, config.use_amp)
    val_eval = evaluate_verification_predictions(val_scored, calibration.threshold)
    test_eval = evaluate_verification_predictions(test_scored, calibration.threshold)
    results_dir = ensure_dir(config.paths.results_dir / config.run_name / "frozen_cosine_baseline")
    save_csv(val_scored, results_dir / "verification_val_scored.csv")
    save_csv(test_scored, results_dir / "verification_test_scored.csv")
    summary = {
        "experiment_name": "frozen_convnext_small_cosine",
        "train_rows": len(train_rows),
        "val_pairs": len(val_pairs),
        "test_pairs": len(test_pairs),
        "threshold": calibration.threshold,
        "val_metrics": val_eval.metrics,
        "test_metrics": test_eval.metrics,
        "results_dir": str(results_dir),
    }
    save_json(summary, results_dir / "summary.json")
    return summary


def run_single_verification_experiment(
    config: DogVerificationSweepConfig,
    loss_name: str,
    seed: int,
    *,
    train_rows: Sequence[Dict[str, str]],
    val_pairs: Sequence[Dict[str, object]],
    test_pairs: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    import torch
    from torch.amp import GradScaler
    from torch.utils.data import DataLoader

    set_seed(seed)
    started = time.time()
    transforms_by_split = build_verification_transforms(config.input_size)
    train_dataset = DogVerificationTrainDataset(train_rows, config.paths.image_root, transforms_by_split["train"])
    label_to_idx = train_dataset.label_to_idx
    train_loader = DataLoader(
        train_dataset,
        **_loader_kwargs(config.batch_size, config.num_workers, shuffle=True, drop_last=True),
    )
    val_loader = DataLoader(
        DogVerificationPairsDataset(val_pairs, config.paths.image_root, transforms_by_split["eval"]),
        collate_fn=_collate_pairs,
        **_loader_kwargs(config.eval_batch_size, config.num_workers, shuffle=False),
    )
    test_loader = DataLoader(
        DogVerificationPairsDataset(test_pairs, config.paths.image_root, transforms_by_split["eval"]),
        collate_fn=_collate_pairs,
        **_loader_kwargs(config.eval_batch_size, config.num_workers, shuffle=False),
    )
    device = _device()
    model = DogEmbeddingModel(embedding_dim=config.embedding_dim).to(device)
    from .metric_heads import build_metric_head

    head = build_metric_head(loss_name, config.embedding_dim, len(label_to_idx))
    if head is not None:
        head = head.to(device)
    params = list(model.parameters()) + (list(head.parameters()) if head is not None else [])
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=config.use_amp and device.type == "cuda")
    best_auc = -1.0
    best_state = None
    history: List[Dict[str, object]] = []
    for epoch in range(1, config.epochs + 1):
        row = {"epoch": epoch}
        row.update(train_one_epoch(model, head, train_loader, optimizer, scaler, device, loss_name, config.use_amp))
        val_scored = score_pairs(model, val_loader, device, config.use_amp)
        calibration = calibrate_threshold(
            [int(item["label"]) for item in val_scored],
            [float(item["score"]) for item in val_scored],
        )
        val_eval = evaluate_verification_predictions(val_scored, calibration.threshold)
        row.update(
            {
                "val_auc": val_eval.metrics["auc"],
                "val_accuracy": val_eval.metrics["accuracy"],
                "val_threshold": calibration.threshold,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        history.append(row)
        if float(row["val_auc"]) > best_auc:
            best_auc = float(row["val_auc"])
            best_state = {
                "model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                "head": {key: value.detach().cpu().clone() for key, value in head.state_dict().items()} if head is not None else None,
                "threshold": calibration.threshold,
                "epoch": epoch,
            }
        scheduler.step()
    if best_state is None:
        raise RuntimeError("Verification experiment did not produce a checkpoint")
    model.load_state_dict(best_state["model"])
    if head is not None and best_state["head"] is not None:
        head.load_state_dict(best_state["head"])
    threshold = float(best_state["threshold"])
    val_scored = score_pairs(model, val_loader, device, config.use_amp)
    test_scored = score_pairs(model, test_loader, device, config.use_amp)
    val_eval = evaluate_verification_predictions(val_scored, threshold)
    test_eval = evaluate_verification_predictions(test_scored, threshold)
    experiment_name = f"convnext_small__{loss_name.lower()}__seed{seed}"
    results_dir = ensure_dir(config.paths.results_dir / config.run_name / experiment_name)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "head_state_dict": head.state_dict() if head is not None else None,
            "threshold": threshold,
            "label_to_idx": label_to_idx,
            "loss": loss_name,
            "seed": seed,
        },
        results_dir / "best_verifier.pt",
    )
    save_csv(history, results_dir / "training_history.csv")
    save_csv(val_scored, results_dir / "verification_val_scored.csv")
    save_csv(test_scored, results_dir / "verification_test_scored.csv")
    summary = {
        "experiment_name": experiment_name,
        "loss": loss_name,
        "seed": seed,
        "train_rows": len(train_rows),
        "train_identities": len(label_to_idx),
        "val_pairs": len(val_pairs),
        "test_pairs": len(test_pairs),
        "threshold": threshold,
        "best_epoch": int(best_state["epoch"]),
        "val_metrics": val_eval.metrics,
        "test_metrics": test_eval.metrics,
        "elapsed_seconds": round(time.time() - started, 1),
        "results_dir": str(results_dir),
    }
    save_json(summary, results_dir / "summary.json")
    return summary


def _leaderboard_row(summary: Dict[str, object]) -> Dict[str, object]:
    val = summary["val_metrics"]
    test = summary["test_metrics"]
    return {
        "experiment_name": summary["experiment_name"],
        "loss": summary.get("loss", "FrozenCosine"),
        "seed": summary.get("seed", ""),
        "threshold": summary["threshold"],
        "val_auc": val["auc"],
        "val_accuracy": val["accuracy"],
        "test_auc": test["auc"],
        "test_accuracy": test["accuracy"],
        "test_f1": test["f1"],
        "test_eer": test["eer"],
        "tar_at_far_1pct": test["tar_at_far_1pct"],
        "tar_at_far_0_1pct": test["tar_at_far_0_1pct"],
        "negative_same_breed_accuracy": test.get("negative_same_breed_accuracy", 0.0),
        "negative_diff_breed_accuracy": test.get("negative_diff_breed_accuracy", 0.0),
        "positive_accuracy": test.get("positive_accuracy", 0.0),
    }


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def run_dog_verification_sweep(config: DogVerificationSweepConfig) -> Dict[str, object]:
    import pandas as pd

    results_root = ensure_dir(config.paths.results_dir / config.run_name)
    if config.dry_run:
        summary = {
            "mode": "dry_run",
            "config": _jsonable(asdict(config)),
            "losses": list(config.losses),
            "seeds": list(config.seeds),
        }
        save_json(summary, results_root / "dry_run.json")
        return summary
    train_rows, val_pairs, test_pairs = _prepare_data(config)
    summaries: List[Dict[str, object]] = [run_frozen_embedding_baseline(config)]
    for loss_name in config.losses:
        for seed in config.seeds:
            summaries.append(
                run_single_verification_experiment(
                    config,
                    loss_name,
                    seed,
                    train_rows=train_rows,
                    val_pairs=val_pairs,
                    test_pairs=test_pairs,
                )
            )
    leaderboard = pd.DataFrame([_leaderboard_row(summary) for summary in summaries])
    leaderboard = leaderboard.sort_values(
        ["val_auc", "negative_same_breed_accuracy", "test_auc"],
        ascending=False,
    )
    leaderboard.to_csv(results_root / "verification_leaderboard.csv", index=False)
    aggregate = leaderboard[leaderboard["loss"] != "FrozenCosine"].groupby("loss").agg(
        val_auc_mean=("val_auc", "mean"),
        val_auc_std=("val_auc", "std"),
        same_breed_acc_mean=("negative_same_breed_accuracy", "mean"),
        test_auc_mean=("test_auc", "mean"),
        runs=("experiment_name", "count"),
    ).reset_index().sort_values(["val_auc_mean", "same_breed_acc_mean"], ascending=False)
    aggregate.to_csv(results_root / "verification_aggregate.csv", index=False)
    best_losses = aggregate.head(2)["loss"].tolist()
    final_summary = {
        "run_name": config.run_name,
        "train_rows": len(train_rows),
        "val_pairs": len(val_pairs),
        "test_pairs": len(test_pairs),
        "leaderboard_csv": str(results_root / "verification_leaderboard.csv"),
        "aggregate_csv": str(results_root / "verification_aggregate.csv"),
        "best_candidate_losses": best_losses,
        "results_root": str(results_root),
    }
    save_json(final_summary, results_root / "verification_summary.json")
    return final_summary


def run_promoted_robust_sweep(
    *,
    candidate_summary_json: Path,
    paths: DogVerificationPaths = DogVerificationPaths(),
    seeds: Sequence[int] = FINAL_SEEDS,
    epochs: int = 20,
    train_identity_fraction: float = 1.0,
    pair_fraction: float = 1.0,
) -> Dict[str, object]:
    payload = json.loads(candidate_summary_json.read_text(encoding="utf-8"))
    losses = payload["best_candidate_losses"]
    config = DogVerificationSweepConfig(
        paths=paths,
        run_name="robust_promoted_candidates",
        train_identity_fraction=train_identity_fraction,
        pair_fraction=pair_fraction,
        losses=losses,
        seeds=seeds,
        epochs=epochs,
    )
    return run_dog_verification_sweep(config)
