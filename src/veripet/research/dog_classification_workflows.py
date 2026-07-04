"""Dog breed classification workflows for Stanford Dogs experiments."""

from __future__ import annotations

import csv
import io
import json
import random
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .io import ensure_dir, save_csv, save_json

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
FINAL_SEEDS = [7, 13, 42, 101, 2026]


@dataclass(frozen=True)
class StanfordDogsPaths:
    base_dir: Path = Path("/content/drive/MyDrive/classification_exp")
    local_data_dir: Path = Path("/content/local_data")
    results_dir: Path = Path("/content/drive/MyDrive/classification_exp/dog_optuna_results")

    @property
    def images_root(self) -> Path:
        local = self.local_data_dir / "Images"
        return local if local.exists() else self.base_dir / "Images"

    @property
    def annotations_root(self) -> Path:
        local = self.local_data_dir / "Annotation"
        return local if local.exists() else self.base_dir / "Annotation"


@dataclass(frozen=True)
class DogClassificationOptunaConfig:
    paths: StanfordDogsPaths = field(default_factory=StanfordDogsPaths)
    n_trials: int = 25
    epochs: int = 8
    seed: int = 42
    batch_size_default: int = 128
    input_size: int = 224
    num_workers: int = 4
    use_amp: bool = True
    cache_images: bool = True
    study_name: str = "dog_convnext_tiny_classification_optuna"
    storage: Optional[str] = None
    dry_run: bool = False
    smoke: bool = False
    early_stopping_patience: int = 3
    smoke_fraction: float = 0.10
    smoke_min_per_breed: int = 2
    smoke_max_per_breed: int = 12


def recommended_search_space() -> Dict[str, object]:
    return {
        "learning_rate": {"type": "loguniform", "low": 2e-5, "high": 5e-4},
        "weight_decay": {"type": "loguniform", "low": 1e-6, "high": 5e-3},
        "label_smoothing": {"type": "float", "low": 0.0, "high": 0.15, "step": 0.025},
        "dropout": {"type": "float", "low": 0.0, "high": 0.3, "step": 0.05},
        "stochastic_depth": {"type": "float", "low": 0.0, "high": 0.2, "step": 0.05},
        "batch_size": {"type": "categorical", "choices": [64, 96, 128, 192]},
        "crop_scale_min": {"type": "float", "low": 0.65, "high": 0.9, "step": 0.05},
        "color_jitter": {"type": "float", "low": 0.0, "high": 0.3, "step": 0.05},
        "rotation_degrees": {"type": "categorical", "choices": [0, 10, 15]},
        "random_erasing": {"type": "float", "low": 0.0, "high": 0.25, "step": 0.05},
        "scheduler": {"type": "categorical", "choices": ["cosine", "plateau"]},
        "freeze_policy": {"type": "categorical", "choices": ["none", "head_only_1_epoch"]},
    }


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


def parse_annotation(annotation_path: Path) -> Optional[Tuple[int, int, int, int]]:
    try:
        xmin = ymin = xmax = ymax = None
        for _event, elem in ET.iterparse(annotation_path, events=("end",)):
            if elem.tag == "xmin":
                xmin = int(elem.text or 0)
            elif elem.tag == "ymin":
                ymin = int(elem.text or 0)
            elif elem.tag == "xmax":
                xmax = int(elem.text or 0)
            elif elem.tag == "ymax":
                ymax = int(elem.text or 0)
            elem.clear()
        if None not in (xmin, ymin, xmax, ymax):
            return int(xmin), int(ymin), int(xmax), int(ymax)
    except Exception:
        return None
    return None


def discover_stanford_dogs(paths: StanfordDogsPaths) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for breed_dir in sorted(path for path in paths.images_root.iterdir() if path.is_dir()):
        breed = breed_dir.name.split("-", 1)[1] if "-" in breed_dir.name else breed_dir.name
        breed = breed.replace("_", " ")
        annotation_dir = paths.annotations_root / breed_dir.name
        for image_path in sorted(breed_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            annotation_path = annotation_dir / image_path.stem
            bbox = parse_annotation(annotation_path) if annotation_path.exists() else None
            rows.append(
                {
                    "filename": str(image_path),
                    "breed": breed,
                    "bbox": bbox,
                }
            )
    return rows


def stratified_split_rows(
    rows: Sequence[Dict[str, object]],
    seed: int = 42,
) -> Dict[str, List[Dict[str, object]]]:
    from sklearn.model_selection import train_test_split

    labels = [str(row["breed"]) for row in rows]
    indices = list(range(len(rows)))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, stratify=labels, random_state=seed
    )
    temp_labels = [labels[index] for index in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=temp_labels, random_state=seed
    )
    return {
        "train": [dict(rows[index]) for index in train_idx],
        "val": [dict(rows[index]) for index in val_idx],
        "test": [dict(rows[index]) for index in test_idx],
    }


def sample_split_rows(
    split_rows: Dict[str, List[Dict[str, object]]],
    *,
    fraction: float,
    seed: int,
    min_per_breed: int = 1,
    max_per_breed: Optional[int] = None,
) -> Dict[str, List[Dict[str, object]]]:
    if fraction >= 1.0 and max_per_breed is None:
        return {split: [dict(row) for row in rows] for split, rows in split_rows.items()}

    rng = random.Random(seed)
    sampled: Dict[str, List[Dict[str, object]]] = {}
    for split, rows in split_rows.items():
        grouped: Dict[str, List[Dict[str, object]]] = {}
        for row in rows:
            grouped.setdefault(str(row["breed"]), []).append(dict(row))
        split_sample: List[Dict[str, object]] = []
        for breed, breed_rows in grouped.items():
            breed_rows = list(breed_rows)
            rng.shuffle(breed_rows)
            keep = max(min_per_breed, int(round(len(breed_rows) * fraction)))
            if max_per_breed is not None:
                keep = min(keep, max_per_breed)
            keep = min(len(breed_rows), keep)
            split_sample.extend(breed_rows[:keep])
        rng.shuffle(split_sample)
        sampled[split] = split_sample
    return sampled


def build_transforms(
    input_size: int,
    crop_scale_min: float = 0.8,
    color_jitter: float = 0.2,
    rotation_degrees: int = 15,
    random_erasing: float = 0.0,
) -> Dict[str, object]:
    from torchvision import transforms

    train_steps: List[object] = [
        transforms.RandomResizedCrop(input_size, scale=(crop_scale_min, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    if color_jitter > 0:
        train_steps.append(
            transforms.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
                hue=min(0.1, color_jitter / 3.0),
            )
        )
    if rotation_degrees > 0:
        train_steps.append(transforms.RandomRotation(rotation_degrees))
    train_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    if random_erasing > 0:
        train_steps.append(transforms.RandomErasing(p=random_erasing, scale=(0.02, 0.12)))
    return {
        "train": transforms.Compose(train_steps),
        "eval": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        ),
    }


class StanfordDogsDataset:
    def __init__(
        self,
        rows: Sequence[Dict[str, object]],
        label_to_idx: Dict[str, int],
        transform: object,
        cache_images: bool = True,
        use_bbox: bool = True,
    ) -> None:
        self.rows = list(rows)
        self.label_to_idx = label_to_idx
        self.transform = transform
        self.use_bbox = use_bbox
        self._cache: Optional[List[bytes]] = None
        if cache_images:
            self._cache = [Path(str(row["filename"])).read_bytes() for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        from PIL import Image

        row = self.rows[index]
        if self._cache is None:
            image = Image.open(str(row["filename"])).convert("RGB")
        else:
            image = Image.open(io.BytesIO(self._cache[index])).convert("RGB")
        bbox = row.get("bbox")
        if self.use_bbox and bbox is not None:
            xmin, ymin, xmax, ymax = bbox  # type: ignore[misc]
            width, height = image.size
            margin_x = int((xmax - xmin) * 0.10)
            margin_y = int((ymax - ymin) * 0.10)
            image = image.crop(
                (
                    max(0, xmin - margin_x),
                    max(0, ymin - margin_y),
                    min(width, xmax + margin_x),
                    min(height, ymax + margin_y),
                )
            )
        return self.transform(image), self.label_to_idx[str(row["breed"])]


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def create_convnext_tiny_classifier(num_classes: int, dropout: float = 0.0, stochastic_depth: float = 0.0):
    import torch.nn as nn
    from torchvision import models

    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    if stochastic_depth > 0:
        for module in model.modules():
            if module.__class__.__name__ == "StochasticDepth" and hasattr(module, "p"):
                module.p = stochastic_depth
    in_features = model.classifier[2].in_features
    head: nn.Module = nn.Linear(in_features, num_classes)
    if dropout > 0:
        head = nn.Sequential(nn.Dropout(dropout), head)
    model.classifier[2] = head
    return model


def _set_head_only(model: object, head_only: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = not head_only
    if head_only:
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True


def _evaluate_classifier(model: object, loader: object, device: object, use_amp: bool) -> Tuple[float, List[int], List[int]]:
    import torch
    from torch.amp import autocast

    model.eval()
    correct = 0
    total = 0
    preds: List[int] = []
    labels_all: List[int] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
                logits = model(images)
            batch_preds = logits.argmax(dim=1)
            correct += int((batch_preds == labels).sum().item())
            total += int(labels.numel())
            preds.extend(batch_preds.detach().cpu().tolist())
            labels_all.extend(labels.detach().cpu().tolist())
    return correct / max(total, 1), labels_all, preds


def _train_classifier_once(
    *,
    split_rows: Dict[str, List[Dict[str, object]]],
    params: Dict[str, object],
    seed: int,
    epochs: int,
    input_size: int,
    num_workers: int,
    use_amp: bool,
    cache_images: bool,
    results_dir: Path,
    run_name: str,
    patience: int = 5,
) -> Dict[str, object]:
    import torch
    import torch.nn as nn
    from sklearn.metrics import accuracy_score, classification_report
    from torch.amp import GradScaler, autocast
    from torch.utils.data import DataLoader

    set_seed(seed)
    device = _device()
    breeds = sorted({str(row["breed"]) for rows in split_rows.values() for row in rows})
    label_to_idx = {breed: index for index, breed in enumerate(breeds)}
    transforms_by_split = build_transforms(
        input_size=input_size,
        crop_scale_min=float(params.get("crop_scale_min", 0.8)),
        color_jitter=float(params.get("color_jitter", 0.2)),
        rotation_degrees=int(params.get("rotation_degrees", 15)),
        random_erasing=float(params.get("random_erasing", 0.0)),
    )
    batch_size = int(params.get("batch_size", 128))
    train_loader = DataLoader(
        StanfordDogsDataset(split_rows["train"], label_to_idx, transforms_by_split["train"], cache_images),
        **_loader_kwargs(batch_size, num_workers, shuffle=True, drop_last=True),
    )
    val_loader = DataLoader(
        StanfordDogsDataset(split_rows["val"], label_to_idx, transforms_by_split["eval"], cache_images),
        **_loader_kwargs(max(batch_size, 128), num_workers, shuffle=False),
    )
    test_loader = DataLoader(
        StanfordDogsDataset(split_rows["test"], label_to_idx, transforms_by_split["eval"], cache_images),
        **_loader_kwargs(max(batch_size, 128), num_workers, shuffle=False),
    )

    model = create_convnext_tiny_classifier(
        num_classes=len(label_to_idx),
        dropout=float(params.get("dropout", 0.0)),
        stochastic_depth=float(params.get("stochastic_depth", 0.0)),
    ).to(device)
    freeze_policy = str(params.get("freeze_policy", "none"))
    freeze_epochs = 1 if freeze_policy == "head_only_1_epoch" else 0
    _set_head_only(model, freeze_epochs > 0)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(params.get("label_smoothing", 0.0)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params.get("learning_rate", 1e-4)),
        weight_decay=float(params.get("weight_decay", 1e-4)),
    )
    scheduler_name = str(params.get("scheduler", "cosine"))
    if scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-6)
    scaler = GradScaler(enabled=use_amp and device.type == "cuda")

    best_state = None
    best_val_acc = -1.0
    best_epoch = 0
    stale = 0
    history: List[Dict[str, object]] = []
    for epoch in range(1, epochs + 1):
        if freeze_epochs and epoch == freeze_epochs + 1:
            _set_head_only(model, False)
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_count = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.item()) * images.size(0)
            train_correct += int((logits.argmax(dim=1) == labels).sum().item())
            train_count += int(images.size(0))
        val_acc, _val_labels, _val_preds = _evaluate_classifier(model, val_loader, device, use_amp)
        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(train_count, 1),
            "train_accuracy": train_correct / max(train_count, 1),
            "val_accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        if scheduler_name == "plateau":
            scheduler.step(val_acc)
        else:
            scheduler.step()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a classifier checkpoint")
    model.load_state_dict(best_state)
    test_acc, test_labels, test_preds = _evaluate_classifier(model, test_loader, device, use_amp)
    report = classification_report(test_labels, test_preds, target_names=breeds, output_dict=True, zero_division=0)
    run_dir = ensure_dir(results_dir / run_name)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "label_to_idx": label_to_idx,
            "params": params,
            "seed": seed,
        },
        run_dir / "best_classifier.pt",
    )
    save_csv(history, run_dir / "training_history.csv")
    save_json(
        {
            "run_name": run_name,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_acc,
            "test_accuracy": test_acc,
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"],
            "params": params,
        },
        run_dir / "summary.json",
    )
    return {
        "run_name": run_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_acc,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "results_dir": str(run_dir),
    }


def _suggest_params(trial: object) -> Dict[str, object]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 2e-5, 5e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 5e-3, log=True),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.15, step=0.025),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3, step=0.05),
        "stochastic_depth": trial.suggest_float("stochastic_depth", 0.0, 0.2, step=0.05),
        "batch_size": trial.suggest_categorical("batch_size", [64, 96, 128, 192]),
        "crop_scale_min": trial.suggest_float("crop_scale_min", 0.65, 0.9, step=0.05),
        "color_jitter": trial.suggest_float("color_jitter", 0.0, 0.3, step=0.05),
        "rotation_degrees": trial.suggest_categorical("rotation_degrees", [0, 10, 15]),
        "random_erasing": trial.suggest_float("random_erasing", 0.0, 0.25, step=0.05),
        "scheduler": trial.suggest_categorical("scheduler", ["cosine", "plateau"]),
        "freeze_policy": trial.suggest_categorical("freeze_policy", ["none", "head_only_1_epoch"]),
    }


def run_dog_classification_optuna(config: DogClassificationOptunaConfig) -> Dict[str, object]:
    results_dir = ensure_dir(config.paths.results_dir / "optuna_convnext_tiny")
    if config.dry_run:
        summary = {
            "mode": "dry_run",
            "config": {key: str(value) for key, value in asdict(config).items()},
            "search_space": recommended_search_space(),
            "results_dir": str(results_dir),
        }
        save_json(summary, results_dir / "optuna_dry_run.json")
        return summary

    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Optuna is required. In Colab, run: !pip install -q optuna") from exc

    rows = discover_stanford_dogs(config.paths)
    split_rows = stratified_split_rows(rows, seed=42)
    if config.smoke:
        split_rows = sample_split_rows(
            split_rows,
            fraction=config.smoke_fraction,
            seed=config.seed,
            min_per_breed=config.smoke_min_per_breed,
            max_per_breed=config.smoke_max_per_breed,
        )

    def objective(trial: object) -> float:
        params = _suggest_params(trial)
        summary = _train_classifier_once(
            split_rows=split_rows,
            params=params,
            seed=config.seed + int(trial.number),
            epochs=1 if config.smoke else config.epochs,
            input_size=config.input_size,
            num_workers=config.num_workers,
            use_amp=config.use_amp,
            cache_images=config.cache_images,
            results_dir=results_dir,
            run_name=f"trial_{int(trial.number):03d}",
            patience=1 if config.smoke else config.early_stopping_patience,
        )
        trial.set_user_attr("best_epoch", summary["best_epoch"])
        trial.set_user_attr("macro_f1", summary["macro_f1"])
        return float(summary["best_val_accuracy"])

    study = optuna.create_study(
        study_name=config.study_name,
        storage=config.storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=1 if config.smoke else config.n_trials)
    trials_df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
    trials_df.to_csv(results_dir / "optuna_trials.csv", index=False)
    summary = {
        "mode": "smoke" if config.smoke else "search",
        "best_value": study.best_value,
        "best_trial": study.best_trial.number,
        "best_params": dict(study.best_trial.params),
        "split_sizes": {split: len(rows) for split, rows in split_rows.items()},
        "trials_csv": str(results_dir / "optuna_trials.csv"),
        "results_dir": str(results_dir),
    }
    save_json(summary, results_dir / "optuna_best_params.json")
    return summary


def run_svm_embedding_baseline(
    *,
    paths: StanfordDogsPaths = StanfordDogsPaths(),
    seed: int = 42,
    input_size: int = 224,
    batch_size: int = 256,
    num_workers: int = 4,
    cache_images: bool = True,
) -> Dict[str, object]:
    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    from torch.utils.data import DataLoader
    from torchvision import models

    set_seed(seed)
    rows = discover_stanford_dogs(paths)
    split_rows = stratified_split_rows(rows, seed=42)
    breeds = sorted({str(row["breed"]) for row in rows})
    label_to_idx = {breed: index for index, breed in enumerate(breeds)}
    transforms_by_split = build_transforms(input_size)
    device = _device()
    feature_model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    feature_model.classifier[2] = torch.nn.Identity()
    feature_model = feature_model.to(device).eval()

    def extract(split: str) -> Tuple[np.ndarray, np.ndarray]:
        loader = DataLoader(
            StanfordDogsDataset(split_rows[split], label_to_idx, transforms_by_split["eval"], cache_images),
            **_loader_kwargs(batch_size, num_workers, shuffle=False),
        )
        features: List[np.ndarray] = []
        labels: List[int] = []
        with torch.no_grad():
            for images, batch_labels in loader:
                images = images.to(device, non_blocking=True)
                batch_features = feature_model(images).detach().cpu().numpy()
                features.append(batch_features)
                labels.extend(batch_labels.numpy().tolist())
        return np.concatenate(features, axis=0), np.asarray(labels)

    x_train, y_train = extract("train")
    x_val, y_val = extract("val")
    x_test, y_test = extract("test")
    model = make_pipeline(
        StandardScaler(),
        LinearSVC(C=1.0, max_iter=5000, random_state=seed),
    )
    model.fit(x_train, y_train)
    val_preds = model.predict(x_val)
    test_preds = model.predict(x_test)
    report = classification_report(y_test, test_preds, target_names=breeds, output_dict=True, zero_division=0)
    results_dir = ensure_dir(paths.results_dir / "svm_embedding_baseline")
    summary = {
        "baseline": "convnext_tiny_imagenet_embeddings_linear_svm",
        "seed": seed,
        "val_accuracy": accuracy_score(y_val, val_preds),
        "test_accuracy": accuracy_score(y_test, test_preds),
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "results_dir": str(results_dir),
    }
    save_json(summary, results_dir / "summary.json")
    return summary


def run_multiseed_final_from_params(
    *,
    best_params_json: Path,
    paths: StanfordDogsPaths = StanfordDogsPaths(),
    seeds: Sequence[int] = FINAL_SEEDS,
    epochs: int = 50,
    input_size: int = 224,
    num_workers: int = 4,
    use_amp: bool = True,
    cache_images: bool = True,
) -> Dict[str, object]:
    started = time.time()
    best_payload = json.loads(best_params_json.read_text(encoding="utf-8"))
    params = dict(best_payload["best_params"])
    rows = discover_stanford_dogs(paths)
    split_rows = stratified_split_rows(rows, seed=42)
    results_dir = ensure_dir(paths.results_dir / "final_multiseed")
    summaries = []
    for seed in seeds:
        summaries.append(
            _train_classifier_once(
                split_rows=split_rows,
                params=params,
                seed=seed,
                epochs=epochs,
                input_size=input_size,
                num_workers=num_workers,
                use_amp=use_amp,
                cache_images=cache_images,
                results_dir=results_dir,
                run_name=f"seed_{seed}",
                patience=8,
            )
        )
    save_csv(summaries, results_dir / "classification_multiseed_results.csv")
    test_accs = [float(row["test_accuracy"]) for row in summaries]
    macro_f1s = [float(row["macro_f1"]) for row in summaries]
    aggregate = {
        "seeds": list(seeds),
        "params": params,
        "test_accuracy_mean": sum(test_accs) / len(test_accs),
        "test_accuracy_min": min(test_accs),
        "test_accuracy_max": max(test_accs),
        "macro_f1_mean": sum(macro_f1s) / len(macro_f1s),
        "results_csv": str(results_dir / "classification_multiseed_results.csv"),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    save_json(aggregate, results_dir / "classification_multiseed_summary.json")
    return aggregate
