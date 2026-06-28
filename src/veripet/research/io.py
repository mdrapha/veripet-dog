"""Filesystem helpers for experiments and artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(payload: Mapping[str, object], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def save_csv(rows: Sequence[Mapping[str, object]], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint_bytes(blob: bytes, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    output_path.write_bytes(blob)
