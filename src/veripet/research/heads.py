"""Metric-learning head registry."""

from __future__ import annotations

from dataclasses import dataclass


AVAILABLE_HEADS = {"Softmax", "ArcFace", "CosFace", "AdaFace", "MagFace"}


@dataclass(frozen=True)
class HeadSpec:
    name: str
    embedding_dim: int
    num_classes: int
    margin: float = 0.5
    scale: float = 30.0


def create_head(
    name: str,
    embedding_dim: int = 512,
    num_classes: int = 2,
    margin: float = 0.5,
    scale: float = 30.0,
) -> HeadSpec:
    if name not in AVAILABLE_HEADS:
        raise ValueError(f"Unsupported head: {name}")
    return HeadSpec(
        name=name,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        margin=margin,
        scale=scale,
    )
