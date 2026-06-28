"""Backbone factory for verification and classification experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


AVAILABLE_BACKBONES: Dict[str, Dict[str, object]] = {
    "convnext_small.fb_in22k_ft_in1k": {"family": "convnext_small", "embedding_dim": 768},
    "efficientnet_b3.ra2_in1k": {"family": "efficientnet_b3", "embedding_dim": 1536},
    "swin_tiny_patch4_window7_224": {"family": "swin_tiny", "embedding_dim": 768},
    "resnet101.a1_in1k": {"family": "resnet101", "embedding_dim": 2048},
}


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    family: str
    pretrained: bool
    output_dim: int


def create_backbone(name: str, pretrained: bool = True) -> BackboneSpec:
    if name not in AVAILABLE_BACKBONES:
        raise ValueError(f"Unsupported backbone: {name}")
    meta = AVAILABLE_BACKBONES[name]
    return BackboneSpec(
        name=name,
        family=str(meta["family"]),
        pretrained=pretrained,
        output_dim=int(meta["embedding_dim"]),
    )
