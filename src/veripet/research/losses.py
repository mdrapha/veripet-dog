"""Loss registry for research experiments."""

from __future__ import annotations

from dataclasses import dataclass


AVAILABLE_LOSSES = {"Softmax", "ArcFace", "CosFace", "AdaFace", "TripletMarginLoss"}


@dataclass(frozen=True)
class LossSpec:
    name: str
    margin: float = 0.2
    miner: str = "batch_hard"


def create_loss(name: str, margin: float = 0.2) -> LossSpec:
    if name not in AVAILABLE_LOSSES:
        raise ValueError(f"Unsupported loss: {name}")
    miner = "batch_hard" if name == "TripletMarginLoss" else "classifier_head"
    return LossSpec(name=name, margin=margin, miner=miner)
