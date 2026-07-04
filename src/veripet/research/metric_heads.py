"""Metric-learning heads used by dog verification experiments.

The implementations are intentionally local and dependency-free. They follow
the public formulations of ArcFace/CosFace/AdaFace/MagFace closely enough for
controlled VeriPet sweeps while keeping the Colab notebooks self-contained.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MetricHeadOutput:
    logits: torch.Tensor
    regularization: torch.Tensor


def l2_norm(input_tensor: torch.Tensor, axis: int = 1) -> torch.Tensor:
    norm = torch.norm(input_tensor, 2, axis, keepdim=True).clamp_min(1e-12)
    return input_tensor / norm


class LinearSoftmaxHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        norms: Optional[torch.Tensor] = None,
    ) -> MetricHeadOutput:
        logits = self.classifier(embeddings)
        return MetricHeadOutput(logits=logits, regularization=embeddings.new_tensor(0.0))


class ArcFaceHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        margin: float = 0.5,
        scale: float = 64.0,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(embedding_dim, num_classes))
        nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale = scale
        self.eps = 1e-7

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        norms: Optional[torch.Tensor] = None,
    ) -> MetricHeadOutput:
        cosine = torch.mm(l2_norm(embeddings), l2_norm(self.weight, axis=0)).clamp(
            -1 + self.eps, 1 - self.eps
        )
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        theta = cosine.acos()
        target = torch.cos((theta + one_hot * self.margin).clamp(self.eps, math.pi - self.eps))
        logits = (one_hot * target + (1.0 - one_hot) * cosine) * self.scale
        return MetricHeadOutput(logits=logits, regularization=embeddings.new_tensor(0.0))


class CosFaceHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        margin: float = 0.35,
        scale: float = 64.0,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(embedding_dim, num_classes))
        nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale = scale
        self.eps = 1e-7

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        norms: Optional[torch.Tensor] = None,
    ) -> MetricHeadOutput:
        cosine = torch.mm(l2_norm(embeddings), l2_norm(self.weight, axis=0)).clamp(
            -1 + self.eps, 1 - self.eps
        )
        margin_hot = torch.zeros_like(cosine)
        margin_hot.scatter_(1, labels.view(-1, 1), self.margin)
        logits = (cosine - margin_hot) * self.scale
        return MetricHeadOutput(logits=logits, regularization=embeddings.new_tensor(0.0))


class AdaFaceHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        margin: float = 0.4,
        scale: float = 64.0,
        h: float = 0.333,
        t_alpha: float = 0.01,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(embedding_dim, num_classes))
        nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale = scale
        self.h = h
        self.t_alpha = t_alpha
        self.eps = 1e-7
        self.register_buffer("batch_mean", torch.ones(1) * 20.0)
        self.register_buffer("batch_std", torch.ones(1) * 100.0)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        norms: Optional[torch.Tensor] = None,
    ) -> MetricHeadOutput:
        if norms is None:
            norms = embeddings.detach().norm(dim=1, keepdim=True)
        safe_norms = norms.detach().clamp(min=1e-3, max=100.0)
        if self.training:
            with torch.no_grad():
                mean = safe_norms.mean()
                std = safe_norms.std(unbiased=False).clamp_min(self.eps)
                self.batch_mean.mul_(1.0 - self.t_alpha).add_(mean * self.t_alpha)
                self.batch_std.mul_(1.0 - self.t_alpha).add_(std * self.t_alpha)

        margin_scaler = ((safe_norms - self.batch_mean) / (self.batch_std + self.eps))
        margin_scaler = (margin_scaler * self.h).clamp(-1.0, 1.0)

        cosine = torch.mm(l2_norm(embeddings), l2_norm(self.weight, axis=0)).clamp(
            -1 + self.eps, 1 - self.eps
        )
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        angular_margin = -self.margin * margin_scaler
        theta = cosine.acos()
        theta_m = (theta + one_hot * angular_margin).clamp(self.eps, math.pi - self.eps)
        cosine = theta_m.cos()

        additive_margin = self.margin + self.margin * margin_scaler
        logits = (cosine - one_hot * additive_margin) * self.scale
        return MetricHeadOutput(logits=logits, regularization=embeddings.new_tensor(0.0))


class MagFaceHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        lower_margin: float = 0.45,
        upper_margin: float = 0.8,
        lower_norm: float = 10.0,
        upper_norm: float = 110.0,
        regularizer_weight: float = 20.0,
        scale: float = 64.0,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(embedding_dim, num_classes))
        nn.init.xavier_uniform_(self.weight)
        self.lower_margin = lower_margin
        self.upper_margin = upper_margin
        self.lower_norm = lower_norm
        self.upper_norm = upper_norm
        self.regularizer_weight = regularizer_weight
        self.scale = scale
        self.eps = 1e-7

    def _adaptive_margin(self, norms: torch.Tensor) -> torch.Tensor:
        clipped = norms.clamp(self.lower_norm, self.upper_norm)
        ratio = (clipped - self.lower_norm) / (self.upper_norm - self.lower_norm)
        return self.lower_margin + ratio * (self.upper_margin - self.lower_margin)

    def _regularizer(self, norms: torch.Tensor) -> torch.Tensor:
        clipped = norms.clamp(self.lower_norm, self.upper_norm)
        return self.regularizer_weight / clipped + clipped / self.regularizer_weight

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        norms: Optional[torch.Tensor] = None,
    ) -> MetricHeadOutput:
        if norms is None:
            norms = embeddings.norm(dim=1, keepdim=True)
        margins = self._adaptive_margin(norms.detach())
        cosine = torch.mm(l2_norm(embeddings), l2_norm(self.weight, axis=0)).clamp(
            -1 + self.eps, 1 - self.eps
        )
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        theta = cosine.acos()
        target = torch.cos((theta + one_hot * margins).clamp(self.eps, math.pi - self.eps))
        logits = (one_hot * target + (1.0 - one_hot) * cosine) * self.scale
        return MetricHeadOutput(
            logits=logits,
            regularization=self._regularizer(norms).mean(),
        )


def build_metric_head(name: str, embedding_dim: int, num_classes: int) -> Optional[nn.Module]:
    normalized = name.lower()
    if normalized == "softmax":
        return LinearSoftmaxHead(embedding_dim, num_classes)
    if normalized == "arcface":
        return ArcFaceHead(embedding_dim, num_classes)
    if normalized == "cosface":
        return CosFaceHead(embedding_dim, num_classes)
    if normalized == "adaface":
        return AdaFaceHead(embedding_dim, num_classes)
    if normalized == "magface":
        return MagFaceHead(embedding_dim, num_classes)
    if normalized == "tripletmarginloss":
        return None
    raise ValueError(f"Unsupported metric head: {name}")
