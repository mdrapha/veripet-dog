"""Simple transform presets for research notebooks."""

from __future__ import annotations

from typing import Dict


def get_classification_transforms(input_size: int = 224) -> Dict[str, Dict[str, object]]:
    return {
        "train": {
            "resize": input_size,
            "horizontal_flip": True,
            "color_jitter": True,
            "normalize": "imagenet",
        },
        "val": {"resize": input_size, "center_crop": True, "normalize": "imagenet"},
    }


def get_verification_transforms(input_size: int = 224) -> Dict[str, Dict[str, object]]:
    return {
        "train": {
            "resize": input_size,
            "horizontal_flip": True,
            "random_erasing": True,
            "normalize": "imagenet",
        },
        "val": {"resize": input_size, "center_crop": True, "normalize": "imagenet"},
    }
