"""
das_anomaly.train
~~~~~~~~~~~~~~~~~

Convenience re-exports for training utilities.
"""
from __future__ import annotations

from .autoencoder import AutoencoderTrainer, TrainAEConfig
from .split_images import ImageSplitter, TrainSplitConfig

__all__: list[str] = [
    "TrainSplitConfig",
    "ImageSplitter",
    "TrainAEConfig",
    "AutoencoderTrainer",
]
