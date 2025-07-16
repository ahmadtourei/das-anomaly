"""
das_anomaly.train.split_images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Randomly split PSD PNGs into *train* and *test* folders.

Example
-------
>>> from das_anomaly.train import TrainSplitConfig, ImageSplitter
>>> cfg = TrainSplitConfig()
>>> ImageSplitter(cfg).run()
"""

from __future__ import annotations

import random
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from das_anomaly.settings import SETTINGS


@dataclass
class TrainSplitConfig:
    """Configuration knobs for the PNG train/test splitter."""

    # source & destination folders
    psd_dir: Path | str = SETTINGS.PSD_PATH
    train_dir: Path | str = SETTINGS.TRAIN_IMAGES_PATH
    test_dir: Path | str = SETTINGS.TEST_IMAGES_PATH
    anomaly_dir: Path | str = SETTINGS.ANOMALY_IMAGES_PATH

    # sampling parameters
    num_images: int = SETTINGS.NUM_IMAGE
    ratio: float = SETTINGS.RATIO

    rng_seed: int | None = 42 * num_images  # reproducible splits

    def __post_init__(self):
        self.psd_dir = Path(self.psd_dir).expanduser()
        self.train_dir = Path(self.train_dir).expanduser()
        self.test_dir = Path(self.test_dir).expanduser()
        self.anomaly_dir = Path(self.anomaly_dir).expanduser()

        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.anomaly_dir.mkdir(parents=True, exist_ok=True)


class ImageSplitter:
    """Copy-based splitter for PNG files."""

    def __init__(self, cfg: TrainSplitConfig):
        self.cfg = cfg
        if self.cfg.ratio < 0 or self.cfg.ratio > 1:
            raise ValueError("ratio must be 0 <= ratio <= 1")

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        """Randomly copy PNGs into train/test dirs according to cfg."""
        train, test = self._pick_files()

        # make …/images/ sub-dirs once
        train_img_dir = (self.cfg.train_dir / "images").resolve()
        test_img_dir = (self.cfg.test_dir / "images").resolve()
        anomaly_img_dir = (self.cfg.anomaly_dir / "images").resolve()
        train_img_dir.mkdir(parents=True, exist_ok=True)
        test_img_dir.mkdir(parents=True, exist_ok=True)
        anomaly_img_dir.mkdir(parents=True, exist_ok=True)

        # copy
        self._copy_all(train, train_img_dir)
        self._copy_all(test, test_img_dir)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _pick_files(self) -> tuple[Sequence[Path], Sequence[Path]]:
        pngs = list(self.cfg.psd_dir.rglob("*.png"))
        if len(pngs) < self.cfg.num_images:
            raise ValueError(
                f"Only {len(pngs)} PNG files found, need {self.cfg.num_images}."
            )

        rng = random.Random(self.cfg.rng_seed)
        selected = rng.sample(pngs, self.cfg.num_images)

        n_test = int(self.cfg.num_images * self.cfg.ratio)
        # train / test lists that also handle the edge-cases ratio == 0 or 1
        if n_test == 0:  # 100 % train, 0 % test
            return selected, []
        elif n_test == self.cfg.num_images:  # 0 % train, 100 % test
            return [], selected
        else:  # 0 < ratio < 1
            return selected[:-n_test], selected[-n_test:]

    @staticmethod
    def _copy_all(files: Sequence[Path], dest: Path) -> None:
        for src in files:
            shutil.copy(src, dest)
