"""
das_anomaly.train.autoencoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Train a convolutional autoencoder on PSD images for anomaly detection.

Example
-------
>>> from das_anomaly.train import TrainAEConfig, AutoencoderTrainer
>>> cfg = TrainAEConfig(num_epoch=100, ratio=0.2)
>>> AutoencoderTrainer(cfg).run()          # fits and saves model and plots
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from das_anomaly import decoder, encoder, plot_train_test_loss
from das_anomaly.settings import SETTINGS


# --------------------------------------------------------------------- #
# configuration                                                         #
# --------------------------------------------------------------------- #
@dataclass
class TrainAEConfig:
    """All knobs for autoencoder training (mirrors SETTINGS defaults)."""

    # directories 
    train_dir: Path | str = SETTINGS.TRAIN_IMAGES_PATH
    test_dir: Path | str = SETTINGS.TEST_IMAGES_PATH
    out_dir: Path | str = SETTINGS.TRAINED_PATH

    # data & training parameters 
    img_size: int = SETTINGS.SIZE
    batch_size: int = SETTINGS.BATCH_SIZE
    num_epoch: int = SETTINGS.NUM_EPOCH

    # augmentation / preprocessing 
    rescale: float = 1.0 / 255

    # random seed for reproducibility 
    seed: Optional[int] = 42

    def __post_init__(self):
        self.train_dir = Path(self.train_dir).expanduser()
        self.test_dir = Path(self.test_dir).expanduser()
        self.out_dir = Path(self.out_dir).expanduser()
        self.out_dir.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------- #
# trainer                                                               #
# --------------------------------------------------------------------- #
class AutoencoderTrainer:
    """Orchestrates data I/O, model creation, fitting, and persistence."""

    def __init__(self, cfg: TrainAEConfig):
        self.cfg = cfg

        self._rng = np.random.default_rng(self.cfg.seed)
        tf.keras.utils.set_random_seed(self.cfg.seed)

        self.train_gen, self.val_gen = self._make_generators()
        self.steps_train = self.train_gen.samples // self.cfg.batch_size
        self.steps_val = self.val_gen.samples // self.cfg.batch_size

        self.model = self._build_model()

    # ------------------------------------------------------------------ #
    # public API                                                         #
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        history = self._fit()
        self._save_model(history)
        plot_train_test_loss(history, self.cfg.out_dir)

    # ------------------------------------------------------------------ #
    # internals                                                          #
    # ------------------------------------------------------------------ #
    def _make_generators(self):
        datagen = ImageDataGenerator(rescale=self.cfg.rescale)

        train = datagen.flow_from_directory(
            self.cfg.train_dir,
            target_size=(self.cfg.img_size, self.cfg.img_size),
            batch_size=self.cfg.batch_size,
            class_mode="input",
            shuffle=True,
            seed=self.cfg.seed,
        )
        val = datagen.flow_from_directory(
            self.cfg.test_dir,
            target_size=(self.cfg.img_size, self.cfg.img_size),
            batch_size=self.cfg.batch_size,
            class_mode="input",
            shuffle=True,
            seed=self.cfg.seed,
        )
        return train, val

    @staticmethod
    def _compile(model: tf.keras.Model) -> tf.keras.Model:
        model.compile(
            optimizer="adam",
            loss="mean_squared_error",
            metrics=["mse"],
        )
        return model

    def _build_model(self):
        enc = encoder(self.cfg.img_size)
        dec = decoder(enc)
        return self._compile(dec)

    def _fit(self):
        return self.model.fit(
            self.train_gen,
            steps_per_epoch=self.steps_train,
            epochs=self.cfg.num_epoch,
            validation_data=self.val_gen,
            validation_steps=self.steps_val,
            shuffle=True,
        )

    # ------------------------------------------------------------------ #
    # save utilities                                                     #
    # ------------------------------------------------------------------ #
    def _save_model(self, history):
        size = self.cfg.img_size
        # encoder is already inside model.layers[0] but save separately too
        enc_path = self.cfg.out_dir / f"encoder_model_{size}"
        self.model.layers[0].save(enc_path, save_format="tf")

        full_path = self.cfg.out_dir / f"model_{size}.h5"
        self.model.save(full_path, include_optimizer=False)

        hist_path = self.cfg.out_dir / f"history_{size}.json"
        hist_path.write_text(json.dumps(history.history))

    # ------------------------------------------------------------------ #
    # CLI                                                                #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _cli():
        import argparse, json as _json, sys

        p = argparse.ArgumentParser(description="Train AE on PSD images")
        p.add_argument("--config", help="JSON file with TrainAEConfig fields")
        args = p.parse_args()

        if args.config:
            cfg = TrainAEConfig(**_json.load(open(args.config)))
        else:
            cfg = TrainAEConfig()  # defaults

        AutoencoderTrainer(cfg).run()
