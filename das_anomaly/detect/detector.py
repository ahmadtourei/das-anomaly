"""
Run a trained auto-encoder to score PSD PNGs and copy any anomalies.

Example
-------
>>> from das_anomaly.detect import DetectConfig, AnomalyDetector
>>> # serial processing with single processor:
>>> cfg = DetectConfig(psd_dir="~/psd_pngs", results_path="~/results")
>>> AnomalyDetector(cfg).run()
>>> # parallel processing with multiple processors using MPI:
>>> AnomalyDetector(cfg).run_parallel()
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from keras.models import load_model
from sklearn.neighbors import KernelDensity
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from das_anomaly import check_if_anomaly
from das_anomaly.settings import SETTINGS

# optional MPI import
try:
    from mpi4py import MPI  # (we want the canonical upper-case name)
except:

    class _DummyComm:
        """Stand-in that mimics the tiny subset we use."""

        def Get_rank(self):
            return 0

        def Get_size(self):
            return 1

        def bcast(self, obj, root=0):
            return obj

    MPI = type("FakeMPI", (), {"COMM_WORLD": _DummyComm()})()  # pragma: no cover
    print(
        "mpi4py not available - A fake MPI communicator is assigned. "
        "Install mpi4py and run your script under `mpirun` for parallel execution."
    )


# ------------------------------------------------------------------ #
# configuration                                                      #
# ------------------------------------------------------------------ #
@dataclass
class DetectConfig:
    """Paths & thresholds for the anomaly-detection pipeline."""

    # directories -----------------------------------------------------
    psd_path: Path | str = SETTINGS.PSD_PATH
    results_path: Path | str = SETTINGS.RESULTS_PATH
    trained_path: Path | str = SETTINGS.TRAINED_PATH
    train_images_path: Path | str = SETTINGS.TRAIN_IMAGES_PATH

    # model / data params --------------------------------------------
    size: int = SETTINGS.SIZE
    batch_size: int = SETTINGS.BATCH_SIZE
    density_threshold: float = SETTINGS.DENSITY_THRESHOLD

    def __post_init__(self):
        # expand paths
        self.psd_path = Path(self.psd_path).expanduser()
        self.results_path = Path(self.results_path).expanduser()
        self.trained_path = Path(self.trained_path).expanduser()
        self.train_images_path = Path(self.train_images_path).expanduser()

        self.results_path.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ #
# detector                                                           #
# ------------------------------------------------------------------ #
class AnomalyDetector:
    """Load AE, fit KDE in latent space, and classify each PSD PNG."""

    def __init__(self, cfg: DetectConfig):
        self.cfg = cfg
        self.model = self._load_model()
        self.encoder = self._extract_encoder()
        self.kde = self._fit_kde()

        self.dest_dir = self.cfg.results_path / "copied_detected_anomalies"
        self.dest_dir.mkdir(exist_ok=True)

    # -------------------------------------------------------------- #
    # public API                                                     #
    # -------------------------------------------------------------- #
    def run(self) -> None:
        """Score every PSD PNG and log / copy anomalies with single processor."""
        root_pngs = list(self.cfg.psd_path.glob("*.png"))
        for i, folder in enumerate(self.cfg.psd_path.iterdir()):
            if not folder.is_dir():
                continue

            spectra = (
                sorted(root_pngs + list(folder.glob("*.png")))
                if i == 0
                else sorted(folder.glob("*.png"))
            )
            out_file = (
                self.cfg.results_path
                / f"{folder.name}_output_model_{self.cfg.size}_anomaly.txt"
            )
            with out_file.open("w") as fh:
                for j, img_path in enumerate(spectra):
                    flag = check_if_anomaly(
                        encoder_model=self.encoder,
                        size=self.cfg.size,
                        img_path=img_path,
                        density_threshold=self.cfg.density_threshold,
                        kde=self.kde,
                    )
                    print(f"Line {j}, image {img_path}: {flag}", file=fh)

                    if flag.endswith("anomaly"):
                        shutil.copy(img_path, self.dest_dir)

    def run_parallel(self) -> None:
        """Score PSD PNGs in parallel (one folder per MPI rank)."""
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        world_size = comm.Get_size()

        # Materialise sub-folders once for every rank
        subdirs: list[Path] = [p for p in self.cfg.psd_path.iterdir() if p.is_dir()]
        if len(subdirs) == 0:
            num_folders = 1
        else:
            num_folders = len(subdirs)
        for i in range(rank, num_folders, world_size):
            # if no directories exist under psd_path, rank 0 looks for PNGs under psd_path
            if rank == 0 and len(subdirs) == 0:
                spectra = sorted(self.cfg.psd_path.glob("*.png"))
                out_file = (
                    self.cfg.results_path
                    / f"{self.cfg.psd_path.name}_output_model_{self.cfg.size}_anomaly.txt"
                )
            else:
                folder_path: Path = subdirs[i]
                spectra = sorted(folder_path.glob("*.png"))
                out_file = (
                    self.cfg.results_path
                    / f"{folder_path.name}_output_model_{self.cfg.size}_anomaly.txt"
                )

            with out_file.open("w") as fh:
                for j, img_path in enumerate(spectra):
                    flag = check_if_anomaly(
                        encoder_model=self.encoder,
                        size=self.cfg.size,
                        img_path=img_path,
                        density_threshold=self.cfg.density_threshold,
                        kde=self.kde,
                    )
                    print(f"Rank {rank} · line {j}, {img_path}: {flag}", file=fh)

                    if flag.endswith("anomaly"):
                        shutil.copy(img_path, self.dest_dir)

    # -------------------------------------------------------------- #
    # internal helpers                                               #
    # -------------------------------------------------------------- #
    def _load_model(self):
        model_path = self.cfg.trained_path / f"model_{self.cfg.size}.h5"
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        return load_model(model_path)

    def _train_generator(self):
        dg = ImageDataGenerator(rescale=1.0 / 255)
        return dg.flow_from_directory(
            self.cfg.train_images_path,
            target_size=(self.cfg.size, self.cfg.size),
            batch_size=self.cfg.batch_size,
            class_mode="input",
        )

    def _extract_encoder(self):
        """Build a stand-alone encoder and copy weights from the trained model."""
        size = self.cfg.size
        enc = Sequential()
        enc.add(
            Conv2D(
                64,
                (3, 3),
                activation="relu",
                padding="same",
                input_shape=(size, size, 3),
            )
        )
        enc.layers[-1].set_weights(self.model.layers[0].get_weights())

        enc.add(MaxPooling2D((2, 2), padding="same"))
        enc.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
        enc.layers[-1].set_weights(self.model.layers[2].get_weights())

        enc.add(MaxPooling2D((2, 2), padding="same"))
        enc.add(Conv2D(16, (3, 3), activation="relu", padding="same"))
        enc.layers[-1].set_weights(self.model.layers[4].get_weights())
        enc.add(MaxPooling2D((2, 2), padding="same"))
        return enc

    def _fit_kde(self):
        """Fit KDE on latent vectors of *train* images."""
        gen = self._train_generator()
        encoded = self.encoder.predict(gen, verbose=0)
        flat = encoded.reshape(encoded.shape[0], -1)
        return KernelDensity(kernel="gaussian", bandwidth=0.2).fit(flat)

    # -------------------------------------------------------------- #
    # CLI convenience                                                #
    # -------------------------------------------------------------- #
    @staticmethod
    def _cli():
        """Command-line entry-point."""
        p = argparse.ArgumentParser(description="Detect anomalies in PSD images")
        p.add_argument("--config", help="JSON/YAML with DetectConfig fields")
        args = p.parse_args()

        if args.config:
            cfg = DetectConfig(**json.load(open(args.config)))
        else:
            cfg = DetectConfig()

        AnomalyDetector(cfg).run()
