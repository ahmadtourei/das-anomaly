"""
Comprehensive tests for das_anomaly.detect.detector in parallel mode
--------------------------------------------------------------------
All heavy TF/Keras operations are mocked so the suite runs fast.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from das_anomaly.detect import AnomalyDetector, DetectConfig


# ------------------------------------------------------------------ #
# helper to fabricate PNGs                                           #
# ------------------------------------------------------------------ #
def _make_png(path: Path):
    rng = np.random.default_rng()
    arr = (rng.random((8, 8, 3)) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


# ------------------------------------------------------------------ #
# fake MPI communicator fixture                                      #
# ------------------------------------------------------------------ #
class _FakeMPIComm:
    def __init__(self, rank: int, size: int):
        self._rank = rank
        self._size = size

    # mpi4py interface methods
    def Get_rank(self):
        return self._rank

    def Get_size(self):
        return self._size

    def bcast(self, obj, root=0):
        return obj  # compatibility


def _inject_fake_mpi(monkeypatch, rank: int, size: int):
    """
    Replace das_anomaly.detect.detector.MPI with a minimal stub
    whose COMM_WORLD behaves like the requested rank/size.
    """
    import das_anomaly.detect.detector as mod

    fake = SimpleNamespace(COMM_WORLD=_FakeMPIComm(rank, size))
    monkeypatch.setattr(mod, "MPI", fake)


class TestRunParallelMPI:
    """Verify folder scheduling & side-effects under a fake MPI world."""

    @pytest.fixture(scope="function")
    def psd_tree(self, tmp_path):
        """
        Create 4 PSD sub-folders, each with one PNG.
        Returns (psd_root_path, list_of_subfolder_names)
        """
        names = [f"folder_{i}" for i in range(4)]
        psd_root = tmp_path / "psd"
        for name in names:
            sub = psd_root / name
            sub.mkdir(parents=True)
            _make_png(sub / f"{name}.png")
        return psd_root, names

    @pytest.mark.parametrize("rank", [0, 1])
    def test_each_rank_handles_its_slice(
        self, tmp_path, psd_tree, patched_tf, monkeypatch, rank
    ):
        psd_root, names = psd_tree
        size_world = 2

        # config & dummy model file
        cfg = DetectConfig(
            psd_path=psd_root,
            results_path=tmp_path / "out",
            train_images_path=tmp_path,
            trained_path=tmp_path,
            density_threshold=1_000,
            size=8,
        )
        (cfg.trained_path / f"model_{cfg.size}.h5").touch()

        # swap real MPI with our stub for this rank
        _inject_fake_mpi(monkeypatch, rank, size_world)

        det = AnomalyDetector(cfg)
        det.run_parallel()

        # list-of-subdirs exactly as run_parallel() sees them
        subdirs = [p.name for p in psd_root.iterdir() if p.is_dir()]

        expected = {subdirs[i] for i in range(rank, len(subdirs), size_world)}

        # check that corresponding log files were written
        produced_logs = {
            p.stem.split("_output_model")[0]
            for p in cfg.results_path.glob("*_output_model_*_anomaly.txt")
        }
        assert expected.issubset(produced_logs)

        # each PNG flagged as anomaly (score 0.9 < threshold 1000) → copied
        copied = list((cfg.results_path / "copied_detected_anomalies").glob("*.png"))
        # since every folder handled by this rank has exactly one PNG
        assert len(copied) == len(expected)


class TestRootPngHandling:
    """
    When one or more sub-directories exist, run_parallel() should ignore PNGs
    that sit directly inside *psd_path*.  They are processed only in the special
    “no-subdir” case.
    """

    def test_root_pngs_ignored_if_subdirs_exist(
        self, tmp_path, patched_tf, monkeypatch
    ):
        # set-up
        psd_root = tmp_path / "psd"
        psd_root.mkdir()

        # three PNGs directly under psd_root (must *not* be copied)
        for i in range(3):
            _make_png(psd_root / f"root_{i}.png")

        # one sub-dir with a single PNG (this one should be copied)
        sub = psd_root / "rank_0"
        sub.mkdir()
        _make_png(sub / "one.png")

        cfg = DetectConfig(
            psd_path=psd_root,
            results_path=tmp_path / "out",
            train_images_path=tmp_path,
            trained_path=tmp_path,
            density_threshold=1_000,
            size=8,
        )
        (cfg.trained_path / f"model_{cfg.size}.h5").touch()

        # fake a single-rank MPI world
        _inject_fake_mpi(monkeypatch, rank=0, size=1)

        # exercise
        AnomalyDetector(cfg).run_parallel()

        # verify
        copied = list((cfg.results_path / "copied_detected_anomalies").glob("*.png"))
        assert len(copied) == 1, "root-level PNGs must be skipped when sub-dirs exist"
        assert copied[0].name == "one.png"


@pytest.mark.parametrize(
    "rank, world_size, expected_n_copies, expect_log",
    [
        (0, 2, 4, True),  # rank-0 handles *all* root-level PNGs
        (1, 2, 0, False),  # every other rank skips work
    ],
)
def test_no_subdirs_root_pngs_only(
    tmp_path, patched_tf, monkeypatch, rank, world_size, expected_n_copies, expect_log
):
    """
    If *psd_path* contains **no sub-directories**:

    • rank 0 must process every PNG found directly in *psd_path*
    • all other ranks do nothing
    """
    # fabricate four root-level PNGs
    psd_root = tmp_path / "psd"
    psd_root.mkdir()
    for i in range(4):
        _make_png(psd_root / f"root_{i}.png")

    # config & dummy model inside tmp_path
    cfg = DetectConfig(
        psd_path=psd_root,
        results_path=tmp_path / "out",
        train_images_path=tmp_path,
        trained_path=tmp_path,
        density_threshold=1_000,
        size=8,
    )
    (cfg.trained_path / f"model_{cfg.size}.h5").touch()

    # fake the MPI world for the chosen rank
    _inject_fake_mpi(monkeypatch, rank=rank, size=world_size)

    # run
    AnomalyDetector(cfg).run_parallel()

    # copied anomalies
    copied = list((cfg.results_path / "copied_detected_anomalies").glob("*.png"))
    assert len(copied) == expected_n_copies

    # root-level log file (rank 0 only)
    log_name = f"{cfg.psd_path.name}_output_model_{cfg.size}_anomaly.txt"
    log_path = cfg.results_path / log_name
    assert log_path.exists() is expect_log
