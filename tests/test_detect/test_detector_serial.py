"""
Comprehensive tests for das_anomaly.detect.detector in serial mode
------------------------------------------------------------------
All heavy TF/Keras operations are mocked so the suite runs fast.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from das_anomaly.detect import AnomalyDetector, DetectConfig


def _make_dummy_png(path: Path):
    rng = np.random.default_rng()
    arr = (rng.random((8, 8, 3)) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


class TestInit:
    """Constructor-level validation."""

    def test_missing_model_raises(self, tmp_path):
        cfg = DetectConfig(
            psd_path=tmp_path / "psd",
            results_path=tmp_path / "out",
            train_images_path=tmp_path,
            trained_path=tmp_path,
        )
        with pytest.raises(FileNotFoundError):
            AnomalyDetector(cfg)


class TestEncoderExtraction:
    """Weights copied from trained AE into stand-alone encoder."""

    def test_weights_set(self, tmp_path, patched_tf):
        cfg = DetectConfig(
            psd_path=tmp_path / "psd",
            results_path=tmp_path / "out",
            train_images_path=tmp_path,
            trained_path=tmp_path,
        )
        (cfg.trained_path / f"model_{cfg.size}.h5").touch()

        _ = AnomalyDetector(cfg)
        conv_layers = [
            la
            for la in patched_tf["encoder_seq"].layers
            if getattr(la, "name", "").startswith("conv")
        ]
        assert any(getattr(la, "_set_called", False) for la in conv_layers)


class TestKDEFitting:
    """KDE.fit receives flattened latent vectors."""

    def test_fit_vector_shape(self, tmp_path, patched_tf):
        cfg = DetectConfig(
            psd_path=tmp_path / "psd",
            results_path=tmp_path / "out",
            train_images_path=tmp_path,
            trained_path=tmp_path,
        )
        (cfg.trained_path / f"model_{cfg.size}.h5").touch()
        _ = AnomalyDetector(cfg)
        assert patched_tf["kde_fit_called"]["shape"] == (4, 16)  # 4 imgs * 16 dims


class TestRunEndToEnd:
    """Full run(): PNG discovery, logging, copying."""

    def test_run_copies_and_logs(self, tmp_path, patched_tf):
        # PSD folder with 3 PNGs
        psd_rank = tmp_path / "psd" / "rank_0"
        psd_rank.mkdir(parents=True)
        for i in range(3):
            _make_dummy_png(psd_rank / f"img_{i}.png")

        cfg = DetectConfig(
            psd_path=tmp_path / "psd",
            results_path=tmp_path / "out",
            train_images_path=tmp_path,
            trained_path=tmp_path,
            density_threshold=1_000,
            size=8,
        )

        (cfg.trained_path / f"model_{cfg.size}.h5").touch()

        det = AnomalyDetector(cfg)
        det.run()

        # log file exists
        log = cfg.results_path / f"{psd_rank.name}_output_model_{cfg.size}_anomaly.txt"
        assert log.exists() and log.stat().st_size > 0

        # anomalies copied
        copied = list((cfg.results_path / "copied_detected_anomalies").glob("*.png"))
        assert len(copied) == 3


class TestCLI:
    """CLI entry-point smoke test."""

    def test_cli_invocation(self, tmp_path, monkeypatch, patched_tf, capsys):
        cfg_file = tmp_path / "cfg.json"
        json.dump(
            {
                "psd_path": str(tmp_path / "psd"),
                "results_path": str(tmp_path / "out"),
                "train_images_path": str(tmp_path),
                "trained_path": str(tmp_path),
                "size": 8,
                "density_threshold": 1,
            },
            cfg_file.open("w"),
        )

        # touch dummy model
        (tmp_path / "out").mkdir(exist_ok=True, parents=True)
        (tmp_path / "model_8.h5").touch()

        monkeypatch.setattr(sys, "argv", ["detect-anomaly", "--config", str(cfg_file)])
        monkeypatch.setattr(AnomalyDetector, "run", lambda self: print("CLI OK"))  # noqa: T201
        AnomalyDetector._cli()
        assert "CLI OK" in capsys.readouterr().out

    def test_cli_defaults(self, monkeypatch, capsys):
        import sys

        import das_anomaly.detect.detector as det_mod

        # DetectConfig spy (records call, returns placeholder)
        call_log = {}
        monkeypatch.setattr(
            det_mod,
            "DetectConfig",
            lambda: call_log.setdefault("default_cfg", object()),
        )

        # neuter heavy constructor BEFORE _cli runs
        monkeypatch.setattr(det_mod.AnomalyDetector, "__init__", lambda self, cfg: None)
        monkeypatch.setattr(
            det_mod.AnomalyDetector,
            "run",
            lambda self: print("DEFAULT OK"),  # noqa: T201
        )

        # invoke CLI without --config
        monkeypatch.setattr(sys, "argv", ["detect-anomaly"])
        det_mod.AnomalyDetector._cli()

        # DetectConfig() actually executed
        assert "default_cfg" in call_log
        assert "DEFAULT OK" in capsys.readouterr().out


class TestRunSkipsNonDirs:
    """`run()` should ignore items in psd_path that are *not* directories."""

    def test_loose_file_is_ignored(self, tmp_path, patched_tf):
        # PSD root contains one valid sub-dir and one stray file
        psd_root = tmp_path / "psd"
        psd_root.mkdir()
        good_sub = psd_root / "rank_0"
        good_sub.mkdir()
        _make_dummy_png(good_sub / "img.png")

        # stray file that must be skipped
        stray = psd_root / "readme.txt"
        stray.write_text("not a directory")

        cfg = DetectConfig(
            psd_path=psd_root,
            results_path=tmp_path / "out",
            train_images_path=tmp_path,
            trained_path=tmp_path,
            size=8,
            density_threshold=1_000,
        )
        (cfg.trained_path / f"model_{cfg.size}.h5").touch()

        AnomalyDetector(cfg).run()

        # only a log for the *directory* should exist
        logs = list(cfg.results_path.glob("*_output_model_*_anomaly.txt"))
        assert len(logs) == 1
        assert "rank_0" in logs[0].name
        # no log created for the stray file
        assert "readme" not in logs[0].name


class TestRootLevelSpectra:
    """
    When *.png files live directly in ``psd_path`` (plus at least one
    sub-directory to trigger the loop) they must still be processed and—
    because our mocks always flag “anomaly”— copied to the results folder.
    """

    def test_root_pngs_are_processed(self, tmp_path, patched_tf):
        psd_root = tmp_path / "psd"
        psd_root.mkdir()

        # ❶ root-level spectra
        for i in range(4):
            _make_dummy_png(psd_root / f"root_{i}.png")

        # ❷ at least one sub-directory (can stay empty)
        (psd_root / "rank_0").mkdir()

        # config & dummy model
        cfg = DetectConfig(
            psd_path=psd_root,
            results_path=tmp_path / "out",
            train_images_path=tmp_path,
            trained_path=tmp_path,
            density_threshold=1_000,  # everything is “anomaly” with our mocks
            size=8,
        )
        (cfg.trained_path / f"model_{cfg.size}.h5").touch()

        # run detector
        AnomalyDetector(cfg).run()

        # assertions
        copied = list((cfg.results_path / "copied_detected_anomalies").glob("*.png"))
        assert len(copied) == 4, "all root-level spectra should be copied"
