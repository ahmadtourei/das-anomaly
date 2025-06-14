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

from das_anomaly.detect import DetectConfig, AnomalyDetector


def _make_dummy_png(path: Path):
    arr = (np.random.rand(8, 8, 3) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


class TestInit:
    """Constructor-level validation."""

    def test_missing_model_raises(self, tmp_path):
        cfg = DetectConfig(psd_path=tmp_path,
                           results_path=tmp_path,
                           train_images_path=tmp_path)
        with pytest.raises(FileNotFoundError):
            AnomalyDetector(cfg)


class TestEncoderExtraction:
    """Weights copied from trained AE into stand-alone encoder."""

    def test_weights_set(self, tmp_path, patched_tf):
        cfg = DetectConfig(psd_path=tmp_path,
                           results_path=tmp_path,
                           train_images_path=tmp_path)
        (cfg.results_path / f"model_{cfg.size}.h5").touch()

        _ = AnomalyDetector(cfg)
        conv_layers = [l for l in patched_tf["encoder_seq"].layers
                    if getattr(l, "name", "").startswith("conv")]
        assert any(getattr(l, "_set_called", False) for l in conv_layers)


class TestKDEFitting:
    """KDE.fit receives flattened latent vectors."""

    def test_fit_vector_shape(self, tmp_path, patched_tf):
        cfg = DetectConfig(psd_path=tmp_path,
                           results_path=tmp_path,
                           train_images_path=tmp_path)
        (cfg.results_path / f"model_{cfg.size}.h5").touch()
        _ = AnomalyDetector(cfg)
        assert patched_tf["kde_fit_called"]["shape"] == (4, 16)  # 4 imgs × 16 dims


class TestRunEndToEnd:
    """Full run(): PNG discovery, logging, copying."""
    def test_run_copies_and_logs(self, tmp_path, patched_tf):
        # PSD folder with 3 PNGs
        psd_rank = tmp_path / "psd" / "rank_0"
        psd_rank.mkdir(parents=True)
        for i in range(3):
            _make_dummy_png(psd_rank / f"img_{i}.png")

        cfg = DetectConfig(psd_path=tmp_path / "psd",
                           results_path=tmp_path / "out",
                           train_images_path=tmp_path,
                           density_threshold=1_000, 
                           size=8)
        (cfg.results_path / f"model_{cfg.size}.h5").touch()

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
        json.dump({
            "psd_path": str(tmp_path / "psd"),
            "results_path": str(tmp_path / "out"),
            "train_images_path": str(tmp_path),
            "size": 8,
            "density_threshold": 1,
        }, cfg_file.open("w"))

        # touch dummy model
        (tmp_path / "out").mkdir(exist_ok=True, parents=True)
        (tmp_path / "out" / "model_8.h5").touch()

        monkeypatch.setattr(sys, "argv", ["detect-anomaly", "--config", str(cfg_file)])
        monkeypatch.setattr(AnomalyDetector, "run", lambda self: print("CLI OK"))
        AnomalyDetector._cli()
        assert "CLI OK" in capsys.readouterr().out

