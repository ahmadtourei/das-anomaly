"""
Tests for utility functions in das_anomaly.utils
"""
from types import SimpleNamespace
from pathlib import Path
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import pytest

from das_anomaly.utils import (
    calculate_percentile,
    check_if_anomaly,
    density,
    plot_spec,
    plot_train_test_loss,
    search_keyword_in_files,
)


class TestCalculatePercentile:
    """Tests for calculate_percentile"""

    def test_basic(self):
        """50 % → median, 0 % → min, empty list → None."""
        data = [1, 2, 3, 4, 5]
        assert calculate_percentile(data, 50) == 3
        assert calculate_percentile(data, 0) == 1
        assert calculate_percentile([], 50) is None

    def test_out_of_bounds(self):
        """Out-of-range percentile → ValueError."""
        with pytest.raises(ValueError):
            calculate_percentile([1, 2, 3], -5)
        with pytest.raises(ValueError):
            calculate_percentile([1, 2, 3], 105)

    def test_interpolation(self):
        """25 % of [1,2,3,4] → 1.75 via linear interpolation."""
        assert np.isclose(calculate_percentile([1, 2, 3, 4], 25), 1.75)


class TestSearchKeyword:
    """Tests for search_keyword_in_files"""

    def test_keyword_found(self, tmp_path: Path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("line apple\nbanana\n")
        f2.write_text("apple pie\nanother apple\n")
        count, lines = search_keyword_in_files(tmp_path, "apple")
        assert count == 3
        assert len(lines) == 3
        assert all("apple" in l for l in lines)


class DummyKDE:
    """Return a constant KDE score for every sample."""
    def __init__(self, score): self.score = score
    def score_samples(self, X): return np.full(len(X), self.score)


class DummyEncoder:
    """Fake encoder: output shape (4,4,1) and constant prediction."""
    output_shape = (None, 4, 4, 1)
    def predict(self, X, verbose=0): return [np.zeros((4, 4, 1))]


@pytest.fixture
def dummy_png(tmp_path: Path):
    """RGB 3×3 dummy image saved as PNG."""
    img = Image.fromarray(np.zeros((3, 3, 3), dtype=np.uint8))
    file_ = tmp_path / "img.png"
    img.save(file_)
    return file_


class TestCheckIfAnomaly:
    """Tests for check_if_anomaly"""

    def test_rgba_branch(self, tmp_path: Path):
        """RGBA source image is converted to RGB without error."""
        rgba = np.zeros((3, 3, 4), dtype=np.uint8)
        rgba[..., 3] = 255
        img_path = tmp_path / "rgba.png"
        Image.fromarray(rgba, mode="RGBA").save(img_path)
        out = check_if_anomaly(DummyEncoder(), 8, img_path, 0.5, DummyKDE(0.8))
        assert out.endswith("normal")

    def test_flags_normal(self, dummy_png):
        out = check_if_anomaly(DummyEncoder(), 8, dummy_png, 0.5, DummyKDE(0.9))
        assert out.endswith("normal")

    def test_flags_anomaly(self, dummy_png):
        out = check_if_anomaly(DummyEncoder(), 8, dummy_png, 0.5, DummyKDE(0.1))
        assert out.endswith("anomaly")


class TestDensity:
    """Tests for density"""

    def test_vectorisation(self):
        enc, kde = DummyEncoder(), DummyKDE(0.7)
        batch = np.zeros((5, 8, 8, 3))
        scores = density(enc, batch, kde)
        assert np.allclose(scores, 0.7)
        assert scores.shape == (5,)


class DummyCoords:
    def get_coord(self, _): return np.arange(10)


def _dummy_patch():
    return SimpleNamespace(
        transpose=lambda *a, **k: SimpleNamespace(data=np.zeros((20, 10))),
        coords=DummyCoords(),
    )


class TestPlotSpec:
    """Input-validation branches in plot_spec"""

    def test_nyquist_warning(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(plt, "savefig", lambda *a, **k: None)
        plot_spec(_dummy_patch(), 0, 600, 1000, "t", 0, tmp_path, 72)
        assert "max_frq" in capsys.readouterr().out

    def test_min_ge_max(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(plt, "savefig", lambda *a, **k: None)
        plot_spec(_dummy_patch(), 100, 50, 1000, "t", 0, tmp_path, 72)
        assert "minFrq 100 >= maxFrq 50" in capsys.readouterr().out


class TestPlotTrainTestLoss:
    """Intercept savefig and ensure a PNG would be written."""

    def test_saves_png(self, tmp_path: Path, monkeypatch):
        hist = SimpleNamespace(history=dict(loss=[1, 0.5], val_loss=[1.1, 0.6]))
        paths: list[Path] = []
        monkeypatch.setattr(plt, "savefig", lambda p, *a, **k: paths.append(Path(p)))
        plot_train_test_loss(hist, tmp_path)
        assert paths and paths[0].name.startswith("Training_and_validation")
