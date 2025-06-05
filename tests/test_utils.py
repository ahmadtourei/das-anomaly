"""
Tests for utility functions in das_anomaly.utils
"""
from types import SimpleNamespace
from pathlib import Path
from PIL import Image
from unittest.mock import MagicMock

import numpy as np
import pytest

from das_anomaly.utils import (  # noqa: E402
    calculate_percentile,
    check_if_anomaly,
    density,
    plot_train_test_loss,
    search_keyword_in_files,
)


def test_calculate_percentile_basic():
    """50 % → median, 0 % → min, empty list → None."""
    data = [1, 2, 3, 4, 5]
    assert calculate_percentile(data, 50) == 3
    assert calculate_percentile(data, 0) == 1
    assert calculate_percentile([], 50) is None



def test_search_keyword(tmp_path: Path):
    """Find keyword across multiple .txt files and count occurrences."""
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

    def __init__(self, score):
        self.score = score

    def score_samples(self, X):
        return np.full(len(X), self.score)


class DummyEncoder:
    """Fake encoder: output shape (4,4,1) and constant prediction."""

    output_shape = (None, 4, 4, 1)

    def predict(self, X, verbose=0):
        return [np.zeros((4, 4, 1), dtype=float)]


@pytest.fixture
def dummy_png(tmp_path: Path):
    """RGB 3*3 dummy image saved as PNG."""
    img = Image.fromarray(np.zeros((3, 3, 3), dtype=np.uint8))
    file_ = tmp_path / "img.png"
    img.save(file_)
    return file_


def test_check_if_anomaly_flags_normal(dummy_png):
    """High KDE score → function returns 'normal'."""
    enc = DummyEncoder()
    kde = DummyKDE(score=0.9)
    out = check_if_anomaly(enc, 8, dummy_png, density_threshold=0.5, kde=kde)
    assert out.endswith("normal")


def test_check_if_anomaly_flags_anomaly(dummy_png):
    """Low KDE score → function returns 'anomaly'."""
    enc = DummyEncoder()
    kde = DummyKDE(score=0.1)
    out = check_if_anomaly(enc, 8, dummy_png, density_threshold=0.5, kde=kde)
    assert out.endswith("anomaly")


def test_density_vectorization():
    """density() returns identical score for each image in batch."""
    enc = DummyEncoder()
    kde = DummyKDE(score=0.7)
    batch = np.zeros((5, 8, 8, 3))  # 5 fake images
    scores = density(enc, batch, kde)
    assert np.allclose(scores, 0.7)
    assert scores.shape == (5,)


def test_plot_train_test_loss(tmp_path: Path, monkeypatch):
    """plot_train_test_loss saves a PNG (savefig is intercepted)."""
    history = SimpleNamespace(history=dict(loss=[1, 0.5], val_loss=[1.1, 0.6]))

    import matplotlib.pyplot as plt

    saved_paths = []

    def fake_savefig(path, *_, **__):
        saved_paths.append(Path(path))

    monkeypatch.setattr(plt, "savefig", fake_savefig)
    plot_train_test_loss(history, tmp_path)

    assert saved_paths       # at least one file path captured
    assert saved_paths[0].name.startswith("Training_and_validation")
