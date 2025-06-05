"""
Tests for utility functions in das_anomaly.utils
"""
from types import SimpleNamespace
from pathlib import Path
from PIL import Image

import matplotlib.pyplot as plt
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

def test_calculate_percentile_out_of_bounds():
    """Out-of-range percentile → ValueError."""
    with pytest.raises(ValueError):
        calculate_percentile([1, 2, 3], -5)     # < 0

    with pytest.raises(ValueError):
        calculate_percentile([1, 2, 3], 105)    # > 100

def test_calculate_percentile_interpolates():
    """25 % of [1,2,3,4]  →  1.75 via linear interpolation."""
    data = [1, 2, 3, 4]
    # pos = 0.75 so the result should be 25 % of the way between 1 and 2.
    assert np.isclose(calculate_percentile(data, 25), 1.75)


def test_check_if_anomaly_rgba_branch(tmp_path):
    """RGBA source image is converted to RGB without error."""
    # Create a 4-channel (RGBA) 3×3 PNG
    rgba_arr = np.zeros((3, 3, 4), dtype=np.uint8)
    rgba_arr[..., 3] = 255          # opaque alpha
    img_path = tmp_path / "rgba.png"
    Image.fromarray(rgba_arr, mode="RGBA").save(img_path)

    enc = DummyEncoder()            # from earlier fixtures
    kde = DummyKDE(score=0.8)       # any score above threshold
    out = check_if_anomaly(
        enc, size=8, img_path=img_path,
        density_threshold=0.5, kde=kde
    )
    assert out.endswith("normal")   # confirms function ran through


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


def test_plot_spec_nyquist_warning(tmp_path, monkeypatch, capsys):
    """When max_freq > Nyquist, function prints an error message."""
    import matplotlib.pyplot as plt
    from das_anomaly.utils import plot_spec

    # --- stub out pyplot to avoid writing a real file ----------------
    monkeypatch.setattr(plt, "savefig", lambda *a, **k: None)

    # --- minimal fake Patch object ----------------------------------
    class DummyCoords:
        def get_coord(self, name):
            # returns an array so .min() / .max() work
            return np.arange(10)

    patch_strain = SimpleNamespace(
        transpose=lambda *a, **k: SimpleNamespace(data=np.zeros((20, 10))),
        coords=DummyCoords(),
    )

    # sampling_rate=1 kHz → Nyquist=500; set max_freq higher to trigger branch
    plot_spec(
        patch_strain,
        min_freq=0,
        max_freq=600,          # > 500
        sampling_rate=1000,
        title="dummy",
        output_rank=0,
        fig_path=tmp_path,
        dpi=72,
    )

    out = capsys.readouterr().out
    assert "Error in plot_spec inputs: max_frq" in out


def test_plot_train_test_loss(tmp_path: Path, monkeypatch):
    """plot_train_test_loss saves a PNG (savefig is intercepted)."""
    history = SimpleNamespace(history=dict(loss=[1, 0.5], val_loss=[1.1, 0.6]))

    saved_paths = []

    def fake_savefig(path, *_, **__):
        saved_paths.append(Path(path))

    monkeypatch.setattr(plt, "savefig", fake_savefig)
    plot_train_test_loss(history, tmp_path)

    assert saved_paths       # at least one file path captured
    assert saved_paths[0].name.startswith("Training_and_validation")


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