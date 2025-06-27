"""
Tests for utility functions in das_anomaly.utils
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Sequential
from PIL import Image

from das_anomaly.utils import (
    calculate_percentile,
    check_if_anomaly,
    decoder,
    density,
    encoder,
    ft,
    get_psd_max_clip,
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

    def test_numpy_input(self):
        """Test NumPy arrays (incl. empty)."""
        arr = np.array([10, 20, 30, 40])
        assert calculate_percentile(arr, 50) == 25  # median of sorted [10,20,30,40]

        empty = np.array([])
        assert calculate_percentile(empty, 90) is None


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
        assert all("apple" in li for li in lines)


class DummyKDE:
    """Return a constant KDE score for every sample."""

    def __init__(self, score):
        self.score = score

    def score_samples(self, x):
        return np.full(len(x), self.score)


class DummyEncoder:
    """Fake encoder: output shape (4,4,1) and constant prediction."""

    output_shape = (None, 4, 4, 1)

    def predict(self, x, verbose=0):
        return [np.zeros((4, 4, 1))]


@pytest.fixture
def dummy_png(tmp_path: Path):
    """RGB 3*3 dummy image saved as PNG."""
    img = Image.fromarray(np.zeros((3, 3, 3), dtype=np.uint8))
    file_ = tmp_path / "img.png"
    img.save(file_)
    return file_


def _dummy_patch():
    return SimpleNamespace(
        transpose=lambda *a, **k: SimpleNamespace(data=np.zeros((20, 10))),
        coords=DummyCoords(),
    )


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


class TestTrainer:
    def test_encoder_architecture(self):
        """encoder() returns 3*(Conv→Pool) stack and ends with 16 filters."""
        size = 32
        enc: Sequential = encoder(size)

        # layer order check: Conv, Pool, Conv, Pool, Conv, Pool
        types = [type(layer) for layer in enc.layers]
        assert types == [
            Conv2D,
            MaxPooling2D,
            Conv2D,
            MaxPooling2D,
            Conv2D,
            MaxPooling2D,
        ]

        # last Conv2D has 16 output channels
        last_conv: Conv2D = enc.layers[-2]
        assert last_conv.filters == 16

    def test_decoder_appends_layers(self):
        enc = encoder(32)
        n_before = len(enc.layers)

        decoder(enc)

        # 7 layers added
        assert len(enc.layers) == n_before + 7

        # pattern: Conv ▸ UpSample repeated + final Conv
        added = enc.layers[n_before:]
        expected = [
            Conv2D,
            UpSampling2D,
            Conv2D,
            UpSampling2D,
            Conv2D,
            UpSampling2D,
            Conv2D,
        ]
        assert [type(l) for l in added] == expected

        last = enc.layers[-1]
        assert isinstance(last, Conv2D)
        assert last.filters == 3
        assert last.activation.__name__ == "sigmoid"


class DummyCoords:
    def get_coord(self, _):
        return np.arange(10)


class TestPlotSpec:
    """Input-validation branches in plot_spec"""

    def test_nyquist_warning(self, tmp_path, monkeypatch):
        """max_freq > nyquist should raise."""
        monkeypatch.setattr(plt, "savefig", lambda *a, **k: None)

        min_freq = 0
        max_freq = 600
        sampling_rate = 1000

        with pytest.raises(
            ValueError,
            match=rf"`max_freq` {max_freq} must be less than `nyquist frequency` 500",
        ):
            plot_spec(
                _dummy_patch(), min_freq, max_freq, sampling_rate, "t", 0, tmp_path, 72
            )

    def test_min_max_freq(self, tmp_path, monkeypatch):
        """min_freq ≥ max_freq should raise."""
        monkeypatch.setattr(plt, "savefig", lambda *a, **k: None)

        min_freq = 100
        max_freq = 50
        with pytest.raises(
            ValueError,
            match=f"`min_freq` {min_freq} must be less than " f"`max_freq` {max_freq}",
        ):
            plot_spec(_dummy_patch(), min_freq, max_freq, 1000, "t", 0, tmp_path, 72)


class TestPlotTrainTestLoss:
    """Intercept savefig and ensure a PNG would be written."""

    def test_saves_png(self, tmp_path: Path, monkeypatch):
        hist = SimpleNamespace(history=dict(loss=[1, 0.5], val_loss=[1.1, 0.6]))
        paths: list[Path] = []
        monkeypatch.setattr(plt, "savefig", lambda p, *a, **k: paths.append(Path(p)))
        plot_train_test_loss(hist, tmp_path)
        assert paths and paths[0].name.startswith("Training_and_validation")


class TestGetPsdMaxClip:
    """Focused checks for Nyquist math and percentile selection."""

    @staticmethod
    def _patch_like(data: np.ndarray):
        """Return object exposing .transpose(...).data → *data*."""
        return SimpleNamespace(transpose=lambda *a, **k: SimpleNamespace(data=data))

    @staticmethod
    def _monkey_fft(monkeypatch, fake_spectrum: np.ndarray):
        """Monkey-patch scipy.fftpack.fft inside utils to return *fake_spectrum*."""
        monkeypatch.setattr(ft, "fft", lambda *_a, **_k: fake_spectrum)

    def test_median_over_selected_bins(self, monkeypatch):
        """50-th percentile of rows 4-7 equals 5.5 for the synthetic spectrum."""
        data = np.ones((40, 2))  # dummy Patch data
        fake_fft = np.tile(np.arange(40)[:, None], (1, 2))  # rows 0..39
        self._monkey_fft(monkeypatch, fake_fft)

        patch = self._patch_like(data)

        # Nyquist = 500 ⇒ bin width 25 Hz; rows 4–7 correspond to 100–200 Hz
        clip = get_psd_max_clip(
            patch, min_freq=100, max_freq=200, sampling_rate=1_000, percentile=50
        )
        assert np.isclose(clip, 5.5)  # median of [4,5,6,7]

    def test_custom_percentile(self, monkeypatch):
        """80-th percentile with different slice/bandwidth is computed correctly."""
        data = np.ones((20, 3))
        fake_fft = np.arange(20)[:, None] * np.array([1, 2, 3])
        self._monkey_fft(monkeypatch, fake_fft)

        patch = self._patch_like(data)

        clip = get_psd_max_clip(
            patch, min_freq=0, max_freq=125, sampling_rate=500, percentile=80
        )
        expected = np.percentile(np.abs(fake_fft[:5]), 80)
        assert np.isclose(clip, expected)
