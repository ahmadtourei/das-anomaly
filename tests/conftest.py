from unittest.mock import MagicMock

import numpy as np
import pytest
from dascore.compat import random_state
from dascore.core import Patch
from dascore.io.core import read
from dascore.utils.downloader import fetch
from PIL import Image

from das_anomaly.psd import PSDConfig


@pytest.fixture()
def dummy_patch() -> Patch:
    """
    Return a small simple array for memory testing.

    Note: This cant be a fixture as pytest seems to hold a reference,
    even for function scoped outputs.
    """
    attrs = {"data_type": "velocity", "time_step": 0.1}
    pa = Patch(
        data=random_state.random((100, 100)),
        coords={"time": np.arange(100) * 0.1, "distance": np.arange(100) * 0.2},
        attrs=attrs,
        dims=("time", "distance"),
    )
    return pa


@pytest.fixture()
def terra15_das_example_path():
    """Return the path to the example terra15 file in velocity."""
    out = fetch("terra15_das_1_trimmed.hdf5")
    assert out.exists()
    return out


@pytest.fixture()
def terra15_das_patch(terra15_das_example_path) -> Patch:
    """Read the terra15 data, return contained DataArray."""
    out = read(terra15_das_example_path, "terra15")[0]
    attr_time = out.attrs["time_max"]
    coortime_step = out.coords.coord_map["time"].max()
    assert attr_time == coortime_step
    return out


@pytest.fixture
def patched_plot_psd(monkeypatch):
    """Spy on das_anomaly.plot_spec so we can assert it was called."""
    spy = MagicMock(name="plot_spec_spy")
    monkeypatch.setattr("das_anomaly.psd.generator.plot_spec", spy)
    return spy


@pytest.fixture
def cfg(tmp_path, terra15_das_example_path, terra15_das_patch):
    """PSDConfig pointing to temp directories (auto-cleaned by pytest)."""
    return PSDConfig(
        data_path=terra15_das_example_path,
        psd_path=tmp_path / "psd",
        start_channel=0,
        end_channel=10,
        t1=terra15_das_patch.coords.coord_map["time"].min(),
        t2=terra15_das_patch.coords.coord_map["time"].max(),
        time_window=0.02,
        time_overlap=0.01,
    )


# Dummy KDE / Encoder used by utils & detector tests
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
def dummy_png(tmp_path):
    """RGB 3*3 PNG on disk."""
    img = Image.fromarray(np.zeros((3, 3, 3), dtype=np.uint8))
    file_ = tmp_path / "img.png"
    img.save(file_)
    return file_


# Heavy-dependency patching for detector tests
def _fake_conv_layer(name):
    class L:
        def __init__(self):
            self.name = name

        def get_weights(self):
            return [np.array([1])]

        def set_weights(self, w):
            self._set_called = True

    return L()


def _fake_sequential():
    seq = MagicMock(name="Sequential")
    seq.layers = [
        _fake_conv_layer("conv0"),
        MagicMock(),
        _fake_conv_layer("conv2"),
        MagicMock(),
        _fake_conv_layer("conv4"),
    ]
    seq.output_shape = (None, 4, 4, 1)

    # adaptively build an ndarray so both _fit_kde and check_if_anomaly work
    def _predict(x, verbose=0):
        # Generator stub passes MagicMock with `.samples`
        if hasattr(x, "samples"):
            n = x.samples
        else:  # list or array batch
            n = len(x)
        return np.zeros((n, 4, 4, 1), dtype=float)

    seq.predict.side_effect = _predict
    return seq


@pytest.fixture
def patched_tf(monkeypatch):
    """
    Patches for:
      * keras.models.load_model
      * tensorflow.keras.Sequential
      * ImageDataGenerator
      * sklearn.neighbors.KernelDensity
    so detector tests run without heavy deps.
    """
    # fake trained AE returned by load_model
    fake_model = _fake_sequential()
    monkeypatch.setattr("das_anomaly.detect.detector.load_model", lambda p: fake_model)

    # ----- Sequential used by _extract_encoder
    encoder_seq = _fake_sequential()
    monkeypatch.setattr("das_anomaly.detect.detector.Sequential", lambda: encoder_seq)

    # ImageDataGenerator.flow_from_directory
    flow = MagicMock(samples=4)  # four training images
    idg = MagicMock(flow_from_directory=lambda *a, **k: flow)
    monkeypatch.setattr(
        "das_anomaly.detect.detector.ImageDataGenerator", lambda *a, **k: idg
    )

    # KernelDensity.fit capture
    kde_fit_called = {}

    class FakeKDE:
        def __init__(self, *a, **k):
            pass

        def fit(self, x):
            kde_fit_called["shape"] = x.shape
            return self

        def score_samples(self, x):
            return np.array([0.9])

    monkeypatch.setattr("das_anomaly.detect.detector.KernelDensity", FakeKDE)

    return dict(
        fake_model=fake_model, encoder_seq=encoder_seq, kde_fit_called=kde_fit_called
    )
