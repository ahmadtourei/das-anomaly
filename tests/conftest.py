from unittest.mock import MagicMock

import numpy as np
import pytest
from dascore.compat import random_state
from dascore.core import Patch
from dascore.io.core import read
from dascore.utils.downloader import fetch
from keras import Model
from keras import layers as L
from keras.layers import Conv2D as _Conv2D
from PIL import Image
from sklearn.neighbors import KernelDensity as SKKDE

import das_anomaly.detect.detector as det_mod
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
    size = 8
    x_in = L.Input((size, size, 3), name="input")
    x = L.Conv2D(8, 3, padding="same", activation="relu", name="conv1")(x_in)
    x = L.MaxPooling2D(2, padding="same", name="pool1")(x)
    x = L.Conv2D(4, 3, padding="same", activation="relu", name="conv2")(x)
    x = L.MaxPooling2D(2, padding="same", name="pool2")(x)
    x = L.UpSampling2D(2, name="up1")(x)  # decoder start
    fake_model = Model(x_in, x, name="ae_tiny")

    # give source model some weights
    for la in fake_model.layers:
        if la.weights:
            la.set_weights([np.ones_like(w) for w in la.get_weights()])

    # make your code load this model
    monkeypatch.setattr(det_mod, "load_model", lambda path: fake_model)

    # mark when set_weights is called on cloned Conv2D layers
    _orig_set = _Conv2D.set_weights

    def _track_set(self, weights):
        self._set_called = True
        return _orig_set(self, weights)

    monkeypatch.setattr(_Conv2D, "set_weights", _track_set, raising=False)

    # capture KernelDensity.fit input shape <<<
    kde_fit_called = {}
    _orig_kde_fit = SKKDE.fit

    def _fit_and_log(self, X, y=None):
        kde_fit_called["shape"] = getattr(X, "shape", None)
        return _orig_kde_fit(self, X, y)

    monkeypatch.setattr(SKKDE, "fit", _fit_and_log, raising=True)

    return {"fake_model": fake_model, "kde_fit_called": kde_fit_called}
