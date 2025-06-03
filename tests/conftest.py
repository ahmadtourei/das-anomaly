from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

import dascore as dc
from dascore.clients.dirspool import DirectorySpool
from dascore.compat import random_state
from dascore.core import Patch
from dascore.io.core import read
from dascore.utils.downloader import fetch
from das_anomaly.psd import PSDConfig      


@pytest.fixture()
def dummy_patch() -> Patch:
    """
    Return a small simple array for memory testing.

    Note: This cant be a fixture as pytest seems to hold a reference,
    even for function scoped outputs.
    """
    attrs = {'data_type': 'velocity', 'time_step': 0.1}
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
