from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from das_anomaly.psd import PSDConfig, PSDGenerator


def test_sampling_rate(dummy_patch):
    sr = PSDGenerator._sampling_rate(dummy_patch)
    # 10 Hz
    assert sr == 10


def test_distance_slice(cfg, dummy_patch):
    gen = PSDGenerator(cfg)
    lo, hi = gen._distance_slice(dummy_patch)
    # start_channel=0, end_channel=10, dx=0.25 m
    assert np.isclose(lo, 0.0)
    assert np.isclose(hi, 2.0)


class TestRunGetPsdVal:
    """Validate that run_get_psd_val returns the expected scalar."""

    def test_returns_mean_of_clips(self, tmp_path: Path, monkeypatch):
        """If get_psd_max_clip yields [10, 20] the method returns 15.0."""
        cfg = PSDConfig(data_path=tmp_path, psd_path=tmp_path)
        gen = PSDGenerator(cfg)

        # stub _iter_patches → two dummy patch/sr tuples
        monkeypatch.setattr(
            gen,
            "_iter_patches",
            lambda: iter([("dummy1", 100), ("dummy2", 100)]),
        )

        # make get_psd_max_clip return 10 then 20
        side_vals = [10.0, 20.0]

        def fake_clip(*_):
            return side_vals.pop(0)

        monkeypatch.setattr("das_anomaly.psd.generator.get_psd_max_clip", fake_clip)

        mean_val = gen.run_get_psd_val()
        assert mean_val == 15.0

    def test_no_patches_returns_nan(self, tmp_path: Path, monkeypatch):
        """Empty iterator → result is NaN and get_psd_max_clip is never called."""
        cfg = PSDConfig(data_path=tmp_path, psd_path=tmp_path)
        gen = PSDGenerator(cfg)

        # empty iterator
        monkeypatch.setattr(gen, "_iter_patches", lambda: iter([]))

        called = {}

        def fake_clip(*_):
            called["hit"] = True
            return 0.0

        monkeypatch.setattr("das_anomaly.psd.generator.get_psd_max_clip", fake_clip)

        mean_val = gen.run_get_psd_val()
        assert np.isnan(mean_val)
        assert "hit" not in called  # function never invoked


"""
Tests for the `data_unit` switch inside PSDGenerator._iter_patches.
"""
# ------------------------------------------------------------------ #
# helpers to fabricate a minimal Patch + Spool                       #
# ------------------------------------------------------------------ #


def _fake_sp(patch):
    """Return a minimal object with the methods PSDGenerator expects."""

    class FakeSpool(list):
        def update(self):
            return self

        def select(self, **_):
            return self

        def sort(self, *_, **__):
            return self

        def chunk(self, **_):
            return self

    return FakeSpool([patch])


# ------------------------------------------------------------------ #
# the actual tests                                                   #
# ------------------------------------------------------------------ #
@pytest.mark.parametrize(
    "unit,conv_calls,det_calls",
    [
        ("velocity", 1, 1),  # conversion + detrend
        ("strain_rate", 0, 1),  # detrend only
    ],
)
def test_iter_patches_respects_data_unit(
    unit, conv_calls, det_calls, dummy_patch, monkeypatch
):
    """_iter_patches branches correctly for 'velocity' vs 'strain_rate'."""
    from das_anomaly.psd.generator import PSDConfig, PSDGenerator

    # --- wire a fake spool so no disk access happens ----------------
    monkeypatch.setattr(
        "das_anomaly.psd.generator.dc.spool",
        lambda *_: _fake_sp(dummy_patch),
    )

    # stub helpers not under test
    monkeypatch.setattr(PSDGenerator, "_distance_slice", lambda *_: (0, 1))
    monkeypatch.setattr(PSDGenerator, "_sampling_rate", lambda *_: 1000)

    # spy counters ---------------------------------------------------
    counters = {"conv": 0, "det": 0}

    def fake_conv(self, *_, **__):
        counters["conv"] += 1
        return self

    def fake_det(self, *_, **__):
        counters["det"] += 1
        return self

    monkeypatch.setattr(
        dummy_patch, "velocity_to_strain_rate_edgeless", fake_conv.__get__(dummy_patch)
    )
    monkeypatch.setattr(
        dummy_patch, "detrend", fake_det.__get__(dummy_patch, SimpleNamespace)
    )

    # --- run --------------------------------------------------------
    cfg = PSDConfig(data_path="unused", data_unit=unit)
    list(PSDGenerator(cfg)._iter_patches())  # exhaust generator

    # --- assertions -------------------------------------------------
    assert counters["conv"] == conv_calls
    assert counters["det"] == det_calls
