import numpy as np

from das_anomaly.psd import PSDGenerator


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
