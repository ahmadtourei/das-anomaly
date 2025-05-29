def test_cfg_defaults_roundtrip(cfg):
    """PSDConfig can be round‑tripped through `dict` without data loss."""
    d = cfg.__dict__
    assert d["data_path"].name == "terra15_das_1_trimmed.hdf5"
    assert d["psd_path"].name == "psd"
