from pathlib import Path
from unittest.mock import patch

import pytest
from das_anomaly.train import TrainSplitConfig, ImageSplitter


def _build_fake_psd_tree(tmp_path: Path, n_png: int):
    """Generate `n_png` fake .png files plus some non-PNG noise."""
    (tmp_path / "sub").mkdir()
    for i in range(n_png):
        (tmp_path / ("sub" if i % 2 else "") / f"img_{i}.png").write_bytes(b"")
    # add two non-png files
    (tmp_path / "skip.txt").write_text("ignore")
    (tmp_path / "sub" / "skip.jpg").write_bytes(b"")


@pytest.fixture
def cfg(tmp_path):
    psd_root = tmp_path / "psd"
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    psd_root.mkdir()
    _build_fake_psd_tree(psd_root, 20)
    return TrainSplitConfig(
        psd_dir=psd_root,
        train_dir=train_dir,
        test_dir=test_dir,
        num_images=10,
        ratio=0.3,
        rng_seed=123,
    )


def test_pick_files_counts(cfg):
    """_pick_files returns correct split sizes + reproducibility."""
    splitter = ImageSplitter(cfg)
    train, test = splitter._pick_files()
    assert len(train) == 7 and len(test) == 3

    # same seed ➜ same selection
    splitter2 = ImageSplitter(cfg)
    assert splitter._pick_files() == splitter2._pick_files()


def test_run_copies_files(cfg, monkeypatch):
    """`run()` should call shutil.copy the expected number of times."""
    copier_calls = []

    def fake_copy(src, dest):
        copier_calls.append((Path(src).name, Path(dest)))

    with patch("das_anomaly.train.split_images.shutil.copy", side_effect=fake_copy):
        ImageSplitter(cfg).run()

    # 10 images copied in total
    assert len(copier_calls) == 10
    # first 7 to train_dir, last 3 to test_dir (order from _pick_files)
    assert all(dest == cfg.train_dir for _, dest in copier_calls[:7])
    assert all(dest == cfg.test_dir for _, dest in copier_calls[7:])
