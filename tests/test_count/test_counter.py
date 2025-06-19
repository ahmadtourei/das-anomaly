"""
Unit-tests for das_anomaly.count.counter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from pathlib import Path

import pytest

from das_anomaly.count.counter import AnomalyCounter, CounterConfig


def _mk_txt(directory: Path, name: str, text: str) -> Path:
    """Create a tiny *.txt* file inside *directory* for test setup."""
    p = directory / name
    p.write_text(text)
    return p


def test_run_counts_and_writes(tmp_path: Path):
    """Three “anomaly” lines ⇒ correct summary string and 3-line file."""
    out_root = tmp_path / "out"
    target = out_root / "results"
    target.mkdir(parents=True)

    _mk_txt(target, "a.txt", "anomaly one\nno hit\nanomaly two\n")
    _mk_txt(target, "b.txt", "still no\nanomaly three\n")

    cfg = CounterConfig(
        results_path=out_root, results_folder_name="results", keyword="anomaly"
    )
    summary = AnomalyCounter(cfg).run()

    assert summary == "Total 'anomaly' lines: 3"
    assert cfg.summary_file.exists()
    assert cfg.summary_file.read_text().count("\n") == 2  # → 3 lines total


def test_custom_keyword(tmp_path: Path):
    """Case-sensitive search for 'foo' finds exactly two matches."""
    out_root = tmp_path / "out"
    target = out_root / "res"
    target.mkdir(parents=True)

    _mk_txt(target, "log.txt", "foo bar\nFOO bar\nfoo again\n")

    cfg = CounterConfig(results_path=out_root, results_folder_name="res", keyword="foo")
    assert AnomalyCounter(cfg).run() == "Total 'foo' lines: 2"


def test_missing_dir_raises(tmp_path: Path):
    """Creating CounterConfig on a non-existent folder should fail."""
    with pytest.raises(FileNotFoundError):
        CounterConfig(results_path=tmp_path, results_folder_name="does_not_exist")


def test_zero_matches(tmp_path: Path):
    """Zero matches ⇒ summary string with 0 and empty summary file."""
    out_root = tmp_path / "out"
    (out_root / "res").mkdir(parents=True)

    cfg = CounterConfig(
        results_path=out_root, results_folder_name="res", keyword="nowhere"
    )
    summary = AnomalyCounter(cfg).run()

    assert summary == "Total 'nowhere' lines: 0"
    assert cfg.summary_file.read_text() == ""
