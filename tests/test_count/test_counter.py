"""
Unit-tests for das_anomaly.count.counter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from pathlib import Path

import pytest

from das_anomaly.count.counter import CounterConfig, AnomalyCounter


def _mk_txt(directory: Path, name: str, text: str) -> Path:
    """Helper to create a small *.txt* file inside *directory*."""
    path = directory / name
    path.write_text(text)
    return path


def test_run_counts_and_writes(tmp_path: Path, capsys):
    """Happy-path: three “anomaly” lines → count 3, summary file with 3 lines."""
    out_root = tmp_path / "out"
    target = out_root / "results"
    target.mkdir(parents=True)

    _mk_txt(target, "a.txt", "anomaly one\nno hit\nanomaly two\n")
    _mk_txt(target, "b.txt", "still no\nanomaly three\n")

    cfg = CounterConfig(results_path=out_root,
                        results_folder_name="results",
                        keyword="anomaly")
    total = AnomalyCounter(cfg).run()

    assert total == 3
    assert cfg.summary_file.exists()
    assert cfg.summary_file.read_text().count("\n") == 2   # 3 lines
    assert "Total 'anomaly' lines: 3" in capsys.readouterr().out


def test_custom_keyword(tmp_path: Path):
    """Searching for lower-case 'foo' returns exactly two matches (case-sensitive)."""
    out_root = tmp_path / "out"
    target = out_root / "res"
    target.mkdir(parents=True)

    _mk_txt(target, "log.txt", "foo bar\nFOO bar\nfoo again\n")

    cfg = CounterConfig(results_path=out_root,
                        results_folder_name="res",
                        keyword="foo")
    assert AnomalyCounter(cfg).run() == 2


def test_missing_dir_raises(tmp_path: Path):
    """CounterConfig should raise FileNotFoundError if the results folder is absent."""
    with pytest.raises(FileNotFoundError):
        CounterConfig(results_path=tmp_path,
                      results_folder_name="does_not_exist")


def test_zero_matches(tmp_path: Path):
    """Zero matches → count 0 and summary file is empty."""
    out_root = tmp_path / "out"
    (out_root / "res").mkdir(parents=True)

    cfg = CounterConfig(results_path=out_root,
                        results_folder_name="res",
                        keyword="nowhere")
    total = AnomalyCounter(cfg).run()

    assert total == 0
    assert cfg.summary_file.read_text() == ""
