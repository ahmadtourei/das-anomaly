"""
Unit-tests for das_anomaly.count.counter.
"""

from pathlib import Path

from das_anomaly.count.counter import AnomalyCounter, CounterConfig


def _mk_txt(where: Path, name: str, text: str) -> None:
    """Write a small *.txt* file into *where* (top level only)."""
    (where / name).write_text(text)


class TestAnomalyCounter:
    """Happy-path and edge-case checks for the counter module."""

    def test_run_counts_and_writes(self, tmp_path: Path):
        """Three matching lines → summary says 3 and file holds 4 lines."""
        out_root = tmp_path / "out"
        out_root.mkdir()

        # keyword must be *last* token in the matching lines
        _mk_txt(out_root, "a.txt", "one anomaly\nno hit here\nanother anomaly\n")
        _mk_txt(out_root, "b.txt", "still no\nthird anomaly\n")

        cfg = CounterConfig(results_path=out_root, keyword="anomaly")
        summary = AnomalyCounter(cfg).run()

        expected_msg = (
            f"Total detected 'anomaly': 3\n" f"Text file saved at {cfg.summary_file}"
        )
        assert summary == expected_msg

        text_lines = cfg.summary_file.read_text().splitlines()
        # 3 matches + summary line
        assert len(text_lines) == 4
        assert text_lines[-1] == "Total detected 'anomaly': 3"

    def test_custom_keyword(self, tmp_path: Path):
        """Case-sensitive search for 'foo' finds exactly two matches."""
        out_root = tmp_path / "out"
        out_root.mkdir()

        _mk_txt(out_root, "x.txt", "spam foo\nFOO ignored\nbar foo\n")

        cfg = CounterConfig(results_path=out_root, keyword="foo")
        expected = f"Total detected 'foo': 2\n" f"Text file saved at {cfg.summary_file}"
        assert AnomalyCounter(cfg).run() == expected

    def test_target_dir_auto_created(self, tmp_path: Path):
        """CounterConfig always creates *count* folder."""
        cfg = CounterConfig(results_path=tmp_path, keyword="x")
        assert cfg.target_dir.exists()

    def test_zero_matches(self, tmp_path: Path):
        """No match → summary file contains only the summary line."""
        out_root = tmp_path / "out"
        out_root.mkdir()

        cfg = CounterConfig(results_path=out_root, keyword="none")
        ret = AnomalyCounter(cfg).run()

        expected = (
            f"Total detected 'none': 0\n" f"Text file saved at {cfg.summary_file}"
        )
        assert ret == expected
        assert cfg.summary_file.read_text() == "Total detected 'none': 0\n"
