"""
das_anomaly.count_counter
~~~~~~~~~~~~~~~~~~~~~~~~~

Count how many lines in the *results* folder contain the keyword
“anomaly” (or any keyword you choose) and write a summary file.

Example
-------
>>> from das_anomaly.count.counter import CounterConfig, AnomalyCounter
>>> cfg = CounterConfig(keyword="anomaly")
>>> anomalies = AnomalyCounter(cfg).run()
>>> print(anomalies)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from das_anomaly import search_keyword_in_files
from das_anomaly.settings import SETTINGS


# ------------------------------------------------------------------ #
# Configuration object                                               #
# ------------------------------------------------------------------ #
@dataclass
class CounterConfig:
    """Where to look and what to search for."""

    # folders
    results_path: Path | str = SETTINGS.RESULTS_PATH

    # search term
    keyword: str = "anomaly"

    def __post_init__(self):
        self.results_path = Path(self.results_path).expanduser()
        self.target_dir = self.results_path / "count"
        self.target_dir.mkdir(parents=True, exist_ok=True)

    # derived output file
    @property
    def summary_file(self) -> Path:
        """Configurate file path for writing counted results."""
        return self.target_dir / f"Counted_{self.keyword}.txt"


# ------------------------------------------------------------------ #
# Counter                                                            #
# ------------------------------------------------------------------ #
class AnomalyCounter:
    """
    Tally keyword hits in text outputs from *detect_anomalies*.

    Parameters
    ----------
    cfg:
        A :class:`CounterConfig` instance describing paths and keyword.

    Returns
    -------
    int
        Total number of matching lines.
    """

    def __init__(self, cfg: CounterConfig):
        self.cfg = cfg

    def run(self) -> int:
        """Perform the search, write a summary file, and return the count."""
        total, lines = search_keyword_in_files(self.cfg.results_path, self.cfg.keyword)

        summary_line = f"Total detected '{self.cfg.keyword}': {total}"

        with self.cfg.summary_file.open("w") as fh:
            if lines:
                fh.write("\n".join(lines) + "\n")
            fh.write(summary_line + "\n")

        result_msg = summary_line + "\n" + f"Text file saved at {self.cfg.summary_file}"
        return result_msg
