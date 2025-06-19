"""
das_anomaly.count_counter
~~~~~~~~~~~~~~~~~~~~~~~~~

Count how many lines in the *results* folder contain the keyword
“anomaly” (or any keyword you choose) and write a summary file.

Example
-------
>>> from das_anomaly.count_counter import CounterConfig, AnomalyCounter
>>> total = AnomalyCounter(CounterConfig(keyword="anomaly")).run()
>>> print(total)
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
    results_folder_name: str = SETTINGS.RESULTS_FOLDER_NAME

    # search term
    keyword: str = "anomaly"

    def __post_init__(self):
        self.results_path = Path(self.results_path).expanduser()
        self.target_dir = self.results_path / self.results_folder_name
        if not self.target_dir.exists():
            raise FileNotFoundError(self.target_dir)

    # derived output file
    @property
    def summary_file(self) -> Path:
        """Configurate file path for writing counted results."""
        return self.results_path / f"{self.keyword}_{self.results_folder_name}.txt"


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
        total, lines = search_keyword_in_files(self.cfg.target_dir, self.cfg.keyword)

        with self.cfg.summary_file.open("w") as fh:
            fh.write("\n".join(lines))
        sr = f"Total '{self.cfg.keyword}' lines: {total}"
        return sr
