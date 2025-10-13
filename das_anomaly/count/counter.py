from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from das_anomaly import search_keyword_in_files
from das_anomaly.settings import SETTINGS

# Helpers
_PATH_RE = re.compile(r"(/[^:\n]+?\.(?:png|jpg|jpeg|tif|tiff|bmp))", re.IGNORECASE)
_TIME_RE = re.compile(
    r"(?P<y1>\d{4})_(?P<m1>\d{2})_(?P<d1>\d{2})T(?P<H1>\d{2})_(?P<M1>\d{2})_(?P<S1>\d{2})__"
    r"(?P<y2>\d{4})_(?P<m2>\d{2})_(?P<d2>\d{2})T(?P<H2>\d{2})_(?P<M2>\d{2})_(?P<S2>\d{2})"
)


def _extract_path(line: str) -> str | None:
    m = _PATH_RE.search(line)
    return m.group(1) if m else None


def _parse_window_from_path(p: str) -> tuple[datetime, datetime] | None:
    base = Path(p).name
    m = _TIME_RE.search(base)
    if not m:
        return None
    try:
        start = datetime(
            int(m["y1"]),
            int(m["m1"]),
            int(m["d1"]),
            int(m["H1"]),
            int(m["M1"]),
            int(m["S1"]),
        )
        end = datetime(
            int(m["y2"]),
            int(m["m2"]),
            int(m["d2"]),
            int(m["H2"]),
            int(m["M2"]),
            int(m["S2"]),
        )
        return start, end
    except ValueError:
        return None


# Configuration object                                               #
@dataclass
class CounterConfig:
    results_path: Path | str = SETTINGS.RESULTS_PATH
    keyword: str = "anomaly"

    # counting behavior
    collapse_transients: bool = False
    max_gap_seconds: int = 0

    # NEW: also report transient vs impulsive breakdown
    classify_types: bool = False  # if True, report counts for transient & impulsive

    def __post_init__(self):
        self.results_path = Path(self.results_path).expanduser()
        self.target_dir = self.results_path / "count"
        self.target_dir.mkdir(parents=True, exist_ok=True)

    @property
    def summary_file(self) -> Path:
        if self.classify_types:
            suffix = "_classified"
        else:
            suffix = "_collapsed" if self.collapse_transients else ""
        return self.target_dir / f"Counted_{self.keyword}{suffix}.txt"


# Counter                                                            #
class AnomalyCounter:
    def __init__(self, cfg: CounterConfig):
        self.cfg = cfg

    def _prepare_items(self, lines: list[str]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for ln in lines:
            p = _extract_path(ln)
            if not p:
                continue
            tw = _parse_window_from_path(p)
            # sort by start-time when available, else by BASENAME (deterministic)
            name = Path(p).name
            sort_key = (tw[0], tw[1]) if tw else (datetime.max, name)
            items.append({"line": ln, "path": p, "tw": tw, "sort_key": sort_key})
        items.sort(key=lambda d: d["sort_key"])
        return items

    def _group_windows(self, items: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """Group overlapping/adjacent windows using max_gap_seconds."""
        if not items:
            return []
        tol = timedelta(seconds=max(0, int(self.cfg.max_gap_seconds)))
        groups: list[list[dict[str, Any]]] = []
        for it in items:
            if not groups:
                groups.append([it])
                continue
            prev = groups[-1][-1]
            prev_tw, curr_tw = prev["tw"], it["tw"]
            if prev_tw and curr_tw and (curr_tw[0] <= (prev_tw[1] + tol)):
                groups[-1].append(it)
            else:
                groups.append([it])
        return groups

    def _group_transients(self, lines: list[str]) -> tuple[int, list[str]]:
        """(Existing) return collapsed count and one representative per group."""
        items = self._prepare_items(lines)
        groups = self._group_windows(items)
        reps = [g[0]["line"] for g in groups]
        return len(groups), reps

    # NEW: classify transient vs impulsive
    def _classify_groups(
        self, lines: list[str]
    ) -> tuple[int, int, int, list[str], list[str]]:
        """
        Return
        ------
            n_groups, n_transient, n_impulsive, transient_reps, impulsive_reps
        """
        items = self._prepare_items(lines)
        groups = self._group_windows(items)

        n_groups = len(groups)
        transient_reps: list[str] = []
        impulsive_reps: list[str] = []

        for g in groups:
            if len(g) >= 2:
                transient_reps.append(g[0]["line"])
            else:
                impulsive_reps.append(g[0]["line"])

        n_transient = len(transient_reps)
        n_impulsive = len(impulsive_reps)
        return n_groups, n_transient, n_impulsive, transient_reps, impulsive_reps

    def run(
        self,
        collapse_transients: bool | None = None,
        classify_types: bool | None = None,
    ) -> str:
        flag_collapse = (
            self.cfg.collapse_transients
            if collapse_transients is None
            else bool(collapse_transients)
        )
        flag_classify = (
            self.cfg.classify_types if classify_types is None else bool(classify_types)
        )

        total_raw, lines = search_keyword_in_files(
            self.cfg.results_path, self.cfg.keyword
        )

        out_lines: list[str] = []
        if lines:
            out_lines.append("# Raw matches")
            out_lines.extend(lines)
            out_lines.append("")

        if flag_classify:
            n_groups, n_transient, n_impulsive, trans_reps, imp_reps = (
                self._classify_groups(lines)
            )
            summary_line = (
                f"Total '{self.cfg.keyword}' events (grouped): {n_groups}; raw hits: {total_raw}"
                f"(transient groups ≥2: {n_transient}, impulsive singles: {n_impulsive})"
            )
            if trans_reps:
                out_lines.append("# Transient representatives (one per group, size ≥2)")
                out_lines.extend(trans_reps)
                out_lines.append("")
            if imp_reps:
                out_lines.append("# Impulsive representatives (group size = 1)")
                out_lines.extend(imp_reps)
                out_lines.append("")

        elif flag_collapse:
            collapsed_total, reps = self._group_transients(lines)
            summary_line = (
                f"Total detected '{self.cfg.keyword}' (collapsed transients): {collapsed_total} "
                f"(raw hits: {total_raw})"
            )
            if reps:
                out_lines.append("# Collapsed representatives (one per group)")
                out_lines.extend(reps)
                out_lines.append("")
        else:
            summary_line = f"Total detected '{self.cfg.keyword}': {total_raw}"

        out_lines.append(summary_line)

        with self.cfg.summary_file.open("w") as fh:
            fh.write("\n".join(out_lines) + "\n")

        return summary_line + "\n" + f"Text file saved at {self.cfg.summary_file}"
