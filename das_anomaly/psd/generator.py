"""
Generate Power Spectral Density (PSD) plots from DAS data.

Example
-------
>>> from das_anomaly.psd import PSDConfig, PSDGenerator
>>> cfg = PSDConfig(data_path="~/data", psd_path="~/results/psd")
>>> PSDGenerator(cfg).run()
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import dascore as dc

from das_anomaly import plot_spec
from das_anomaly.settings import SETTINGS


@dataclass
class PSDConfig:
    """All knobs for the PSD workflow (values mirror SETTINGS defaults)."""
    data_path: Path | str = SETTINGS.DATA_PATH
    psd_path: Path | str = SETTINGS.PSD_PATH

    step_multiple: int = SETTINGS.STEP_MULTIPLE
    start_channel: int = SETTINGS.START_CHANNEL
    end_channel: int = SETTINGS.END_CHANNEL

    time_window: int = SETTINGS.TIME_WINDOW
    time_overlap: int = SETTINGS.TIME_OVERLAP
    dpi: int = SETTINGS.DPI

    # time selection (None ➜ no trimming)
    t1: Optional[np.datetime64] = getattr(SETTINGS, "T_1", None)
    t2: Optional[np.datetime64] = getattr(SETTINGS, "T_2", None)

    # derived / overrideable
    min_freq: float = 0.0
    max_freq_safety_factor: float = 0.9 # 0.9 × Nyquist by default

    def __post_init__(self):
        self.data_path = Path(self.data_path).expanduser()
        self.psd_path = Path(self.psd_path).expanduser()
        self.psd_path.mkdir(parents=True, exist_ok=True)


class PSDGenerator:
    """Orchestrates spool reading, preprocessing and PSD plotting."""

    def __init__(self, config: PSDConfig):
        self.cfg = config

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def run(self) -> None:
        """Entry point - iterate over patches and produce PSD plots."""
        for patch in self._iter_patches():
            self._plot_patch(patch)

    # --------------------------------------------------------------------- #
    # Internals
    # --------------------------------------------------------------------- #
    def _iter_patches(self):
        """Return an iterator of detrended strain-rate patches."""
        sp = dc.spool(self.cfg.data_path).update()

        patch0 = sp[0]
        sr = self._sampling_rate(patch0)

        sub_sp = sp.select(
            time=(self.cfg.t1, self.cfg.t2),
            distance=self._distance_slice(patch0),
        )
        # chunk into windowed sub‑patches
        for patch in (
            sub_sp.sort("time")
            .chunk(time=self.cfg.time_window, overlap=self.cfg.time_overlap)
        ):
            yield (
                patch.velocity_to_strain_rate_edgeless(
                    step_multiple=self.cfg.step_multiple
                ).detrend("time"),
                sr,
            )

    def _plot_patch(self, patch_sr_tuple):
        """Handle a single patch to PSD plot."""
        patch, sampling_rate = patch_sr_tuple

        max_freq = self.cfg.max_freq_safety_factor * 0.5 * sampling_rate
        title = patch.get_patch_name()

        plot_spec(
            patch,
            self.cfg.min_freq,
            max_freq,
            sampling_rate,
            title,
            output_rank=0,                 # keeps MPI + serial use happy
            fig_path=self.cfg.psd_path,
            dpi=self.cfg.dpi,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sampling_rate(patch):
        """
        Return sampling rate in Hz.

        Works whether coords.step("time") comes back as a numpy.timedelta64
        or a plain integer/float (already in seconds).
        """
        time_step = patch.coords.step("time")

        # timedelta64  ➜  seconds as float
        if isinstance(time_step, np.timedelta64):
            dt_seconds = time_step / np.timedelta64(1, "s")
        else:
            # assume numeric seconds (int, float, or numpy scalar)
            dt_seconds = float(time_step)

        if dt_seconds == 0:
            raise ValueError("Time step is zero; cannot compute sampling rate.")
        return int(round(1.0 / dt_seconds))

    def _distance_slice(self, patch):
        """Convert channel range to distance range."""
        start = patch.coords["distance"][self.cfg.start_channel]
        end = patch.coords["distance"][self.cfg.end_channel] 
        return (start, end)
