"""
Generate Power Spectral Density (PSD) plots from DAS data.

Example
-------
>>> from das_anomaly.psd import PSDConfig, PSDGenerator
>>> cfg = PSDConfig(data_path="~/data", psd_path="~/results/psd")
>>> # serial processing with single processor:
>>> PSDGenerator(cfg).run()
>>> # parallel processing with multiple processors using MPI:
>>> PSDGenerator(cfg).run_parallel()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import dascore as dc
import numpy as np

# optional MPI import
try:
    from mpi4py import MPI  # (we want the canonical upper-case name)
except ModuleNotFoundError:

    class _DummyComm:
        """Stand-in that mimics the tiny subset we use."""

        def get_rank(self):
            return 0

        def get_size(self):
            return 1

        def bcast(self, obj, root=0):
            return obj

    MPI = type("FakeMPI", (), {"COMM_WORLD": _DummyComm()})()  # pragma: no cover

from das_anomaly import get_psd_max_clip, plot_spec
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
    attr_t1 = getattr(SETTINGS, "T_1", None)
    attr_t2 = getattr(SETTINGS, "T_2", None)
    t1: np.datetime64 | None = attr_t1
    t2: np.datetime64 | None = attr_t2

    # derived / overrideable
    min_freq: float = SETTINGS.MIN_FREQ
    max_freq: float = SETTINGS.MAX_FREQ

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
        """
        Entry point - iterate over patches and produce PSD plots.
        """
        for patch in self._iter_patches():
            self._plot_patch(patch)

    def run_parallel(self) -> None:
        """
        Entry point - iterate over patches using MPI and produce PSD plots.
        Each patch is assigned to one MPI rank (i.e., processor).
        """
        for patch in self._iter_patches_parallel():
            self._plot_patch(patch)

    def run_get_psd_val(self) -> None:
        """
        Entry point - iterate over patches and get the mean value of max
        value for clipping colorbar.
        """
        values: list[float] = []
        for patch in self._iter_patches():
            val = self._get_max_clip(patch)
            values.append(val)
        return float(np.mean(values)) if values else float("nan")

    # --------------------------------------------------------------------- #
    # Internals
    # --------------------------------------------------------------------- #
    def _iter_patches(self):
        """Yield detrended strain-rate patches."""
        sp = dc.spool(self.cfg.data_path).update()

        patch0 = sp[0]
        sr = self._sampling_rate(patch0)

        sub_sp = sp.select(
            time=(self.cfg.t1, self.cfg.t2),
            distance=self._distance_slice(patch0),
        )
        # chunk into windowed sub-patches
        sub_sp_chunked = sub_sp.sort("time").chunk(
            time=self.cfg.time_window, overlap=self.cfg.time_overlap
        )
        if len(sub_sp_chunked) == 0:
            raise ValueError("No patch of DAS data found within data path: %s")
        # iterate over patches and perform preprocessing
        for patch in sub_sp_chunked:
            yield (
                patch.velocity_to_strain_rate_edgeless(
                    step_multiple=self.cfg.step_multiple
                ).detrend("time"),
                sr,
            )

    def _iter_patches_parallel(self, flag: bool = True):
        """
        Yield detrended strain-rate patches in an MPI-friendly round-robin.

        Parameters
        ----------
        flag : bool, default ``True``
            If *True*, each rank prints which patch number they are working on.
            Set to *False* to silence these progress messages.
        """
        # Initiate MPI
        comm = MPI.COMM_WORLD
        rank = comm.get_rank()
        size = comm.get_size()

        if rank == 0:
            sp = dc.spool(self.cfg.data_path).update()

            patch0 = sp[0]
            sr = self._sampling_rate(patch0)

            sub_sp = sp.select(
                time=(self.cfg.t1, self.cfg.t2),
                distance=self._distance_slice(patch0),
            )

            sub_sp_chunked = sub_sp.sort("time").chunk(
                time=self.cfg.time_window, overlap=self.cfg.time_overlap
            )
            if len(sub_sp_chunked) == 0:
                raise ValueError("No patch of DAS data found within data path: %s")
        else:
            sub_sp_chunked = sr = None

        # broadcast the variables to other ranks
        sub_sp_chunked = comm.bcast(sub_sp_chunked, root=0)
        sr = comm.bcast(sr, root=0)

        # chunk into windowed sub-patches
        for i in range(rank, len(sub_sp_chunked), size):
            if flag:
                pass
            patch = sub_sp_chunked[i]
            yield (
                patch.velocity_to_strain_rate_edgeless(
                    step_multiple=self.cfg.step_multiple
                ).detrend("time"),
                sr,
            )

    def _plot_patch(self, patch_sr_tuple):
        """Handle a single patch to PSD plot."""
        patch, sampling_rate = patch_sr_tuple

        title = patch.get_patch_name()

        plot_spec(
            patch,
            self.cfg.min_freq,
            self.cfg.max_freq,
            sampling_rate,
            title,
            output_rank=0,  # keeps MPI + serial use happy
            fig_path=self.cfg.psd_path,
            dpi=self.cfg.dpi,
        )

    def _get_max_clip(self, patch_sr_tuple):
        """Handle a single patch to the max value for colorbar clipping."""
        patch, sampling_rate = patch_sr_tuple

        return get_psd_max_clip(
            patch,
            self.cfg.min_freq,
            self.cfg.max_freq,
            sampling_rate,
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
        return round(1.0 / dt_seconds)

    def _distance_slice(self, patch):
        """Convert channel range to distance range."""
        start = patch.coords["distance"][self.cfg.start_channel]
        end = patch.coords["distance"][self.cfg.end_channel]
        return (start, end)
