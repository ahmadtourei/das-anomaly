"""
Generate Power Spectral Density (PSD) plots from DAS data.

Example
-------
>>> from das_anomaly.psd import PSDConfig, PSDGenerator
>>> cfg = PSDConfig()
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
except:

    class _DummyComm:
        """Stand-in that mimics the tiny subset we use."""

        def Get_rank(self):
            return 0

        def Get_size(self):
            return 1

        def bcast(self, obj, root=0):
            return obj

    MPI = type("FakeMPI", (), {"COMM_WORLD": _DummyComm()})()  # pragma: no cover
    print(
        "mpi4py not available - A fake MPI communicator is assigned. "
        "Install mpi4py and run your script under `mpirun` for parallel execution."
    )

from das_anomaly import get_psd_max_clip, plot_spec
from das_anomaly.settings import SETTINGS


@dataclass
class PSDConfig:
    """All knobs for the PSD workflow (values mirror SETTINGS defaults)."""

    data_unit: Path | str = SETTINGS.DATA_UNIT
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

    hide_axes: bool = True
    save_fig: bool = True

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
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        for patch in self._iter_patches_parallel(rank=rank, size=size):
            # Initiate MPI
            self._plot_patch(patch, rank=rank, mpi=True)

    def run_get_psd_val(self, percentile=95) -> None:
        """
        Entry point - iterate over patches and get the mean value of max
        value for clipping colorbar.
        """
        # unlink any previous index file for spool of BN_DATA_PATH
        _ = dc.spool(self.cfg.data_path).indexer.index_path.unlink()
        values: list[float] = []
        for patch in self._iter_patches(select_time=False):
            val = self._get_max_clip(patch, percentile=percentile)
            values.append(val)
        return float(np.mean(values)) if values else float("nan")

    # --------------------------------------------------------------------- #
    # Internals
    # --------------------------------------------------------------------- #
    def _iter_patches(self, select_time=True):
        """Yield detrended strain-rate patches."""
        data_path = self.cfg.data_path
        sp = dc.spool(data_path).update()

        if select_time:
            sub_sp_time = sp.select(time=(self.cfg.t1, self.cfg.t2))
            sub_sp_time_distance = sub_sp_time.select(
                distance=(self._distance_slice(sub_sp_time[0]))
            )
            sub_sp = sub_sp_time_distance
        else:
            sub_sp_distance = sp.select(distance=(self._distance_slice(sp[0])))
            sub_sp = sub_sp_distance
        # chunk into windowed sub-patches
        sub_sp_chunked = sub_sp.sort("time").chunk(
            time=self.cfg.time_window, overlap=self.cfg.time_overlap
        )
        if len(sub_sp_chunked) == 0:
            raise ValueError(
                f"No patch of DAS data found within data path: {data_path}"
            )
        # iterate over patches and perform preprocessing
        for patch in sub_sp_chunked:
            sr = self._sampling_rate(patch)
            if self.cfg.data_unit == "velocity":
                yield (
                    patch.velocity_to_strain_rate_edgeless(
                        step_multiple=self.cfg.step_multiple
                    ).detrend("time"),
                    sr,
                )
            elif self.cfg.data_unit == "strain_rate":
                yield (
                    patch.detrend("time"),
                    sr,
                )
            else:
                raise ValueError(
                    "Unsupported data_unit: {data_unit!r}. "
                    "Expected 'velocity' or 'strain_rate'."
                )

    def _iter_patches_parallel(self, rank, size):
        """
        Yield detrended strain-rate patches in an MPI-friendly round-robin.
        """
        if rank == 0:
            data_path = self.cfg.data_path
            sp = dc.spool(data_path).update()

            sub_sp_time = sp.select(time=(self.cfg.t1, self.cfg.t2))
            sub_sp_time_distance = sub_sp_time.select(
                distance=(self._distance_slice(sub_sp_time[0]))
            )

            sub_sp_chunked = sub_sp_time_distance.sort("time").chunk(
                time=self.cfg.time_window, overlap=self.cfg.time_overlap
            )
            if len(sub_sp_chunked) == 0:
                raise ValueError(
                    f"No patch of DAS data found within data path: {data_path}"
                )
        else:
            sub_sp_chunked = None
        # broadcast the variables to other ranks
        sub_sp_chunked = MPI.COMM_WORLD.bcast(sub_sp_chunked, root=0)

        # chunk into windowed sub-patches
        for i in range(rank, len(sub_sp_chunked), size):
            patch = sub_sp_chunked[i]
            sr = self._sampling_rate(patch)
            if self.cfg.data_unit == "velocity":
                yield (
                    patch.velocity_to_strain_rate_edgeless(
                        step_multiple=self.cfg.step_multiple
                    ).detrend("time"),
                    sr,
                )
            elif self.cfg.data_unit == "strain_rate":
                yield (
                    patch.detrend("time"),
                    sr,
                )
            else:
                raise ValueError(
                    "Unsupported data_unit: {data_unit!r}. "
                    "Expected 'velocity' or 'strain_rate'."
                )

    def _plot_patch(self, patch_sr_tuple, rank=0, mpi=False):
        """Handle a single patch to PSD plot."""
        patch, sampling_rate = patch_sr_tuple

        title = patch.get_patch_name()

        if mpi:
            output_rank = rank
        else:
            output_rank = 0
        plot_spec(
            patch,
            self.cfg.min_freq,
            self.cfg.max_freq,
            sampling_rate,
            title,
            output_rank=output_rank,  # keeps MPI + serial use happy
            fig_path=self.cfg.psd_path,
            dpi=self.cfg.dpi,
            hide_axes=self.cfg.hide_axes,
            save_fig=self.cfg.save_fig,
        )

    def _get_max_clip(self, patch_sr_tuple, percentile):
        """Handle a single patch to the max value for colorbar clipping."""
        patch, sampling_rate = patch_sr_tuple

        return get_psd_max_clip(
            patch,
            self.cfg.min_freq,
            self.cfg.max_freq,
            sampling_rate,
            percentile=percentile,
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

        if isinstance(time_step, np.timedelta64):
            # timedelta64  ➜  seconds as float
            dt_seconds = time_step / np.timedelta64(1, "s")
        elif time_step is None:
            # infer ∆t from the time coordinate
            t = patch.coords["time"]
            time_step = (t.max() - t.min()) / (len(t) - 1)
            dt_seconds = float(time_step / np.timedelta64(1, "s"))
        else:
            # Assume time_Step is already an int or float in seconds
            dt_seconds = float(time_step)

        if dt_seconds == 0:
            raise ValueError("Time step is zero; cannot compute sampling rate.")
        return round(1.0 / dt_seconds)

    def _distance_slice(self, patch):
        """Convert channel range to distance range."""
        start = patch.coords["distance"][self.cfg.start_channel]
        end = patch.coords["distance"][self.cfg.end_channel]
        return (start, end)
