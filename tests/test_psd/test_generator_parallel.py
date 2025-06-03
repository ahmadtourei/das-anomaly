from types import SimpleNamespace

# import pytest
import random

from das_anomaly.psd import PSDGenerator


class DummyComm:
    """Minimal MPI communicator with rank=0, size=1 behaviour."""
    def Get_rank(self):
        return 0
    def Get_size(self):
        return random.randint(1, 10)
    def bcast(self, obj, root=0):
        return obj
    

# @pytest.fixture
# def patched_mpi(monkeypatch):
#     """Replace mpi4py.MPI with a dummy communicator."""
#     dummy_mpi = SimpleNamespace(COMM_WORLD=DummyComm())
#     monkeypatch.setattr("das_anomaly.psd.generator.MPI", dummy_mpi)
#     return dummy_mpi


def test_iter_patches_parallel(cfg, patched_plot_psd):
    gen = PSDGenerator(cfg)
    gen.run_parallel()

    # Same 14 calls as each processor should work on one patch
    assert patched_plot_psd.call_count == 14


# from collections import defaultdict
# from types import SimpleNamespace

# import pytest

# from das_anomaly.psd import PSDGenerator


# class DummyComm:
#     """
#     Minimal replacement for mpi4py.MPI that **remembers** the two objects
#     the root rank (0) sends out via `bcast`:

#         – the list of chunks   (1st call)
#         – the sampling rate    (2nd call)

#     All subsequent non‑root ranks receive the stored objects in the same order.
#     """
#     _stored = {}        # class‑level – shared by every instance

#     def __init__(self, rank: int, size: int):
#         self._rank = rank
#         self._size = size
#         self._next_key = 0   # 0 → "chunks", 1 → "sr"

#     def Get_rank(self):
#         return self._rank

#     def Get_size(self):
#         return self._size

#     def bcast(self, obj, root=0):
#         # Two keys in the order the generator calls bcast
#         keys = ("chunks", "sr")

#         if self._rank == root:
#             # root supplies the objects – remember them
#             DummyComm._stored[keys[self._next_key]] = obj
#             self._next_key ^= 1       # flip 0 ↔ 1
#             return obj

#         # non‑root: hand back the previously stored object
#         out = DummyComm._stored[keys[self._next_key]]
#         self._next_key ^= 1
#         return out


# def _run_and_collect(run_method, patched_plot, bucket: set[str]) -> None:
#     """
#     Execute `run_method` (serial or parallel) and add
#     `patch.get_patch_name()` for every plot call to `bucket`.
#     """
#     def _rec(patch, *_a, **_k):
#         bucket.add(patch.get_patch_name())
#     breakpoint()
#     patched_plot.side_effect = _rec
#     run_method()


# @pytest.mark.parametrize("size", [2, 3, 4])
# def test_iter_patches_parallel(cfg, patched_plot_psd, monkeypatch, size):
#     """All 14 chunks must be covered exactly once over all virtual ranks."""
#     serial_names = set()
#     _run_and_collect(PSDGenerator(cfg).run, patched_plot_psd, serial_names)
#     assert len(serial_names) == 14
#     patched_plot_psd.reset_mock()

#     seen_by_rank = defaultdict(set)
#     total_calls = 0

#     for rank in range(size):
#         monkeypatch.setattr(
#             "das_anomaly.psd.generator.MPI",
#             SimpleNamespace(COMM_WORLD=DummyComm(rank, size)),
#             raising=True,
#         )

#         _run_and_collect(
#             PSDGenerator(cfg).run_parallel,
#             patched_plot_psd,
#             seen_by_rank[rank],
#         )

#         total_calls += patched_plot_psd.call_count
#         patched_plot_psd.reset_mock()

#     union_ids = set().union(*seen_by_rank.values())
#     assert union_ids == serial_names            # no omission
#     for r1 in range(size):
#         for r2 in range(r1 + 1, size):
#             assert seen_by_rank[r1].isdisjoint(seen_by_rank[r2])   # no overlap
#     assert total_calls == 14                  # expected global call count
