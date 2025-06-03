from types import SimpleNamespace

# import pytest
import random

from das_anomaly.psd import PSDGenerator


class DummyComm:
    """Minimal MPI communicator with rank=0, size=1 behaviour."""
    def Get_rank(self):
        return 0
    def Get_size(self):
        return random.randint(1, 4)
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


