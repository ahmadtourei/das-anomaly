"""
das-anomaly package: A Python package for detecting anomalies in DAS data.
"""
import sys
import os
import warnings
from importlib.metadata import version

from .utils import calculate_percentile, check_if_anomaly, density, plot_spec, plot_train_test_loss, search_keyword_in_files


__all__ = [
    "calculate_percentile",
    "check_if_anomaly",
    "density",
    "plot_spec",
    "plot_train_test_loss",
    "search_keyword_in_files",
]

def ignore_hdf5_warning():
    """
    Suppresses HDF5 warning globally for the package.
    """
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    os.environ['HDF5_USE_FILE'] = 'FALSE'

# Run the warning suppressor during package initialization
ignore_hdf5_warning()

# Assign version
try:
    # Set the version dynamically from the package metadata
    __version__ = version("das-anomaly")
except Exception:
    __version__ = "unknown"
