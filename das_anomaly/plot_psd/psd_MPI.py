"""
Generate power spectral density (PSD) plots with MPI parallelization.
"""
import numpy as np
import sys

import dascore as dc
from mpi4py import MPI
from das_anomaly import plot_spec
from das_anomaly.settings import SETTINGS


# Path to the data and results
data_path = SETTINGS.DATA_PATH
psd_dir = SETTINGS.FIG_PATH

# Set parameters for preprocessing the data
step_multiple = SETTINGS.STEP_MULTIPLE
start_channel = SETTINGS.START_CHANNEL
end_channel = SETTINGS.END_CHANNEL
time_window = SETTINGS.TIME_WINDOW
time_overlap = SETTINGS.TIME_OVERLAP  
dpi = SETTINGS.DPI  

# Initiate MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Get the spool and create a sub-spool (if needed) on the first rank. E.g.:
if rank == 0:
    sp = dc.spool(data_path).update()
    patch = sp[0]
    time_step = patch.coords.step("time")
    sampling_rate = int(1 / (time_step / np.timedelta64(1, "s")))
    distance_step = np.round(patch.coords.step("distance"), 3)
    t_1 = SETTINGS.T_1
    t_2 = SETTINGS.T_2
    sub_sp = sp.select(time=(t_1, t_2), distance=(start_channel*distance_step, end_channel*distance_step))
    # Chunk the spool to sub-patches with time_window size
    sub_sp_chunked = sub_sp.sort("time").chunk(time=time_window, overlap=time_overlap) 
    num_patch = len(sub_sp_chunked)

    print("Totall number of patches in the spool: ", num_patch)

    splits = num_patch
    if splits < 1:
        raise ValueError("No patch of DAS data found within data path: %s" % (data_path))

else:
    splits = sub_sp_chunked = sampling_rate = distance_step = num_sec = None

# Broadcast the variables to other ranks
splits = comm.bcast(splits, root=0)
sub_sp_chunked = comm.bcast(sub_sp_chunked, root=0)
sampling_rate = comm.bcast(sampling_rate, root=0)
distance_step = comm.bcast(distance_step, root=0)

# Define min and max frequencies for PSDs
min_freq = 0
max_freq = 0.9 * 0.5 * sampling_rate

# Loop over files (MPI)
for i in range(rank, splits, size):
    print(f"Rank {rank} is wroking on patch number: {i}")
    patch = sub_sp_chunked[i]
    # Velocity to strain rate
    patch_strain = patch.velocity_to_strain_rate_edgeless(step_multiple=step_multiple)
    patch_strain_detrended = patch_strain.detrend("time")

    # Get number of seconds in patch
    num_sec = patch_strain_detrended.seconds

    # Get the base name of the patch
    title = patch.get_patch_name()
    output_rank = str(rank).zfill(int(len(str(int(size)))))
    plot_spec(
        patch_strain_detrended,
        min_freq,
        max_freq,
        sampling_rate,
        title,
        output_rank,
        fig_path,
        dpi,
    )

sys.exit()
