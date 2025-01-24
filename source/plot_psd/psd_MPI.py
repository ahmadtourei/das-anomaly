"""
Plot power spectral density (PSD) plots with MPI parallelization.
"""
import numpy as np
import os
import sys
import time

import dascore as dc
from mpi4py import MPI

# Add the parent directory to the system path to enable importing modules from it.
current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.join(current_dir, '..')
sys.path.append(source_dir)
from utils import plot_spec


# Ignore hdf5 warning 
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
os.system('set HDF5_USE_FILE=FALSE')

# Intials
t0 = time.time()
data_path = '/u/pa/nb/tourei/scratch/sits/data/das_dec2022/'
fig_path = '/u/pa/nb/tourei/scratch/sits/ae_anomaly_detection/spectrum_plots/dec22/first_week/'

# Initiate MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Get the spool and some other info on the first rank
if rank==0:
    sp = dc.spool(data_path).update()
    t_1 = '2022-12-01 00:00:00'
    t_2 = '2022-12-08 00:00:00'    
    sub_sp = sp.select(time=(t_1, t_2))
    num_patch = len(sub_sp)
    
    print("Totall number of patches in the spool: ", num_patch)

    splits = num_patch
    if splits<1:
        raise ValueError('No patch of DAS data found within data path: %s'%(data_path))
    
    patch = sub_sp[0]
    time_step = patch.coords.step("time")
    sampling_rate = int(1/(time_step / np.timedelta64(1, 's')))
    distance_step = round(patch.coords.step("distance"), 4) 
else:
    splits = sub_sp = sampling_rate = num_sec = distance_step = None

# Broadcast the variables to other ranks
splits = comm.bcast(splits, root=0)
sub_sp = comm.bcast(sub_sp, root=0)
sampling_rate = comm.bcast(sampling_rate, root=0)
distance_step = comm.bcast(distance_step, root=0)

# Set parameters for preprocessing the data
step_multiple = 2 # gauge length to channel spacing ratio
start_channel = 0
end_channel = 800
min_freq = 0
max_freq = 0.9 * 0.5 * sampling_rate
time_window = 2 # sec.
time_overlap = 1 # sec.
dpi = 300 # saved image quality

# Loop over files (MPI)
for i in range(rank, splits, size):
  print(f"Rank {rank} is wroking on patch number: {i}")
  patch = sub_sp[i]
  # Velocity to strain rate
  patch_strain = patch.velocity_to_strain_rate_edgeless(step_multiple=step_multiple)
  patch_strain = patch_strain.detrend("time")

  # Get number of seconds in patch
  num_sec = patch_strain.seconds 
  num_digits = len(str(int(num_sec)))

  # Loop over time windows and plot the PSDs
  for j in range(0, int(num_sec), time_overlap):
    start_time = j
    end_time = start_time + time_window
    if end_time>num_sec:
       continue
    # Get the base name of the file and construct the title with additional information
    title = os.path.basename(os.path.splitext(list(sub_sp.get_contents()["path"])[i])[0])
    title += f"_{str(start_time).zfill(num_digits)}-{str(end_time).zfill(num_digits)}sec"
    output_rank = str(rank).zfill(int(len(str(int(size)))))
    plot_spec(patch_strain, start_time, end_time, start_channel, end_channel, min_freq, max_freq, sampling_rate, title, output_rank, fig_path, dpi)

sys.exit()
