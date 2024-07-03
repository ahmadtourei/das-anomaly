"""
Plot power spectral density (PSD) plots using MPI.
"""
import numpy as np
import os
import sys
import time

import dascore as dc
from mpi4py import MPI

current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.join(current_dir, '..')
sys.path.append(source_dir)
from utils import plot_spec


if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
os.system('set HDF5_USE_FILE=FALSE')

# Intials
t0 = time.time()
data_path = '/projects/casermminefiber/UTC-YMD20220617-HMS155316.989/'
fig_path = '/globalscratch/ahmad9/caserm/spectrum_analysis/spectrum_plots/UTC-YMD20220617-HMS155316.989/'

# Initiate MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank==0:
    sp = dc.spool(data_path).update()
    num_patch = len(sp)
    
    print("Number of patches in the spool: ", num_patch)

    splits = num_patch
    if splits<1:
        raise ValueError('No patch of DAS data found within data path: %s'%(data_path))
    
    patch = sp[0]
    time_step = patch.coords.step("time")
    sampling_rate = int(1/(time_step / np.timedelta64(1, 's')))
    num_sec = patch.seconds
    distance_step = round(patch.coords.step("distance"),5) 
else:
    splits = sp = sampling_rate = num_sec = distance_step = None

# Broadcast the variables
splits = comm.bcast(splits, root=0)
sp = comm.bcast(sp, root=0)
sampling_rate = comm.bcast(sampling_rate, root=0)
num_sec = comm.bcast(num_sec, root=0)
distance_step = comm.bcast(distance_step, root=0)

# Set parameters for preprocessing the data
gauge_length = distance_step*2
start_channel = 100
end_channel = 375
min_freq = 0
max_freq = 0.9*0.5*sampling_rate
time_window = 2 # sec
time_overlap = 1 # sec

# Loop over files (MPI)
for i in range(rank,splits,size):
  print(f"wroking on patch number: {i}")
  patch = sp[i].transpose("time", "distance")
  das_data = patch.data
  # Velocity to strain rate
  gauge_samples = int(round(gauge_length / distance_step))
  strain_rate = das_data[:, gauge_samples:] - das_data[:, :-gauge_samples]
  strain_rate = strain_rate / gauge_length

  num_digits = len(str(int(num_sec)))

  # Loop over time windows and plot the PSDs
  for j in range(0,int(num_sec),time_overlap):
    start_time = j
    end_time = start_time + time_window
    if end_time>num_sec:  
      continue
    title = os.path.splitext(str(sp.get_contents()["path"][i]))[0]+'_'+str(start_time).zfill(num_digits)+'-'+str(end_time).zfill(num_digits)+'sec'
    output_rank = str(rank).zfill(int(len(str(int(size)))))
    plot_spec(strain_rate, start_time, end_time, start_channel, end_channel, min_freq, max_freq, sampling_rate, title, output_rank, fig_path)

sys.exit()
