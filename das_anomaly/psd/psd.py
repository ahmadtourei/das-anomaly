"""
Generate power spectral density (PSD) plots.
"""
import numpy as np

import dascore as dc
from das_anomaly import plot_spec
from das_anomaly.settings import SETTINGS


# Path to the data and results
data_path = SETTINGS.DATA_PATH
psd_path = SETTINGS.PSD_PATH

# Set parameters for preprocessing the data
step_multiple = SETTINGS.STEP_MULTIPLE
start_channel = SETTINGS.START_CHANNEL
end_channel = SETTINGS.END_CHANNEL
time_window = SETTINGS.TIME_WINDOW
time_overlap = SETTINGS.TIME_OVERLAP  
dpi = SETTINGS.DPI  

# Get the spool and create a sub-spool (if needed). E.g.:
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

# Define min and max frequencies for PSDs
min_freq = 0
max_freq = 0.9 * 0.5 * sampling_rate

# Loop over files (MPI)
for patch in sub_sp_chunked:
    # Velocity to strain rate
    patch_strain = patch.velocity_to_strain_rate_edgeless(step_multiple=step_multiple)
    patch_strain_detrended = patch_strain.detrend("time")

    # Get number of seconds in patch
    num_sec = patch_strain_detrended.seconds

    # Get the base name of the patch
    title = patch.get_patch_name()
    output_rank = 0 # to make the plot_spec works for MPI and non-MPI use
    plot_spec(
        patch_strain_detrended,
        min_freq,
        max_freq,
        sampling_rate,
        title,
        output_rank,
        psd_path,
        dpi,
    )
