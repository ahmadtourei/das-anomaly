"""
Plot power spectral density (PSD) plots with MPI parallelization.
"""

import os

import dascore as dc
import numpy as np

from das_anomaly import plot_spec

# Path to the data and results
data_path = "/path/to/the/das/data"
fig_path = "/path/to/the/saving/PSD/plots"

# Get the spool and create a sub-spool (if needed). E.g.:
sp = dc.spool(data_path).update()
t_1 = "2022-12-01 00:00:00"
t_2 = "2022-12-08 00:00:00"
sub_sp = sp.select(time=(t_1, t_2))
num_patch = len(sub_sp)

print("Totall number of patches in the spool: ", num_patch)

if num_patch < 1:
    raise ValueError("No patch of DAS data found within data path: %s" % (data_path))

patch = sub_sp[0]
time_step = patch.coords.step("time")
sampling_rate = int(1 / (time_step / np.timedelta64(1, "s")))

# Set parameters for preprocessing the data
step_multiple = 2  # gauge length to channel spacing ratio
start_channel = 0
end_channel = 800
min_freq = 0
max_freq = 0.9 * 0.5 * sampling_rate
time_window = 2  # sec.
time_overlap = 1  # sec.
dpi = 300  # saved image quality

# Loop over files (MPI)
for i in range(num_patch):
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
        if end_time > num_sec:
            continue
        # Get the base name of the file and construct the title with additional information
        title = os.path.basename(os.path.splitext(list(sub_sp.get_contents()["path"])[i])[0])
        title += f"_{str(start_time).zfill(num_digits)}-{str(end_time).zfill(num_digits)}sec"
        output_rank = 0 # used for MPI config
        plot_spec(
            patch_strain,
            start_time,
            end_time,
            start_channel,
            end_channel,
            min_freq,
            max_freq,
            sampling_rate,
            title,
            output_rank,
            fig_path,
            dpi,
        )

