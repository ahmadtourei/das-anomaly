#!/bin/bash
#SBATCH --ntasks=24
#SBATCH -t 6:00:00
#SBATCH -A YOUR_ACCOUNT
#SBATCH --mem-per-cpu=16G

# Print start time
echo "Job started at: $(date)"

# Load modules and environment
module load openmpi/gcc/64/4.1.5
source activate dasanomaly

# Recommended in a SLURM environment: use srun, not mpirun
mpirun -n $SLURM_NTASKS python -u psd_parallel.py

# Print end time
echo "Job ended at: $(date)"
