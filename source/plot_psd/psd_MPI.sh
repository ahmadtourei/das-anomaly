#!/bin/bash
#SBATCH --ntasks=100
#SBATCH -t 48:00:00
#SBATCH -A 2206240948
#SBATCH --output=/u/pa/nb/tourei/scratch/sits/ae_anomaly_detection/outputs/spectrum_plots/dec22/first_week/output_MPI_%j.out
#SBATCH --mem-per-cpu=16G

# Print start time
echo "Job started at: $(date)"

# Load modules and environment
module load mpi/openmpi/gcc/3.1.3
source conda activate mpih5py

# Run the script
mpirun -n ${SLURM_NTASKS} python psd_MPI.py

# Print end time
echo "Job ended at: $(date)"