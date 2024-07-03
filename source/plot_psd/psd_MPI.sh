#!/bin/bash
#SBATCH --ntasks=250
#SBATCH -t 12:00:00
#SBATCH -A casermminefiber
#SBATCH --output=/globalscratch/ahmad9/caserm/spectrum_analysis/outputs/output_MPI_%j.out
#SBATCH --mem-per-cpu=16G

# Print start time
echo "Job started at: $(date)"

module load openmpi/gcc/64/4.0.4
source activate casermmpi
mpirun -n ${SLURM_NTASKS} python spectrum_MPI.py

# Print end time
echo "Job ended at: $(date)"