#!/bin/bash
#SBATCH -N1
#SBATCH --ntasks-per-node=1
#SBATCH -t 144:00:00
#SBATCH -A 2206240948
#SBATCH --output=/u/pa/nb/tourei/scratch/caserm/spectrum_analysis/background_noise/outputs/output_train_%j.out
#SBATCH --mem-per-cpu=10G

# Print start time
echo "Job started at: $(date)"

source /u/pa/nb/tourei/anaconda3/condabin/conda
source activate mpih5py

python train.py

# Print end time
echo "Job ended at: $(date)"
