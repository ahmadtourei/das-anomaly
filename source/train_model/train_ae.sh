#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH -t 48:00:00
#SBATCH -A 2206240948
#SBATCH --output=/u/pa/nb/tourei/scratch/sits/ae_anomaly_detection/outputs/train/dec22/first_week/output_MPI_%j.out
#SBATCH --mem-per-cpu=16G

# Print start time
echo "Job started at: $(date)"

# Load modules and environment
source /u/pa/nb/tourei/anaconda3/condabin/conda
source activate mpih5py

# Run the script
python train_ae.py

# Print end time
echo "Job ended at: $(date)"
