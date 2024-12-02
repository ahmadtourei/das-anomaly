#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
#SBATCH -t 12:00:00
#SBATCH -A 2206240948
#SBATCH --output=/u/pa/nb/tourei/scratch/sits/ae_anomaly_detection/outputs/train/dec22/first_week/output_train_gpu_%j.out
#SBATCH --mem-per-cpu=40G

# Print start time
echo "Job started at: $(date)"

# Load modules and environment
source /u/pa/nb/tourei/anaconda3/condabin/conda
source activate tf-gpu

# Run the script
python train_ae.py

# Print end time
echo "Job ended at: $(date)"
