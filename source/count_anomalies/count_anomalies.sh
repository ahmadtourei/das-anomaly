#!/bin/bash
#SBATCH -t 00:10:00
#SBATCH -A casermminefiber
#SBATCH --output=/globalscratch/ahmad9/caserm/spectrum_analysis/outputs/output_count_anomaly_%j.out
#SBATCH --mem-per-cpu=16G

# Print start time
echo "Job started at: $(date)"

source activate casermml
python count_anomalies.py

# Print end time
echo "Job ended at: $(date)"