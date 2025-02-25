#!/bin/bash
#SBATCH -t 36:00:00
#SBATCH -A casermminefiber
#SBATCH --output=/globalscratch/ahmad9/caserm/spectrum_analysis/outputs/output_event_detection_%j.out
#SBATCH --mem-per-cpu=128G

# Print start time
echo "Job started at: $(date)"

source activate casermml
python detect_anomalies.py

# Print end time
echo "Job ended at: $(date)"
