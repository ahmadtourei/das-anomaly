#!/bin/bash
#SBATCH -p h200_normal_q
#SBATCH --gres=gpu:1
#SBATCH --qos=tc_h200_normal_short
#SBATCH -t 1-00:00:00
#SBATCH -A YOUR_ACCOUNT

# Print start time
echo "Job started at: $(date)"

# Load modules and environment
source activate dasanomaly

# Run the script
python <<'PY'
from das_anomaly.train import TrainAEConfig, AutoencoderTrainer

cfg = TrainAEConfig()
AutoencoderTrainer(cfg).run()
PY

# Print end time
echo "Job ended at: $(date)"
