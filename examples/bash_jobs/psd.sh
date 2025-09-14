#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH -A YOUR_ACCOUNT
#SBATCH --mem-per-cpu=16G

# Print start time
echo "Job started at: $(date)"

# Load modules and environment
source activate dasanomaly

# Run the script
python << EOF
from das_anomaly.psd import PSDConfig, PSDGenerator

cfg = PSDConfig()
# serial processing with single processor:
PSDGenerator(cfg).run()
EOF

# Print end time
echo "Job ended at: $(date)"
