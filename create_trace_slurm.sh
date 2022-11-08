#!/bin/bash
# NOTE: before running this do `conda activate itn2`.
# run as `sbatch rosbag_to_vista.sh`

# Job Details
#SBATCH --partition=amd
#SBATCH -J train
#SBATCH -o ./runs/%j-slurm-run.txt # STDOUT/STDERR

# Resources
#SBATCH -t 23:59:00
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=24
#SBATCH --exclude=falcon2,falcon3

# Actual job command(s)
srun python -u create_trace.py "$@"

