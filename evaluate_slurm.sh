#!/bin/bash
# NOTE: before running this do `conda activate itn2`.
# run as `sbatch vista_evaluate.sh`

# Job Details
#SBATCH --partition=gpu
#SBATCH -J vista
#SBATCH -o ./runs/%j-slurm-run.txt # STDOUT/STDERR

# Resources
#SBATCH -t 23:59:00
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:tesla:1
#SBATCH --exclude=falcon2

# Actual job command(s)
module load ffmpeg
srun python -u vista_sim/vista_evaluate.py "$@"
