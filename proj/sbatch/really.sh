#!/bin/bash
#
#BATCH --job-name=100
#SBATCH -o "output/rly.stdout"
#SBATCH -e "output/rly.stderr"
#SBATCH --ntasks=1
#SBATCH --time=05:00:00
#SBATCH --mem=256G
#SBATCH --partition=compute

# Add ICL-Slurm binaries to path
PATH=/opt/slurm/bin:$PATH

# JOB STEPS (example: write hostname to output file, and wait 1 minute)
python3 -c "print('Hellworld')"
