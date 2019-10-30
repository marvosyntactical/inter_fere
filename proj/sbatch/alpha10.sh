#!/bin/bash
#
#BATCH --job-name=100
#SBATCH -o "output/alpha10.stdout"
#SBATCH -e "output/alpha10.stderr"
#SBATCH --ntasks=1
#SBATCH --time=2-23:00:00
#SBATCH --mem=256G
#SBATCH --partition=compute

# Add ICL-Slurm binaries to path
PATH=/opt/slurm/bin:$PATH

# JOB STEPS (example: write hostname to output file, and wait 1 minute)
python3 court.py
