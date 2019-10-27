#!/bin/bash
#
#BATCH --job-name=💯%true
#SBATCH -o "output/big_tries.stdout"
#SBATCH -e "output/big_tries.stderr"
#SBATCH --ntasks=1
#SBATCH --time=05:00:00
#SBATCH --mem=120G
#SBATCH --partition=compute

# Add ICL-Slurm binaries to path
PATH=/opt/slurm/bin:$PATH

# JOB STEPS (example: write hostname to output file, and wait 1 minute)
python3 court.py
