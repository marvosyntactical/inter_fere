#!/bin/bash
#
#BATCH --job-name=💯%😍
#SBATCH -o "output/smokedouble.stdout"
#SBATCH -e "output/smokedouble.stderr"
#SBATCH --ntasks=1
#SBATCH --time=05:00:00
#SBATCH --mem=120G
#SBATCH --partition=compute

# Add ICL-Slurm binaries to path
PATH=/opt/slurm/bin:$PATH

# JOB STEPS (example: write hostname to output file, and wait 1 minute)
python3 court.py
