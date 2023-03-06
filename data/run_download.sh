#!/bin/bash
#SBATCH -t 240:00:00  # time requested in hour:minute:second
#SBATCH --mem=50G
#SBATCH --partition=compsci


python download.py