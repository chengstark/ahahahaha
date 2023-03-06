#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --time=3-00:00:00
#SBATCH --mem=128GB
# SBATCH --gres=gpu:2080rtx:0
#SBATCH --output=./jupyter_log.log

source /home/users/zg78/_conda/bin/activate
conda activate base_torch
jupyter-notebook --ip=0.0.0.0 --port=8881 --no-browser