#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --time=3-00:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu
#SBATCH --output=./jupyter_log.log

source /home/users/ds592/anaconda3/bin/activate
conda activate base_torch
jupyter-notebook --ip=0.0.0.0 --port=8881 --no-browser