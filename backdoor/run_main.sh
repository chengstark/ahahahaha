#!/bin/bash
#SBATCH -t 240:00:00  # time requested in hour:minute:second
#SBATCH --mem=128G
#SBATCH --gres=gpu:2080rtx:1
#SBATCH --partition=compsci-gpu
#SBATCH --output=/home/users/zg78/adver_ml/backdoor/slurm_outputs/%j.out

source /home/users/zg78/_conda/bin/activate
conda activate base_torch

echo "JOB START"

nvidia-smi

python main.py