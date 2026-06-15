#!/bin/bash
#SBATCH -p GPU-small
#SBATCH -N 1
#SBATCH --gres=gpu:p100:1
#SBATCH --mail-type=ALL
#SBATCH -t 07:30:00
module load anaconda3
source activate det
module load cuda/10.1
python nn_step1.py 0.01 0.9
