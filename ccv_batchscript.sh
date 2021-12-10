#!/bin/bash

# Request a GPU partition node and access to 1 GPU
#SBATCH -p gpu --gres=gpu:1

# Request 1 CPU core
#SBATCH -n 1

#SBATCH --mem=24G
#SBATCH -t 16:00:00
#SBATCH -o sum.out

module load python/3.7.4
module load pytorch/1.3.1
module load cuda/11.1.1    
module load cudnn/8.2.0
module load gcc/10.2

python main.py