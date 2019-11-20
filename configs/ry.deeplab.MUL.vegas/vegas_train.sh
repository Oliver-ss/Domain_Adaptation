#!/bin/sh
#SBATCH -o vegas_train_out
#SBATCH -e vegas_train_err
#SBATCH --gres=gpu:3

python3 train.py
