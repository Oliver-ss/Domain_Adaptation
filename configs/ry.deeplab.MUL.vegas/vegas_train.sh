#!/bin/sh
#SBATCH -o vegas_train_out
#SBATCH -e vegas_train_err
#SBATCH --gres=gpu:3

python3 train.py --start_epoch 79 --resume 'train_log/models/epoch78.pth'
