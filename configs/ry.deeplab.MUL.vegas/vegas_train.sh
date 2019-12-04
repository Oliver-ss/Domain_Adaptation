#!/bin/sh
#SBATCH -o vegas_train_out
#SBATCH -e vegas_train_err
#SBATCH --gres=gpu:3
#SBATCH -w, --nodelist=linux48

python3 train.py --start_epoch 436 --resume 'train_log/models/epoch436.pth'
