#!/bin/sh
#SBATCH -o shanghai_train_out
#SBATCH -e shanghai_train_err
#SBATCH --gres=gpu:3
#SBATCH -w, --nodelist=linux42

python3 train.py
