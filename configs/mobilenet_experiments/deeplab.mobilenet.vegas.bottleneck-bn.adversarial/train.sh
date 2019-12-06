#!/bin/bash
#SBATCH -o log_out
#SBATCH -e log__err
#SBATCH --gres=gpu:3
python3 train.py
