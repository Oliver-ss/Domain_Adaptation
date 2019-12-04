#!/bin/sh
#SBATCH -o vegas_test_out
#SBATCH -e vegas_test_err
#SBATCH --gres=gpu:3

python3 ../../scripts/base_test.py 'train_log/models/epoch599.pth' True 0
