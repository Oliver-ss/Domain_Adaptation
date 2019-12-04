#!/bin/sh
#SBATCH -o shanghai_test_out
#SBATCH -e shanghai_test_err
#SBATCH --gres=gpu:3

python3 ../../scripts/base_test.py 'train_log/models/epoch320.pth' True 0
