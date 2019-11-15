#!/bin/sh
#SBATCH -o sample_test_out
#SBATCH -e sample_test__err
#SBATCH --gres=gpu:1
python3 train,py
