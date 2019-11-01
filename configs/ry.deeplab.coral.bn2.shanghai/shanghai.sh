#!/bin/sh
#SBATCH -o bn_432_out
#SBATCH -e bn_432_err
#SBATCH --gres=gpu:1

python3 ~/Domain_Adaptation/configs/ry.deeplab.coral.shanghai/coral_test.py "/usr/xtmp/satellite/train_models/xh.deeplab.mobilenet.shanghai/epoch432.pth" True 0
