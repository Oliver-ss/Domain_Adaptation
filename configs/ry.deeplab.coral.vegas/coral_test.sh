#!/bin/sh
#SBATCH -o coral_neck_427_out
#SBATCH -e coral_neck_427_err
#SBATCH --gres=gpu:1

python3 ~/Domain_Adaptation/configs/ry.deeplab.coral.vegas/coral_test.py "/usr/xtmp/satellite/train_models/xh.deeplab.mobilenet.vegas/epoch463.pth" True 0
