#!/bin/sh
#SBATCH -o paris_bn_3583_out
#SBATCH -e paris_bn_3583_err
#SBATCH --gres=gpu:1

python3 ~/Domain_Adaptation/configs/ry.deeplab.coral.paris/coral_test.py "/usr/xtmp/satellite/train_models/xh.deeplab.mobilenet.paris/epoch3583.pth" True 0
