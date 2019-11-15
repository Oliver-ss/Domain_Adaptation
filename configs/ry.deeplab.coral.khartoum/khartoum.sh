#!/bin/sh
#SBATCH -o khartoum_bn_3006_out
#SBATCH -e khartoum_bn_3006_err
#SBATCH --gres=gpu:1

python3 ~/Domain_Adaptation/configs/ry.deeplab.coral.khartoum/coral_test.py "/usr/xtmp/satellite/train_models/xh.deeplab.mobilenet.khartoum/epoch3006.pth" True 0
