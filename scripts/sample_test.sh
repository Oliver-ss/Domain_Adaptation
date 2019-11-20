#!/bin/sh
#SBATCH -o sample_test_out
#SBATCH -e sample_test_err
#SBATCH --gres=gpu:3

python3 ~/Domain_Adaptation/configs/ry.deeplab.coral.shanghai/coral_test.py "/usr/xtmp/satellite/train_models/xh.deeplab.mobilenet.shanghai/epoch427.pth" True 0
