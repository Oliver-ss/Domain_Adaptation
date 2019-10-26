#!/bin/sh
#SBATCH -o test_427_out
#SBATCH -e test_427_err
#SBATCH --gres=gpu:1

python3 ~/Domain_Adaptation/configs/ry.deeplab.coral.shanghai/test.py "/usr/xtmp/satellite/train_models/xh.deeplab.mobilenet.shanghai/epoch427.pth"