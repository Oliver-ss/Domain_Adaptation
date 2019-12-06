#!/bin/bash
#SBATCH -o log_out
#SBATCH -e log_err
#SBATCH --gres=gpu:3
python3 train.py --resume='/usr/xtmp/satellite/train_models/xh.deeplab.mobilenet.vegas/epoch463.pth' --visdom=False
