#!/bin/bash
#SBATCH -o log_out
#SBATCH -e log_err
#SBATCH --gres=gpu:1
make test MODEL='/usr/xtmp/satellite/train_models/deeplab.mobilenet.vegas.adversarial.finetune/epoch63.pth' BN=True SAVE=-1
