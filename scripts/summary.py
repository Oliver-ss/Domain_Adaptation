'''*************************************************************************
	> File Name: visualize.py
	> Author: yuansong
	> Mail: yuansongwx@outlook.com
	> Created Time: Mon 28 Oct 2019 11:44:50 AM EDT
 ************************************************************************'''
import sys, os
sys.path.append(os.getcwd())
from model.sync_batchnorm.replicate import patch_replication_callback
from model.deeplab import *
import torch
from torchsummary import summary
from common import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeepLab(num_classes=2,
                backbone=config.backbone,
                output_stride=config.out_stride,
                sync_bn=config.sync_bn,
                freeze_bn=config.freeze_bn).to(device)

summary(model, (3, 400, 400))

