import os
import sys

sys.path.append(os.getcwd())
import torch
import argparse
from model.deeplab import *
from tqdm import tqdm
import json
from utils.metrics import Evaluator
from data import spacenet
from common import config
from torch.utils.data import DataLoader
import numpy as np
import cv2
import torchvision
import matplotlib.pyplot as plt
import torch.nn
from model.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


def extract_bn_mean_var(model, filepath):
    model_path = '/home/home1/xw176/work/Domain_Adaptation/configs/mobilenet.baseline/xh.deeplab.mobilenet.' + model.lower() + '.n/train_log/best_shanghai.pth'
    model = DeepLab(num_classes=2,
                    backbone=config.backbone,
                    output_stride=config.out_stride,
                    sync_bn=config.sync_bn,
                    freeze_bn=True)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    mean = []
    var = []
    for m in model.modules():
        if isinstance(m, SynchronizedBatchNorm2d) or isinstance(m, nn.BatchNorm2d):
            mean.append(m.running_mean)
            var.append(m.running_var)

    data = {
        'mean': mean,
        'var': var
    }

    torch.save(data, filepath)


cuda = True
# bn = False

if __name__ == '__main__':
    model = 'Shanghai'
    filepath = '/home/home1/xw176/work/Domain_Adaptation/configs/xh.gather_statistics/statistics/' + model + '_bn.pth'
    extract_bn_mean_var(model, filepath)
