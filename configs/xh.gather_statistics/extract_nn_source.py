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
import torch.nn as nn

i = 0
activation = []

def save_output(module, input, output):
    global i, activation
    # save output
    channels = output.permute(1, 0, 2, 3)
    c = channels.shape[0]
    features = channels.reshape(c, -1)
    if len(activation) == i:
        activation.append(features)
    else:
        activation[i] = torch.cat([activation[i], features], dim=1)
    i += 1
    return


def extract_nn_source(domain, model_path, save_path):
    train_set = spacenet.Spacenet(city=domain, split='test', img_root=config.img_root)
    loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.train_num_workers, drop_last=True)

    model = DeepLab(num_classes=2,
                    backbone=config.backbone,
                    output_stride=config.out_stride,
                    sync_bn=config.sync_bn,
                    freeze_bn=True)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    for h in model.modules():
        if isinstance(h, nn.Conv2d):
            h.register_forward_hook(save_output)

    model.eval()
    for num, sample in enumerate(loader):
        print(num*16)
        global i
        i = 0
        image, target = sample['image'], sample['label']
        # image, target = image.cuda(), target.cuda()
        with torch.no_grad():
            output = model(image)

    torch.save(activation, save_path)

if __name__ == '__main__':
    domain = "Khartoum"
    model_path = '/home/home1/xw176/work/Domain_Adaptation/pretrained_model/' + domain.lower() + '.pth'
    save_path = '/home/home1/xw176/work/Domain_Adaptation/configs/xh.gather_statistics/sta/nn_' + domain.lower() + '.pth'
    extract_nn_source(domain, model_path, save_path)