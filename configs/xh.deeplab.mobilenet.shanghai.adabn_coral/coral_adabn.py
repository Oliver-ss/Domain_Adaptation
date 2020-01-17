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
import numpy as np
import scipy.linalg
from utils.metrics import Evaluator


i = 0
covs = torch.load('/home/home1/xw176/work/Domain_Adaptation/configs/xh.gather_statistics/sta/' + 'khartoum' + '_all_layer_cov.pth')

def coral(module, input, output):
    # do coral after conv i
    global i, covs
    print(i)
    b, c, h, w = output.shape
    Ds = covs[i] #layer i source domain cov
    channels = output.permute(1, 0, 2, 3)
    assert c == Ds.shape[0], "dimention is wrong!"
    features = channels.reshape(c, -1)
    assert len(output.shape) == 4
    Dt = np.cov(features.numpy())
    assert Dt.shape == Ds.shape
    Cs = Ds + np.eye(Ds.shape[0])
    Ct = Dt + np.eye(Dt.shape[0])
    transform = np.linalg.inv(Ct)
    transform = scipy.linalg.sqrtm(transform)
    transform = transform @ scipy.linalg.sqrtm(Cs)
    print("before:{}".format(np.mean(np.cov(features.numpy()) - Ds)))
    features = features.transpose(0,1)
    if transform.dtype == np.complex128:
        transform = transform.real
    output = torch.mm(features, torch.tensor(transform, dtype=torch.float32))
    output = output.transpose(0,1)
    print("after:{}".format(np.mean(np.cov(output.numpy()) - Ds)))
    output = output.reshape(c, b, h, w)
    output = output.permute(1, 0, 2, 3)
    i+=1
    return output


def train(t_domain, model_path, save_path):
    # adabn with coral on layer 1
    train_set = spacenet.Spacenet(city=t_domain, split='train', img_root=config.img_root)
    loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=False,
                              num_workers=config.train_num_workers, drop_last=True)

    model = DeepLab(num_classes=2,
                    backbone=config.backbone,
                    output_stride=config.out_stride,
                    sync_bn=config.sync_bn,
                    freeze_bn=True)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    coral_layer = 0

    # coral on layer 1
    for h in model.modules():
        if isinstance(h, nn.Conv2d):
            h.register_forward_hook(coral)
            coral_layer+=1
            if coral_layer == 1:
                break

    #adabn
    model.train()
    for num, sample in enumerate(loader):
        print(num*16)
        global i
        i = 0
        image, target = sample['image'], sample['label']
        # image, target = image.cuda(), target.cuda()
        with torch.no_grad():
            output = model(image)
    torch.save(model.state_dict(), save_path)
    return model

def test(model, t_domain):
    evaluater = Evaluator(2)
    evaluater.reset()
    test_set = spacenet.Spacenet(city=t_domain, split='test', img_root=config.img_root)
    loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False,
                        num_workers=config.train_num_workers, drop_last=True)
    model.eval()
    tbar = tqdm(loader, desc='\r')
    for num, sample in enumerate(tbar):
        global i
        i = 0
        image, target = sample['image'], sample['label']
        with torch.no_grad():
            output = model(image)
        pred = output.numpy()
        target = target.numpy()
        pred = np.argmax(pred, axis=1)
        # Add batch sample into evaluator
        evaluater.add_batch(target, pred)
    Acc = evaluater.Building_Acc()
    IoU = evaluater.Building_IoU()
    mIoU = evaluater.Mean_Intersection_over_Union()
    print("IoU:{}".format(IoU))
    return Acc, IoU, mIoU


if __name__ == '__main__':
    s_domain = "Khartoum"
    t_domain = 'Paris'
    model_path = '/home/home1/xw176/work/Domain_Adaptation/pretrained_model/' + s_domain.lower() + '.pth'
    save_path = '/home/home1/xw176/work/Domain_Adaptation/configs/xh.deeplab.mobilenet.shanghai.adabn_coral/coral_adabn_model/s_' + s_domain + '_t_' + t_domain + '.pth'
    model = train(t_domain, model_path, save_path)
    test(model, t_domain)