import os
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

class Test:
    def __init__(self, model_path, source, target, cuda=False):
        self.source_set = spacenet.Spacenet(city=source, split='test', img_root=config.img_root)
        self.target_set = spacenet.Spacenet(city=target, split='test', img_root=config.img_root)
        self.source_loader = DataLoader(self.source_set, batch_size=16, shuffle=False, num_workers=2)
        self.target_loader = DataLoader(self.target_set, batch_size=16, shuffle=False, num_workers=2)

        self.model = DeepLab(num_classes=2,
                backbone=config.backbone,
                output_stride=config.out_stride,
                sync_bn=config.sync_bn,
                freeze_bn=config.freeze_bn)
        if cuda:
            self.checkpoint = torch.load(model_path)
        else:
            self.checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        #print(self.checkpoint.keys())
        self.model.load_state_dict(self.checkpoint)
        self.evaluator = Evaluator(2)
        self.cuda = cuda
        if cuda:
            self.model = self.model.cuda()

    def get_performance(self, dataloader):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(dataloader, desc='\r')
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Building_Acc()
        IoU = self.evaluator.Building_IoU()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        return Acc, IoU, mIoU

    def test(self):
        sA, sI, sIm = self.get_performance(self.source_loader)
        tA, tI, tIm = self.get_performance(self.target_loader)
        print('Test for source domain:')
        print("Acc:{}, IoU:{}, mIoU:{}".format(sA, sI, sIm))
        print('Test for target domain:')
        print("Acc:{}, IoU:{}, mIoU:{}".format(tA, tI, tIm))
        res = {'source':{'Acc': sA, 'IoU': sI, 'mIoU': sIm},
                'target':{'Acc': tA, 'IoU': tI, 'mIoU': tIm}}
        with open('train_log/test.json', 'w') as f:
            json.dump(res, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', default='train_log/models/epoch320.pth')
    parser.add_argument('--source', default='Vegas')
    parser.add_argument('--target', default= ['Shanghai', 'Paris', 'Khartoum'])
    parser.add_argument('--cuda', default=True)

    args = parser.parse_args()

    test = Test(args.model, args.source, args.target, args.cuda)
    test.test()


