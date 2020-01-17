from tqdm import tqdm
from common import config
import data.spacenet as spacenet
from torch.utils.data import DataLoader
import torch
import os
import json

def compute_mean_variance(domain, filepath):
    train_set = spacenet.Spacenet(city=domain, split='train', img_root=config.img_root)
    loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.train_num_workers, drop_last=True)
    model = DeepLab(num_classes=2,
                    backbone=config.backbone,
                    output_stride=config.out_stride,
                    sync_bn=config.sync_bn,
                    freeze_bn=True)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))


    R = []
    G = []
    B = []
    for sample in tqdm(loader):
        data = sample['image']
        data = data.permute(1, 0, 2, 3)
        data = data.reshape(data.size(0), -1)
        R.append(data[0])
        G.append(data[1])
        B.append(data[2])
    R = torch.cat(R, dim=0)
    G = torch.cat(G, dim=0)
    B = torch.cat(B, dim=0)
    statistics = {
        'R': R,
        'G': G,
        'B': B
    }

    torch.save(statistics, filepath)

if __name__ == '__main__':
    domain = "Vegas"
    filepath = '/home/home1/xw176/work/Domain_Adaptation/configs/xh.gather_statistics/statistics/' + domain + '.pth'
    compute_mean_variance(domain, filepath)