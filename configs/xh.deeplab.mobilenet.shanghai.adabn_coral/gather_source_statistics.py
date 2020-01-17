from tqdm import tqdm
from common import config
import data.spacenet as spacenet
from torch.utils.data import DataLoader
import torch
import os
import json

def extract_nn_source(domain, filepath):
    train_set = spacenet.Spacenet(city=domain, split='train', img_root=config.img_root)
    loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.train_num_workers, drop_last=True)
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
    filepath = '/home/home1/xw176/work/Domain_Adaptation/pretrained_model/' + domain + '.pth'
    extract_nn_source(domain, filepath)