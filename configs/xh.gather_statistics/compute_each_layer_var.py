import torch
from tqdm import tqdm
import numpy as np

def compute_variance(layers_data):
    all_cov=[]
    for i in tqdm(range(len(layers_data))):
        # print(i)
        data = layers_data[i].numpy()
        cov = np.cov(data)
        # print(i, cov.shape)
        all_cov.append(cov)
    torch.save(all_cov, save_path)



if __name__ == '__main__':
    domain = 'Khartoum'
    data = torch.load('/home/home1/xw176/work/Domain_Adaptation/configs/xh.gather_statistics/sta/nn_' + domain.lower() + '.pth')
    save_path = '/home/home1/xw176/work/Domain_Adaptation/configs/xh.gather_statistics/sta/' + domain.lower() + '_all_layer_cov.pth'
    compute_variance(data)
