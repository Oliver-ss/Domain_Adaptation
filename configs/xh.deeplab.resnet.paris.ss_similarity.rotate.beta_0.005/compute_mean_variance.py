from tqdm import tqdm
from common import config
from data import make_data_loader
import data.spacenet as spacenet
from torch.utils.data import DataLoader

def compute_mean_variance():
    train_set = spacenet.Spacenet(city=config.dataset, split='train', img_root=config.img_root)
    loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.train_num_workers, drop_last=True)
    mean = 0.
    std = 0.
    nb_samples = 0.
    for sample in tqdm(loader):
        data = sample['image']
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    print('mean: ', mean)
    print('std:', std)


if __name__ == '__main__':
    compute_mean_variance()