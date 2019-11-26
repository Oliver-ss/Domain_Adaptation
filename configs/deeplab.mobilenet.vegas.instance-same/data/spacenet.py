'''*************************************************************************
	> File Name: spacenet.py
	> Author: yuansong
	> Mail: yuansongwx@outlook.com
	> Created Time: Mon 21 Oct 2019 04:01:05 PM EDT
 ************************************************************************'''
import os
import numpy as np
import scipy.misc as m
import cv2
from torch.utils import data
from torchvision import transforms
import data.custom_transforms as tr
#import custom_transforms as tr
import json
from PIL import Image

class Spacenet(data.Dataset):
    NUM_CLASSES = 2

    def __init__(self, city='Shanghai', split='train', img_root='/usr/xtmp/satellite/spacenet/', if_pair=False):
        self.img_root = img_root
        self.name_root = '../../dataset/spacenet/domains/' + city
        with open(os.path.join(self.name_root, split + '.json')) as f:
            self.files = json.load(f)

        self.split = split
        self.classes = [0, 1]
        self.class_names = ['bkg', 'building']
        self.if_pair = if_pair

        if not self.files:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files), split))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        #img = cv2.imread(os.path.join(self.img_root, self.files[index] + '_RGB.tif'))
        img = Image.open(os.path.join(self.img_root, self.files[index] + '_RGB.tif')).convert('RGB')
        #target = cv2.imread(os.path.join(self.img_root, self.files[index] + '_GT.tif'))
        target = Image.open(os.path.join(self.img_root, self.files[index] + '_GT.tif'))
        sample = {'image': img, 'label': target}
        if self.split == 'train':
            if self.if_pair:
                return self.transform_pair_train(sample)
            else:
                return self.transform_tr(sample)
        elif self.split == 'val':
            if self.if_pair:
                return self.transform_pair_val(sample)
            else:
                return self.transform_val(sample)
        elif self.split == 'test':
            return self.transform_ts(sample)

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=400, crop_size=400, fill=0),
            tr.RandomGaussianBlur(),
            #tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.Normalize(),
            tr.ToTensor(),
        ])
        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(400),
            #tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.Normalize(),
            tr.ToTensor(),
        ])
        return composed_transforms(sample)

    def transform_pair_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(400),
            #tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.HorizontalFlip(),
            tr.GaussianBlur(),
            tr.Normalize(if_pair=True),
            tr.ToTensor(if_pair=True),
        ])
        return composed_transforms(sample)

    def transform_ts(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixedResize(size=400),
            #tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.Normalize(),
            tr.ToTensor(),
        ])
        return composed_transforms(sample)

    def transform_pair_train(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=400, crop_size=400, fill=0),
            #tr.RandomGaussianBlur(),
            tr.HorizontalFlip(),
            tr.GaussianBlur(),
            tr.Normalize(if_pair=True),
            tr.ToTensor(if_pair=True),
        ])
        return composed_transforms(sample)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    spacenet_train = Spacenet(split='val', if_pair=True)
    dataloader = DataLoader(spacenet_train, batch_size=2, shuffle=True, num_workers=2)
    #print(spacenet_train.__getitem__(0))
    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'][jj].numpy()
            gt = sample['label'][jj].numpy()
            pair = sample['image_pair'][jj].numpy()
            #img = img.transpose(1,2,0)
            #gt = gt[:,:,None]
            print(img.shape)
            print(gt.shape)
            print(pair.shape)
            #gt_ = gt.repeat(3, axis=2)
            #show = np.hstack((img, gt_))
            #cv2.imshow('show', show[:,:,::-1])
            #c = chr(cv2.waitKey(0) & 0xff)
            #if c == 'q':
            #    exit()





