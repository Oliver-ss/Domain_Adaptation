import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class BottleneckClassLoss(object):
    def loss(self, source, target, mask_s, mask_t):
        if mask_s.dim() == 3:
            mask_s = torch.unsqueeze(mask_s, 1)
        if mask_t.dim() == 3:
            mask_t = torch.unsqueeze(mask_t, 1)
        n, c, h, w = source.size()
        mse = nn.MSELoss()
        
        mask_s = F.interpolate(mask_s, size=(h, w))
        mask_t = F.interpolate(mask_t, size=(h, w))
        # mask for building
        mask_s1 = mask_s.expand(source.size())
        mask_t1 = mask_t.expand(source.size())
        # mask for background
        mask_s0 = 1 - mask_s1
        mask_t0 = 1 - mask_t1

        source_1, source_0, target_1, target_0 = source * mask_s1, source * mask_s0, target* mask_t1, target * mask_t0
        s1_mean = source_1.sum(dim=0).sum(dim=1).sum(dim=1) / mask_s1.sum(dim=0).sum(dim=1).sum(dim=1)
        s0_mean = source_0.sum(dim=0).sum(dim=1).sum(dim=1) / mask_s0.sum(dim=0).sum(dim=1).sum(dim=1)
        t1_mean = target_1.sum(dim=0).sum(dim=1).sum(dim=1) / mask_t1.sum(dim=0).sum(dim=1).sum(dim=1)
        t0_mean = target_0.sum(dim=0).sum(dim=1).sum(dim=1) / mask_t0.sum(dim=0).sum(dim=1).sum(dim=1)
        #loss = mse(s1_mean, t1_mean) + mse(s0_mean, t0_mean)
        s1_var = ((F.interpolate(s1_mean.reshape(1, -1, 1, 1), size=(h,w)) - source_1)**2 * mask_s1).sum(dim=0).sum(dim=1).sum(dim=1) / mask_s1.sum(dim=0).sum(dim=1).sum(dim=1)
        s0_var = ((F.interpolate(s0_mean.reshape(1, -1, 1, 1), size=(h,w)) - source_0)**2 * mask_s0).sum(dim=0).sum(dim=1).sum(dim=1) / mask_s0.sum(dim=0).sum(dim=1).sum(dim=1)
        t1_var = ((F.interpolate(t1_mean.reshape(1, -1, 1, 1), size=(h,w)) - target_1)**2 * mask_t1).sum(dim=0).sum(dim=1).sum(dim=1) / mask_t1.sum(dim=0).sum(dim=1).sum(dim=1)
        t0_var = ((F.interpolate(t0_mean.reshape(1, -1, 1, 1), size=(h,w)) - target_0)**2 * mask_t0).sum(dim=0).sum(dim=1).sum(dim=1) / mask_t0.sum(dim=0).sum(dim=1).sum(dim=1)
        loss = mse(s1_var, t1_var) + mse(s0_var, t0_var) + mse(s1_mean, t1_mean) + mse(s0_mean, t0_mean)
        return loss

class BottleneckLoss(object):
    def loss(self, source, target):
        n, c, h, w = source.size()
        s_mean = source.mean(dim=0).mean(dim=1).mean(dim=1)
        t_mean = target.mean(dim=0).mean(dim=1).mean(dim=1)
        s_var = source.var(dim=0).var(dim=1).var(dim=1)
        t_var = target.var(dim=0).var(dim=1).var(dim=1)
        mse = nn.MSELoss()
        loss = mse(s_mean, t_mean) + mse(s_var, t_var)
        return loss

class MinimizeEntropyLoss(object):
    def loss(self, v):
        """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
        """
        assert v.dim() == 4
        n, c, h, w = v.size()
        softmax = nn.Softmax()
        v = softmax(v)
        return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * np.log2(c))

class BatchLoss(object):
    def loss(self, source, target):
        loss = nn.MSELoss()
        return loss(source, target)

class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss


if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




