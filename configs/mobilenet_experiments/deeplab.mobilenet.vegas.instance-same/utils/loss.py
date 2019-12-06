import torch
import torch.nn as nn
import numpy as np

class InstanceLoss(object):
    def __init__(self, cuda=True):
        self.cuda = cuda

    def remove(self, tensor, idx):
        '''
        Remove certain batch from the tensor
        Input: tenosr n*c*h*w
        Output: tensor (n-1)*c*h*w
        '''
        if idx == 0:
            return tensor[1:]
        if idx == tensor.size()[0]-1:
            return tensor[:-1]
        else:
            return torch.cat((tensor[:idx], tensor[idx+1:]), dim=0)
    def __call__(self, u, v):
        n, c, h, w = u.size()
        norm_u = torch.sqrt(torch.mul(u, u).sum(dim=1).sum(dim=1).sum(dim=1) + 1e-30)
        norm_v = torch.sqrt(torch.mul(v, v).sum(dim=1).sum(dim=1).sum(dim=1) + 1e-30)
        ans = torch.zeros(n,2*n-1)
        for i in range(n):
            ans[i,:n] = u[i].unsqueeze(dim=0).repeat(n,1,1,1).mul(v).sum(dim=1).sum(dim=1).sum(dim=1).div(norm_v * norm_u[i])
            ans[i,n:] = u[i].unsqueeze(dim=0).repeat(n-1,1,1,1).mul(self.remove(u,i)).sum(dim=1).sum(dim=1).sum(dim=1).div(self.remove(norm_u,i)*norm_u[i])
        s = torch.exp(ans).sum(dim=1)
        ans = torch.exp(ans[:,0]).div(s)
        if self.cuda:
            return -torch.log(ans).sum().cuda()
        else:
            return -torch.log(ans).sum()


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
    #loss = SegmentationLosses(cuda=True)
    #a = torch.rand(1, 3, 7, 7).cuda()
    #b = torch.rand(1, 7, 7).cuda()
    #print(loss.CrossEntropyLoss(a, b).item())
    #print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    #print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())
    ins = InstanceLoss()
    u = torch.ones(8,2,10,10)
    v = torch.ones(8,2,10,10)*2
    print(ins.loss(u,v))



