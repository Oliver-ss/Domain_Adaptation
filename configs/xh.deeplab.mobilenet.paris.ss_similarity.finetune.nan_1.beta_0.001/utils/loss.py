import torch
import torch.nn as nn


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


class SimilarityLosses(object):
    def __init__(self, cuda=False):
        self.cuda = cuda

    def get_loss(self, feature1, feature2, gt_similarity, gamma=2):
        feature1 = feature1.view(1, -1)
        feature2 = feature2.view(1, -1)

        cos = nn.CosineSimilarity().cuda()
        similarity = cos(feature1, feature2)

        # focal loss
        loss = -(gt_similarity ** gamma) * torch.log(similarity)
        return loss


if __name__ == "__main__":
    # loss = SegmentationLosses(cuda=True)
    # a = torch.rand(1, 3, 7, 7).cuda()
    # b = torch.rand(1, 7, 7).cuda()
    # print(loss.CrossEntropyLoss(a, b).item())
    # print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    # print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())

    loss = SimilarityLosses(cuda=True)
    a = torch.ones(3, 2, 2).cuda()
    b = torch.eye(3, 2, 2).cuda()
    gt_similarity = 0.9
    print(loss.get_loss(a, b, gt_similarity))
    print(loss.get_loss(a, b, gt_similarity, gamma=2))
