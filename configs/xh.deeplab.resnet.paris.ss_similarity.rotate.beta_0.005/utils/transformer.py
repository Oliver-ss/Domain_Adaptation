from torch.nn import functional as F
from torchvision import transforms
import numpy as np
import torch
import math


class Transformer():
    def __init__(self, cuda):
        # self.transpose = np.random.randint(2)
        # self.rotate = np.random.randint(4)
        self.transpose = 0
        self.rotate = 3
        self.cuda = cuda

    def __call__(self, input):
        is_target = False

        # unsqueeze target
        if len(input.shape) == 3:
            is_target = True
            input = input.unsqueeze(1)

        n, _, _, _ = input.shape

        if self.transpose == 1:
            theta = torch.tensor([
                [0, 1, 0],
                [1, 0, 0]
            ], dtype=torch.float)
            grid = F.affine_grid(theta.expand(n, 2, 3), size=input.size())
            if self.cuda:
                input = input.cuda()
                grid = grid.cuda()
            input = F.grid_sample(input, grid)

        if self.rotate != 0:
            angle = self.rotate * 90 * math.pi / 180
            theta = torch.tensor([
                [math.cos(angle), math.sin(-angle), 0],
                [math.sin(angle), math.cos(angle), 0]
            ], dtype=torch.float)
            grid = F.affine_grid(theta.expand(n, 2, 3), size=input.size())
            if self.cuda:
                input = input.cuda()
                grid = grid.cuda()
            input = F.grid_sample(input, grid)

        if is_target:
            input = input.squeeze(1)
        return input


if __name__ == '__main__':
    t = Tranformer()
    print(t)
    input1 = torch.rand((2, 2, 2))
    output1 = t(input1)
    print(input1)
    print(output1)
    input2 = torch.rand((2, 2, 2, 2))
    output2 = t(input2)
    print(input2)
    print(output2)
