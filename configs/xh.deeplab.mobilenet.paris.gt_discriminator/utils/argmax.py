import torch
from torch.autograd import Variable

class ArgMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        idx = torch.argmax(input, 1, keepdim=True)
        output = torch.zeros_like(input, requires_grad=True)
        output.scatter_(1, idx, 1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


if __name__ == '__main__':
    a = torch.rand((1, 2, 2, 2), requires_grad=True)
    argmax = ArgMax()
    b = argmax.apply(a)
    c= b.sum()*3
    c.backward()
    print('b: ', b)
    print('c: ',c )
    print('b: ',b.grad)
    print('a: ' ,a.grad)