import torch
from torch.autograd import Variable

class ArgMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        idx = torch.argmax(input, 1, keepdim=True)
        output = torch.zeros_like(input, requires_grad=True)
        output.scatter_(1, idx, 1)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output = ctx.saved_variables[0]
        return grad_output.mul(output)


if __name__ == '__main__':
    a = torch.rand((2, 3, 2, 2), requires_grad=True)
    print(a)
    argmax = ArgMax()
    b = argmax.apply(a)
    c = b.sum()*3
    c.backward()
    print('b: ', b)
    print('a: ' ,a.grad)