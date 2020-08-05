import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, size=None):
        super().__init__()
        self._size = size or (1, 1)
        self.ap = nn.AdaptiveAvgPool2d(self._size)
        self.mp = nn.AdaptiveMaxPool2d(self._size)

    def forward(self, x):
        batch_size = len(x)
        return torch.cat([self.mp(x), self.ap(x)], 1).view(batch_size, -1)


class AdaptiveAvgPool2d(nn.Module):
    def __init__(self, size=None):
        super().__init__()
        self._size = size or (1, 1)
        self.ap = nn.AdaptiveAvgPool2d(self._size)

    def forward(self, x):
        batch_size = len(x)
        return self.ap(x).view(batch_size, -1)


class AdaptiveMaxPool2d(nn.Module):
    def __init__(self, size=None):
        super().__init__()
        self._size = size or (1, 1)
        self.mp = nn.AdaptiveMaxPool2d(self._size)

    def forward(self, x):
        batch_size = len(x)
        return self.mp(x).view(batch_size, -1)


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class GeM(nn.Module):
    """https://arxiv.org/pdf/1711.02512.pdf"""
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        batch_size = len(x)
        return gem(x, p=self.p, eps=self.eps).view(batch_size, -1)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class MishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.tanh(F.softplus(x))   # x * tanh(ln(1 + exp(x)))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid = torch.sigmoid(x)
        tanh_sp = torch.tanh(F.softplus(x))
        return grad_output * (tanh_sp + x * sigmoid * (1 - tanh_sp * tanh_sp))


class Mish(nn.Module):
    def forward(self, x):
        return MishFunction.apply(x)



class AdaptiveConcatGeMPool2d(nn.Module):
    def __init__(self, size=None, **kwargs):
        super().__init__()
        self._size = size or (1, 1)
        self.ap = nn.AdaptiveAvgPool2d(self._size)
        self.mp = nn.AdaptiveMaxPool2d(self._size)
        self.gem = GeM(**kwargs)

    def forward(self, x):
        batch_size = len(x)
        return torch.cat([self.mp(x).reshape(batch_size, -1), self.ap(x).reshape(batch_size, -1), self.gem(x)], 1)
