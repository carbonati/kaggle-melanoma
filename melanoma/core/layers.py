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


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class GeM(nn.Module):
    """https://arxiv.org/pdf/1711.02512.pdf"""
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'
