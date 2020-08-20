import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class AdaptiveAvgPool2d(nn.Module):
    """2D adaptive average pooling wrapper."""
    def __init__(self, size=None):
        super().__init__()
        self._size = size or (1, 1)
        self.ap = nn.AdaptiveAvgPool2d(self._size)

    def forward(self, x):
        batch_size = len(x)
        return self.ap(x).view(batch_size, -1)


class AdaptiveMaxPool2d(nn.Module):
    """2D adaptive max pooling wrapper."""
    def __init__(self, size=None):
        super().__init__()
        self._size = size or (1, 1)
        self.mp = nn.AdaptiveMaxPool2d(self._size)

    def forward(self, x):
        batch_size = len(x)
        return self.mp(x).view(batch_size, -1)


class AdaptiveConcatPool2d(nn.Module):
    """Applies a concatentation of 2D adaptive average and max pooling over an
    input signal composed of several input planes.
    """
    def __init__(self, size=None):
        super().__init__()
        self._size = size or (1, 1)
        self.ap = AdaptiveAvgPool2d(self._size)
        self.mp = nnAdaptiveMaxPool2d(self._size)

    def forward(self, x):
        batch_size = len(x)
        return torch.cat([self.mp(x), self.ap(x)], 1).view(batch_size, -1)


def compute_gem(x, p=3, eps=1e-5):
    """Computes the generalized mean over an input signal."""
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class GeMPool2d(nn.Module):
    """Generalized Mean (GeM) pooling as proposed in "Fine-tuning CNN Image Retrieval
    with No Human Annotation" (https://arxiv.org/pdf/1711.02512.pdf).

    Applies a 2D generalized mean pooling over an input signal composed of several input planes.

    Parameters
    ----------
    p : float (default=3)
        Exponent where p=0 is equivalent to average pooling, p=np.inf is equivalent
        to max pooling and all real values (0, np.inf) applies signifies the
        generalized mean.
    eps: float (default=1e-5)
        A value added to the denominator for numerical stability.
    """
	def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.tensor([p]), dtype=torch.float32)
        self._eps = eps

    def forward(self, x):
        batch_size = len(x)
        return compute_gem(x, p=self.p, eps=self._eps).view(batch_size, -1)

    def __repr__(self):
        format_str = f'{self.__class__.__name__}('
        format_str += f'p={self.p.data.tolist()[0]:.4f}, '
        format_str += f'eps={self._eps})'
        return format_str


class AdaptiveConcatGeMPool2d(nn.Module):
    def __init__(self, size=None, **kwargs):
        super().__init__()
        self._size = size or (1, 1)
        self.ap = nn.AdaptiveAvgPool2d(self._size)
        self.mp = nn.AdaptiveMaxPool2d(self._size)
        self.gem = GeMPool2d(**kwargs)

    def forward(self, x):
        batch_size = len(x)
        return torch.cat([self.mp(x).reshape(batch_size, -1), self.ap(x).reshape(batch_size, -1), self.gem(x)], 1)


class AttentionPool1d(nn.Module):
    """MIL pooling as propsed in "Attention-based Deep Multiple Instance Learning"
    (http://proceedings.mlr.press/v80/ilse18a/ilse18a.pdf).

    Parameters
    ----------
    in_channels : int
        Number of channels in the input.
    num_hidden : int
        Number of features in the attention mechanism. Also, referred to as the
        embedding size, K, described in Section 2.4.
    dropout : float (default=0)
        Probability of an element to be zeroed in the attention mechanism.
    """
    def __init__(self, in_channels, num_hidden, dropout=0):
        super().__init__()
        self._in_channels = in_channels
        self._num_hidden = num_hidden
        self._dropout = dropout

        modules = [
            nn.Linear(self._in_channels, self._num_hidden, bias=False),
            nn.Tanh()
        ]
        if self._dropout > 0:
            modules.append(nn.Dropout(self._dropout))
        modules.append(nn.Linear(self._num_hidden, 1, bias=False))

        self.attention = nn.Sequential(*modules)

    def forward(self,x):
        num_patches = x.size(1)
        x = x.view(-1, x.size(2))
        A = self.attention(x)
        A = A.view(-1, num_patches, 1)
        weights = F.softmax(A, dim=1)
        return (x.view(-1, num_patches, self._in_channels) * weights).sum(dim=1), A
