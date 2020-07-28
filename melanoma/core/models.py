import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import model as enet
from torchvision import models as _models
from core.layers import AdaptiveConcatPool2d
from utils import model_utils
import config as melanoma_config


class Model(nn.Module):
    """melanoma model."""
    def __init__(self,
                 backbone,
                 num_classes=1,
                 output_net_params=None,
                 pool_params=None,
                 pretrained=True):
        super().__init__()
        self._backbone = backbone
        self._num_classes = num_classes
        self._pool_params = pool_params
        self._output_net_params = output_net_params
        if self._pool_params is None:
            self._pool_method = 'concat'
            self._pool_params = {'params': {}}
        else:
            self._pool_method = self._pool_params['method']
            self._pool_params['params'] = self._pool_params.get('params', {})
        self._pretrained = pretrained

        self.encoder, in_features = model_utils.get_backbone(self._backbone,
                                                             self._pretrained)
        self.pool_layer = melanoma_config.POOLING_MAP[self._pool_method](**self._pool_params['params'])

        if not backbone.startswith('efficientnet'):
            in_features = in_features * 2 if self._pool_method == 'concat' else in_features

        if self._output_net_params is not None:
            hidden_dim = self._output_net_params.get('hidden_dim', 512)
            modules = [
                nn.Linear(in_features, hidden_dim, bias=False),
                nn.ReLU()
            ]

            if self._output_net_params.get('bn'):
                modules.append(nn.BatchNorm1d(hidden_dim))
            if self._output_net_params.get('dropout'):
                modules.append(nn.Dropout(self._output_net_params['dropout']))
            modules.append(nn.Linear(hidden_dim, self._num_classes, bias=False))

            self.output_net = nn.Sequential(*modules)
        else:
            self.output_net = nn.Linear(in_features,
                                        self._num_classes,
                                        bias=False)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool_layer(x)
        x = self.output_net(x)
        return x
