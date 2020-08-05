import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import model as enet
from torchvision import models as _models
from core.layers import AdaptiveConcatPool2d
from utils import model_utils
import config as melanoma_config


class BaseModel(nn.Module):
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
        # in_features = in_features * 2 if self._pool_method == 'concat' else in_features

        if self._output_net_params is not None:
            #hidden_dim = self._output_net_params.get('hidden_dim', 512)
            #modules = [
            #    nn.Linear(in_features, hidden_dim, bias=False),
            #    nn.ReLU()
            #]

            modules = []
            if self._output_net_params.get('bn'):
                modules.append(nn.BatchNorm2d(in_features,
                                              **self._output_net_params['bn']))
            modules.append(self.pool_layer)
            if self._pool_method == 'concat':
                in_features *= 2
            elif self._pool_method == 'concat_gem':
                in_features *= 3
            if self._output_net_params.get('dropout'):
                modules.append(nn.Dropout(self._output_net_params['dropout']))

            modules.append(nn.Linear(in_features, self._num_classes, bias=False))
            self.output_net = nn.Sequential(*modules)
        else:
            if self._pool_method == 'concat':
                in_features *= 2
            elif self._pool_method == 'concat_gem':
                in_features *= 3
            self.output_net = nn.Sequential(
                self.pool_layer,
                nn.Linear(in_features, self._num_classes, bias=False)
            )

    def forward(self, x):
        x = self.encoder(x)
        #x = self.pool_layer(x)
        x = self.output_net(x)
        return x


class EfficientModel(nn.Module):
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
        # in_features = in_features * 2 if self._pool_method == 'concat' else in_features

        if self._output_net_params is not None:
            #hidden_dim = self._output_net_params.get('hidden_dim', 512)
            #modules = [
            #    nn.Linear(in_features, hidden_dim, bias=False),
            #    nn.ReLU()
            #]

            modules = []
            if self._output_net_params.get('bn'):
                modules.append(nn.BatchNorm2d(in_features, **self._output_net_params['bn']))
            modules.append(self.pool_layer)
            in_features = in_features * 2 if self._pool_method == 'concat' else in_features
            if self._output_net_params.get('dropout'):
                modules.append(nn.Dropout(self._output_net_params['dropout']))
            modules.append(nn.Linear(in_features, self._num_classes, bias=False))
            self.output_net = nn.Sequential(*modules)
        else:
            if self._pool_method == 'concat':
                in_features *= 2
            elif self._pool_method == 'concat_gem':
                in_features *= 3
            self.output_net = nn.Sequential(
                self.pool_layer,
                nn.Linear(in_features, self._num_classes, bias=False)
            )

    def extract_features(self, x):
        return self.encoder.extract_features(x)

    def forward(self, x):
        x = self.extract_features(x)
        x = self.output_net(x)
        return x
