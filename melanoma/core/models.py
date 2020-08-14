import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import model as enet
from torchvision import models as _models
from core import layers
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

        if self._pool_method == 'concat':
            in_features *= 2
        elif self._pool_method == 'concat_gem':
            in_features *= 3

        if self._output_net_params is not None:
            hidden_dim = self._output_net_params.get('hidden_dim', [])
            if not isinstance(hidden_dim, list):
                hidden_dim = [hidden_dim]
            hidden_dim = [in_features] + hidden_dim + [self._num_classes]

            dropout = self._output_net_params.get('dropout')
            bn = self._output_net_params.get('bn')

            modules = []
            for i in range(len(hidden_dim)-1):
                if dropout is not None:
                    if isinstance(dropout, list):
                        modules.append(nn.Dropout(dropout[i]))
                    else:
                        modules.append(nn.Dropout(dropout))
                if bn is not None:
                    if isinstance(bn, list):
                        modules.append(nn.BatchNorm1d(hidden_dim[i], **bn[i]))
                    else:
                        modules.append(nn.BatchNorm1d(hidden_dim[i], **bn))

                modules.append(nn.Linear(hidden_dim[i], hidden_dim[i+1], bias=False))
                if i < (len(hidden_dim)-2):
                    modules.append(nn.ReLU(inplace=True))
            self.output_net = nn.Sequential(*modules)
        else:
            self.output_net = nn.Sequential(
                nn.Linear(in_features, self._num_classes, bias=False)
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool_layer(x)
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

        if self._pool_method == 'concat':
            in_features *= 2
        elif self._pool_method == 'concat_gem':
            in_features *= 3

        if self._output_net_params is not None:
            hidden_dim = self._output_net_params.get('hidden_dim', [])
            if not isinstance(hidden_dim, list):
                hidden_dim = [hidden_dim]
            hidden_dim = [in_features] + hidden_dim + [self._num_classes]

            dropout = self._output_net_params.get('dropout')
            bn = self._output_net_params.get('bn')

            modules = []
            for i in range(len(hidden_dim)-1):
                if dropout is not None:
                    if isinstance(dropout, list):
                        modules.append(nn.Dropout(dropout[i]))
                    else:
                        modules.append(nn.Dropout(dropout))
                if bn is not None:
                    if isinstance(bn, list):
                        modules.append(nn.BatchNorm1d(hidden_dim[i], **bn[i]))
                    else:
                        modules.append(nn.BatchNorm1d(hidden_dim[i], **bn))

                modules.append(nn.Linear(hidden_dim[i], hidden_dim[i+1], bias=False))
                if i < (len(hidden_dim)-2):
                    modules.append(nn.ReLU(inplace=True))
            self.output_net = nn.Sequential(*modules)
        else:
            self.output_net = nn.Sequential(
                nn.Linear(in_features, self._num_classes, bias=False)
            )

    def extract_features(self, x):
        return self.encoder.extract_features(x)

    def forward(self, x):
        x = self.extract_features(x)
        x = self.pool_layer(x)
        x = self.output_net(x)
        return x


class TileModel(nn.Module):
    """Melanoma tile model with attention."""
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
            self._pool_method = 'attention'
            self._pool_params = {'params': {}}
        else:
            self._pool_method = self._pool_params['method']
            self._pool_params['params'] = self._pool_params.get('params', {})
        self._pretrained = pretrained

        self.encoder, self._in_features = model_utils.get_backbone(self._backbone,
                                                                  self._pretrained)

        self._pool_params['params']['in_channels'] = self._in_features
        self.pool_layer = melanoma_config.POOLING_MAP[self._pool_method](**self._pool_params['params'])

        # build output network
        if self._output_net_params is not None:
            modules = []
            if self._output_net_params.get('bn'):
                modules.append(nn.BatchNorm1d(self._in_features, **self._output_net_params['bn']))
            if self._output_net_params.get('dropout'):
                modules.append(nn.Dropout(self._output_net_params['dropout']))
            modules.append(nn.Linear(self._in_features, self._num_classes, bias=False))
            self.output_net = nn.Sequential(*modules)
        else:
            self.output_net = nn.Sequential(
                nn.Linear(self._in_features, self._num_classes, bias=False)
            )
        self.weights = None


    def forward(self, x):
        batch_size, num_tiles, c, h, w = x.shape
        x = torch.cat([*x])
        x = self.encoder(x)
        x = x.reshape(batch_size, num_tiles, self._in_features)
        x, self.weights = self.pool_layer(x)
        x = self.output_net(x)
        return x
