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


class TileModel(nn.Module):
    """Melanoma tile model with attention."""
    def __init__(self,
                 backbone,
                 num_classes=1,
                 attention_net_params=None,
                 output_net_params=None,
                 attention_dropout=0.5,
                 pool_params=None,
                 pretrained=True):
        super().__init__()
        self._backbone = backbone
        self._num_classes = num_classes
        self._pool_params = pool_params
        self._attention_net_params = attention_net_params
        self._attention_dropout = attention_dropout
        self._output_net_params = output_net_params
        if self._pool_params is None:
            self._pool_method = 'avg'
            self._pool_params = {'params': {}}
        else:
            self._pool_method = self._pool_params['method']
            self._pool_params['params'] = self._pool_params.get('params', {})
        self._pretrained = pretrained

        self.encoder, self._in_features = model_utils.get_backbone(self._backbone,
                                                                  self._pretrained)
        self.pool_layer = melanoma_config.POOLING_MAP[self._pool_method](**self._pool_params['params'])

        if self._pool_method == 'concat':
            self._in_features *= 2
        elif self._pool_method == 'concat_gem':
            self._in_features *= 3

        # build the attention network
        if self._attention_net_params is not None:
            self._attention_dim = self._attention_net_params.get('hidden_dim', self._in_features)
            modules = []
            # add dropout
            if self._attention_net_params.get('dropout'):
                modules.append(nn.Dropout(self._attention_net_params['dropout']))

            modules.append(nn.Conv2d(self._in_features, self._attention_dim, kernel_size=3, padding=1))
            modules.append(nn.ReLU())

            # add batchnorm
            if self._attention_net_params.get('bn'):
                modules.append(nn.BatchNorm2d(self._attention_dim, **self._attention_net_params['bn']))
            modules.append(nn.Conv2d(self._attention_dim, 1, kernel_size=3, padding=1))
            self.attention_net = nn.Sequential(*modules)
        else:
            self._attention_dim = self._in_features
            self.attention_net = nn.Sequential(
                nn.Conv2d(self._in_features, self._attention_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(self._attention_dim, 1, kernel_size=3, padding=1)
            )

        if self._attention_dropout > 0:
            self.attn_dropout = nn.Dropout(self._attention_dropout)
        else:
            self.attn_dropout = nn.Identity()
        # build output network
        if self._output_net_params is not None:
            modules = []
            if self._output_net_params.get('dropout'):
                modules.append(nn.Dropout(self._output_net_params['dropout']))
            modules.append(nn.ReLU())
            if self._output_net_params.get('bn'):
                modules.append(nn.BatchNorm1d(self._attention_dim, **self._output_net_params['bn']))
            modules.append(nn.Linear(self._in_features, self._num_classes, bias=False))
            self.output_net = nn.Sequential(*modules)
        else:
            self.output_net = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(self._in_features, self._num_classes, bias=False)
            )
        self.weights = None

    def compute_attention(self, x, num_tiles, batch_size=1):
        x = x.reshape(batch_size, num_tiles, self._in_features, 1, 1)
        x = x.transpose(1, 2).contiguous().view(batch_size, self._in_features, 7, 7)

        x_attention = self.attention_net(x)
        weights = F.softmax(x_attention.reshape(batch_size, -1), dim=1)
        x = self.attn_dropout(x)
        x = x * weights.view_as(x_attention)
        x = self.pool_layer(x)
        return x, weights

    def forward(self, x):
        batch_size, num_tiles, c, h, w = x.shape
        x = torch.cat([*x])
        x = self.encoder(x)
        x, self.weights = self.compute_attention(x, num_tiles, batch_size=batch_size)
        x = self.output_net(x)
        return x
