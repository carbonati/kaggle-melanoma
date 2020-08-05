import os
import json
import glob
import warnings
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import pretrainedmodels
from torchvision import models as _models
from efficientnet_pytorch import EfficientNet as enet
from core import models as melanoma_models
import config as melanoma_config


def load_config(ckpt_dir):
    config_filepath = os.path.join(ckpt_dir, 'config.json')
    if not os.path.exists(config_filepath):
        # walk up one directory to find config file.
        config_filepath = os.path.join(
            os.path.realpath(os.path.join(ckpt_dir, '..')),
            'config.json'
        )
    with open(config_filepath, 'r') as f:
        return json.load(f)


def load_state_dict(ckpt_dir, step=None, filename=None, device='cuda'):
    if step is None and filename is None:
        raise ValueError('`step` and `filename` cannot both be None.')
    if step is not None:
        filepaths = glob.glob(os.path.join(ckpt_dir, f'ckpt_{step:04d}_*'))
        if len(filepaths) == 0:
            raise ValueError(f'Found 0 matches for step {step} in {ckpt_dir}')
        elif len(filepaths) > 1:
            warnings.warn(f'Found 3 matches for step {step} in {ckpt_dir}. Loading the first match.')
            filepath = filepaths[0]
        else:
            filepath = filepaths[0]
    else:
        filepath = os.path.join(ckpt_dir, filename)

    print(f'Loading state_dict from {filepath}')
    state_dict = torch.load(filepath, map_location=torch.device(device))
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    return state_dict


def load_best_state_dict(ckpt_dir, step=None, device='cuda'):
    if step is None:
        steps = [int(fn.split('_')[1]) for fn in os.listdir(ckpt_dir) if fn.startswith('ckpt_')]
        if len(steps) == 0:
            raise ValueError(f'Found 0 ckpts in {ckpt_dir}')
        best_step = max(steps)
    elif isinstance(step, str):
        df_hist = pd.read_csv(os.path.join(ckpt_dir, 'history.csv'))
        if step == 'val_loss':
            best_step = df_hist.loc[df_hist[step].idxmin(), 'epoch']
        else:
            best_step = df_hist.loc[df_hist[step].idxmax(), 'epoch']
    else:
        best_step = step
    filename = [fn for fn in os.listdir(ckpt_dir) if fn.startswith(f'ckpt_{best_step:04d}')][0]
    state_dict = load_state_dict(ckpt_dir, filename=filename, device=device)
    return state_dict


def load_model(ckpt_dir,
               step=None,
               filename=None,
               device='cuda',
               **kwargs):
    """Loads pretrained model from disk."""
    config = load_config(ckpt_dir)
    model_params = config['model']
    model_params['pretrained'] = False
    model_params.update(kwargs)
    if filename is None:
        state_dict = load_best_state_dict(ckpt_dir, step=step, device=device)
    else:
        state_dict = load_state_dict(ckpt_dir, filename=filename, device=device)
    model = get_model(**model_params).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_model(backbone, **kwargs):
    if backbone.startswith('efficientnet'):
        model = melanoma_models.EfficientModel(backbone, **kwargs)
    else:
        model = melanoma_models.BaseModel(backbone, **kwargs)
    return model


def get_backbone(backbone, pretrained=True, **kwargs):
    if backbone in ['resnext50_32x4d_ssl', 'resnet18_ssl', 'resnet50_ssl', 'resnext101_32x4d_ssl']:
        if pretrained:
            model = torch.hub.load(melanoma_config.ARCH_TO_PRETRAINED[backbone], backbone)
        else:
            model = getattr(_models, backbone.split('_ssl')[0])(pretrained=pretrained)
        encoder = nn.Sequential(*list(model.children())[:-2])
        in_features = model.fc.in_features
    elif backbone in ['resnet18', 'resnet34', 'resnet50']:
        pretrained = 'imagenet' if pretrained else None
        model = getattr(_models, backbone)(pretrained=pretrained)
        in_features = model.fc.in_features
        encoder = nn.Sequential(*list(model.children())[:-2])
    elif backbone in ['se_resnext50_32x4d', 'se_resnext101_32x4d', 'se_resnet50', 'se_resnet101', 'se_resnet152']:
        pretrained = 'imagenet' if pretrained else None
        model = getattr(pretrainedmodels, backbone)(pretrained=pretrained)
        encoder = nn.Sequential(*list(model.children())[:-2])
        in_features = model.last_linear.in_features
    elif backbone.startswith('efficientnet'):
        if pretrained:
            encoder = enet.from_pretrained(backbone, **kwargs)
        else:
            encoder = enet.from_name(backbone, **kwargs)
        in_features = encoder._fc.in_features
    elif backbone == 'inception_resnet_v2':
        pretrained = 'imagenet' if pretrained else None
        encoder = pretrainedmodels.inceptionresnetv2(pretrained=pretrained)
        in_features = encoder.last_linear.in_features
        encoder.last_linear = nn.Identity()
    elif backbone == 'inception_v4':
        pretrained = 'imagenet' if pretrained else None
        encoder = pretrainedmodels.inceptionv4(pretrained=pretrained)
        in_features = encoder.last_linear.in_features
        encoder.last_linear = nn.Identity()
    else:
        raise ValueError(f'Unrecognized backbone {backbone}')

    return encoder, in_features


def get_emb_model(model):
    emb_model = nn.Sequential(*list(model.children())[:-1])
    for param in emb_model.parameters():
        param.requires_grad = False
    return emb_model
