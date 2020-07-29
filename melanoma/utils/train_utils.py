import os
import gc
import json
import time
import numpy as np
import tqdm
import torch
from torch.utils.data import DataLoader, RandomSampler

import config as melanoma_config
from data.dataset import MelanomaDataset
from data.augmentation import MelanomaAugmentor
from data.samplers import BatchStratifiedSampler, DistributedSamplerWrapper
from utils import model_utils


def get_optimizer(method, model, params=None):
    params = params or {}
    optim_cls = melanoma_config.OPTIMIZER_MAP[method]
    optim = optim_cls(model.parameters(), **params)
    return optim


def get_scheduler(config, optim, steps_per_epoch=None):
    """Returns an instantiated learning rate scheduler."""
    if not config.get('scheduler'):
        return None
    params = config['scheduler']['params']
    method = config['scheduler']['method']
    if method == 'one_cycle':
        params['epochs'] = params.get('steps', config['steps']+1)
        params['steps_per_epoch'] = params.get('steps_per_epoch', steps_per_epoch)
        # params['total_steps'] = params.get('total_steps', config['steps']+1)
        params['pct_start'] = params.get('pct_start', 1 / config['steps'])
    elif method == 'cosine':
        params['T_max'] = params.get('T_max', config['steps']-1)
    elif method == 'cyclic':
        params['base_lr'] = params.get('base_lr', config['optimizer']['params']['lr'])


    sched_cls = melanoma_config.SCHEDULER_MAP[method]
    sched = sched_cls(optim, **params)
    if config['scheduler'].get('gradual'):
        sched = GradualWarmupScheduler(optim,
                                       after_scheduler=sched,
                                       **config['scheduler']['gradual'])

    return sched


def get_criterion(method, params=None):
    params = params or {}
    if method in melanoma_config.CRITERION_MAP.keys():
        return melanoma_config.CRITERION_MAP[method](**params)
    else:
        return nn.CrossEntropyLoss()


def get_augmentors(params,
                   norm_cols=None,
                   tta_val=False,
                   tta_test=True,
                   fp_16=False):
    if params is None or len(params) == 0:
       train_aug = None
       val_aug = None
       test_aug = None
    else:
        dtype = 'float16' if fp_16 else 'float32'
        train_aug = MelanomaAugmentor(params, norm_cols=norm_cols, dtype=dtype)

        if tta_val:
            val_aug = MelanomaAugmentor(params, norm_cols=norm_cols, dtype=dtype)
        elif 'normalize' in params:
            val_aug = MelanomaAugmentor({'normalize': params['normalize']},
                                        norm_cols=norm_cols,
                                        dtype=dtype)
        else:
            val_aug = None

        if tta_test:
            test_aug = MelanomaAugmentor(params, norm_cols=norm_cols)
        elif 'normalize' in params:
            test_aug = MelanomaAugmentor({'normalize': params['normalize']}, norm_cols=norm_cols)
        else:
            test_aug = None

    return train_aug, val_aug, test_aug


def get_img_stats(dl):
    img_mean = 0
    img_var = 0
    num_samples = len(dl.dataset)

    for i, (x, y) in tqdm(enumerate(dl), total=len(dl)):
        x_batch = x.view(x.shape[0], 3, -1)
        img_mean += x_batch.mean(-1).sum(0)
        img_var += x_batch.var(-1).sum(0)

    img_mean /= num_samples
    img_var /= num_samples
    img_stats = {
        'mean': img_mean,
        'std': np.sqrt(img_var)
    }
    return img_stats


def get_sampler(ds,
                method='random',
                params=None,
                distributed=False,
                batch_size=None,
                random_state=None):
    if method is not None:
        params = params or {}
        if method == 'batch':
            sampler = BatchStratifiedSampler(ds,
                                             list(range(len(ds))),
                                             ds.get_labels(),
                                             batch_size=batch_size,
                                             random_state=random_state,
                                             **params)
        elif melanoma_config.SAMPLER_MAP.get(method):
            if method == 'weighted_random':
                params['num_samples'] = len(ds.image_ids)
            sampler = melanoma_config.SAMPLER_MAP[method](ds, **params)
        else:
            raise ValueError("Unrecognized sampler.")
    else:
        sampler = None

    if distributed:
        sampler = DistributedSamplerWrapper(sampler)
    return sampler
