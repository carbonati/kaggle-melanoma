import os
import gc
import json
import time
import numpy as np
import tqdm
import torch
from copy import deepcopy
from sklearn import metrics as sk_metrics
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from torch.utils.data.distributed import DistributedSampler

import config as melanoma_config
from data.dataset import MelanomaDataset
from data.augmentation import MelanomaAugmentor
from data.samplers import BatchStratifiedSampler, DistributedSamplerWrapper, DistributedBatchSampler
from utils import model_utils
from evaluation import metrics as eval_metrics


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


def get_criterion(method,
                  params=None,
                  df_train=None,
                  target_col='target'):
    params = params or {}
    if params.get('class_weight') == 'balanced' and df_train is not None:
        params.pop('class_weight')
        pos_weight = (df_train[target_col] == 0).sum() / (df_train[target_col] == 1).sum()
        print(f'Setting `pos_weight` to {pos_weight}')
        params = dict(params, **{'pos_weight': torch.tensor(pos_weight)})

    if method in melanoma_config.CRITERION_MAP.keys():
        return melanoma_config.CRITERION_MAP[method](**params)
    else:
        return nn.CrossEntropyLoss()


def get_augmentors(transforms=None,
                   norm_cols=None,
                   post_norm=None,
                   tta_val=False,
                   tta_test=True,
                   train_only=None,
                   fp_16=False):
    if transforms is None or len(transforms) == 0:
       train_aug = None
       val_aug = None
       test_aug = None
    else:
        dtype = 'float16' if fp_16 else 'float32'
        train_aug = MelanomaAugmentor(transforms, post_norm=post_norm, norm_cols=norm_cols, dtype=dtype)
        eval_transforms = deepcopy(transforms)
        if train_only is not None:
            for p in train_only:
                print(f'Removing `{p}` from eval augmentations.')
                eval_transforms.pop(p)

        if tta_val:
            val_aug = MelanomaAugmentor(eval_transforms, post_norm=post_norm, norm_cols=norm_cols, dtype=dtype)
        elif 'normalize' in transforms:
            val_aug = MelanomaAugmentor({'normalize': eval_transforms['normalize']},
                                        norm_cols=norm_cols,
                                        post_norm=post_norm,
                                        dtype=dtype)
        else:
            val_aug = None

        if tta_test:
            test_aug = MelanomaAugmentor(eval_transforms, post_norm=post_norm, norm_cols=norm_cols)
        elif 'normalize' in transforms:
            test_aug = MelanomaAugmentor({'normalize': eval_transforms['normalize']}, post_norm=post_norm, norm_cols=norm_cols)
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
                rank=None,
                num_replicas=None,
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

    # df = sampler.data_source.df
    if distributed:
        sampler = DistributedSamplerWrapper(sampler, batch_size, num_replicas=num_replicas, rank=rank)
        # sampler = DistributedSampler(sampler)
    #indices = []
    #for idx in sampler:
    #    indices.append(idx)
    #print(indices[:100])
    return sampler


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_optimal_th(y_true, y_pred):
    fpr, tpr, thresholds = sk_metrics.roc_curve(y_true, y_pred)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold


def compute_scores(y_true, y_pred, metrics, logits=True, th=''):
    if logits:
        y_pred = sigmoid(y_pred)
    th = compute_optimal_th(y_true, y_pred)
    scores = {}
    for name in metrics:
        if hasattr(sk_metrics, name):
            fn = getattr(sk_metrics, name)
        elif hasattr(eval_metrics, name):
            fn = getattr(eval_metrics, name)
        else:
            raise ValueError(f'Unrecognized `metric` {name}.')
        if name in melanoma_config.BINARY_METRICS:
            y_pred_final = y_pred > th
        else:
            y_pred_final = y_pred
        try:
           score = fn(y_true, y_pred_final)
        except:
            score = 0
        scores[name] = score

    return scores


def regularized_criterion(criterion, y_pred, y, y_b, lam):
    return lam * criterion(y_pred, y) + (1 - lam) * criterion(y_pred, y_b)
