import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations
from torch.utils import data
from core import losses
from core import layers
from data import samplers
from data import augmentation

# update root path for pretrained models
ROOT_PATH = os.path.join(os.getenv('HOME'), 'workspace/kaggle-melanoma')


SCHEDULER_MAP = {
    'one_cycle': torch.optim.lr_scheduler.OneCycleLR,
    'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
    'reduce': torch.optim.lr_scheduler.ReduceLROnPlateau,
    'cyclic': torch.optim.lr_scheduler.CyclicLR,
    'cosine_warm': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    'step': torch.optim.lr_scheduler.StepLR
}

OPTIMIZER_MAP = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD
}

AUGMENTATION_MAP = {
    'transpose': albumentations.Transpose,
    'vertical': albumentations.VerticalFlip,
    'horizontal': albumentations.HorizontalFlip,
    'normalize': albumentations.Normalize,
    'brightness': albumentations.RandomBrightness,
    'contrast': albumentations.RandomContrast,
    'brightness_constrast': albumentations.RandomBrightnessContrast,
    'shift': albumentations.ShiftScaleRotate,
    'gray': albumentations.ToGray,
    'pca': albumentations.FancyPCA,
    'hue': albumentations.HueSaturationValue,
    'gauss_noise': albumentations.GaussNoise,
    'gauss_blur': albumentations.GaussianBlur,
    'compress': albumentations.ImageCompression,
    'dropout': augmentation.CoarseDropout,
    'one_of': albumentations.OneOf,
    'hair': augmentation.AdvancedHairAugmentation,
    'gamma': albumentations.RandomGamma,
    'channel': albumentations.ChannelShuffle,
    'constancy': augmentation.ColorConstancy,
}

ARCH_TO_PRETRAINED = {
    'efficientnet-b0': os.path.join(ROOT_PATH, 'pretrained-models/efficientnet-b0-08094119.pth'),
    'efficientnet-b1': os.path.join(ROOT_PATH, 'pretrained-models/efficientnet-b1-dbc7070a.pth'),
    'efficientnet-b3': os.path.join(ROOT_PATH, 'pretrained-models/efficientnet-b3-c8376fa2.pth'),
    'efficientnet-b4': os.path.join(ROOT_PATH, 'pretrained-models/efficientnet-b4-e116e8b3.pth'),
    'efficientnet-b5': os.path.join(ROOT_PATH, 'pretrained-models/efficientnet-b5-586e6cc6.pth'),
    'efficientnet-b7': os.path.join(ROOT_PATH, 'pretrained-models/efficientnet-b7-dcc49843.pth'),
    'resnext50_32x4d_ssl': 'facebookresearch/semi-supervised-ImageNet1K-models',
    'resnext101_32x4d_ssl': 'facebookresearch/semi-supervised-ImageNet1K-models',
    'resnet18_ssl': 'facebookresearch/semi-supervised-ImageNet1K-models',
    'resnet50_ssl': 'facebookresearch/semi-supervised-ImageNet1K-models',
}

CRITERION_MAP = {
    'bce_smth': losses.BCELabelSmoothingLoss,
    'lbl_smth': losses.LabelSmoothingLoss,
    'mse': nn.MSELoss,
    # 'bce': nn.BCEWithLogitsLoss,
    'bce': losses.BCELoss,
    'l1': nn.L1Loss,
    'l1_smth': nn.SmoothL1Loss,
    'cauchy': losses.CauchyLoss,
    'mse_clip': losses.ClippedMSELoss
}

CLF_CRITERION = ['xent', 'lbl_smth']

SAMPLER_MAP = {
    'random': data.RandomSampler,
    'sequential': data.SequentialSampler,
    'weighted_random': data.WeightedRandomSampler,
    'imbalanced': samplers.ImbalancedSampler,
    'batch': samplers.BatchStratifiedSampler,
    'oversample': samplers.OverSampler,
}

POOLING_MAP = {
    'concat': layers.AdaptiveConcatPool2d,
    'gem': layers.GeM,
    'avg': layers.AdaptiveAvgPool2d,
    'max': layers.AdaptiveMaxPool2d,
    'concat_gem': layers.AdaptiveConcatGeMPool2d,
    'attention': layers.AttentionPool1d,
}

#POSTPROCESSOR_MAP = {
#    'optimized_rounder': postprocess.OptimizedRounder
#}

ACTIVATION_MAP = {
    'relu': nn.ReLU,
    'mish': layers.Mish
}

ANATOM_MAP = {
    'anterior torso': 'torso',
    'lateral torso': 'torso',
    'posterior torso': 'torso',
    'back': 'torso',
    'trunk': 'torso',
    'abdomen': 'torso',
    'face': 'head/neck',
    'chest': 'torso',
    'foot': 'palms/soles',
    'neck': 'head/neck',
    'scalp': 'head/neck',
    'hand': 'palms/soles',
    'ear': 'head/neck',
    'genital': 'oral/genital',
    'acral': 'palms/soles'
}

TRAIN_COLS = [
    'image_name',
    'patient_id',
    'sex',
    'age_approx',
    'anatom_site_general_challenge',
    'source',
    'target'
]


BINARY_METRICS = [
    'precision_score',
    'recall_score',
    'accuracy_score',
    'f1_score',
    'specificity_score',
    'fpr_score'
]
