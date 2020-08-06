import random
import albumentations
from albumentations import DualTransform
import numpy as np
import config as melanoma_config
import utils.generic_utils as utils


class MelanomaAugmentor:
    """Melanoma augmentor."""

    def __init__(self,
                 augmentations=None,
                 norm_cols=None,
                 train_mode=True,
                 dtype='float32'):
        self.augmentations = augmentations
        self.norm_cols = norm_cols
        if self.norm_cols is not None and not isinstance(self.norm_cols, list):
            self.norm_cols = [self.norm_cols]
        self.train_mode = train_mode
        self.dtype = dtype
        if self.augmentations is not None:
            self._augmentations = self.augmentations.copy()
            self._normalize = self._augmentations.pop('normalize', None)
        else:
            self._augmentations = None
            self._normalize = None

        # check if normalizing by a stratified factor
        if self._normalize is not None:
            if self.norm_cols is None:
                self._normalize['mean'] = np.array(self._normalize['mean'], dtype=self.dtype)
                self._normalize['std'] = np.array(self._normalize['std'], dtype=self.dtype)
            else:
                for col in self.norm_cols:
                    for k in self._normalize[col].keys():
                        self._normalize[col][k]['mean'] = np.array(self._normalize[col][k]['mean'], dtype=self.dtype)
                        self._normalize[col][k]['std'] = np.array(self._normalize[col][k]['std'], dtype=self.dtype)

        self.transform = self._set_transforms()

    def _transform(self, img, stratify_list=None):
        if self.norm_cols is not None:
            norm_dict = utils.deep_get(self._normalize, *stratify_list)
            img = (img - norm_dict['mean']) / self._normalize['std']
        elif self._normalize is not None:
            img = (img - self._normalize['mean']) / self._normalize['std']
        return self.transform(image=img.astype(self.dtype))['image']

    def _set_transforms(self):
        transforms = []
        for key, value in self._augmentations.items():
            if key == 'oneof':
                transforms.append(
                    albumentations.OneOf([
                        melanoma_config.AUGMENTATION_MAP[k](**v) for k, v in value['transforms'].items()
                    ], p=value['p']),
                )
            else:
                transforms.append(melanoma_config.AUGMENTATION_MAP[key](**value))
        return albumentations.Compose(transforms)

    def __call__(self, img, **kwargs):
        return self._transform(img, **kwargs)


class CoarseDropout(DualTransform):

    def __init__(self,
                 n=4,
                 scale=.1,
                 n_max=None,
                 scale_max=None,
                 fill_value=0,
                 fill_params=None,
                 always_apply=False,
                 p=.5):
        super(CoarseDropout, self).__init__(p, always_apply)
        self._n = n
        self._scale = scale
        self._fill_value = fill_value
        self._n_max = n if n_max is None else n_max
        self._scale_max = scale if scale_max is None else scale_max
        self._fill_params = fill_params
        if self._fill_params is not None:
            self._fill_method = self._fill_params['method']
            self._fill_stats = self._fill_params.get('params', {})

    def apply(self, img, **params):
        img = img.copy()
        h, w, d = img.shape
        n = random.randint(self._n, self._n_max)

        for i in range(n):
            h_0 = random.randint(0, h)
            w_0 = random.randint(0, w)
            scale = random.uniform(self._scale, self._scale_max)
            dim = int(scale * h) // 2

            hb = [max(0, h_0 - dim), min(h, h_0 + dim)]
            wb = [max(0, w_0 - dim), min(w, w_0 + dim)]
            if self._fill_params is None:
                img[slice(*hb), slice(*wb),:] = self._fill_value
            elif self._fill_method == 'normal':
                size = ((hb[1]-hb[0]), (wb[1]-wb[0]), 3)
                img[slice(*hb), slice(*wb),:] = np.random.normal(size=size, **self._fill_stats)
            else:
                size = ((hb[1]-hb[0]), (wb[1]-wb[0]), 3)
                img[slice(*hb), slice(*wb),:] = np.random.uniform(size=size, **self._fill_stats)

        return img

    def get_transform_init_args_names(self):
        return ("n", "scale", "n_max", "scale_max", "fill_value", "fill_params")
