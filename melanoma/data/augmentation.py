import os
import random
import cv2
import albumentations
from albumentations import DualTransform
import numpy as np
import config as melanoma_config
import utils.generic_utils as utils


class MelanomaAugmentor:
    """Melanoma augmentor."""

    def __init__(self,
                 augmentations=None,
                 post_norm=None,
                 norm_cols=None,
                 train_mode=True,
                 dtype='float32'):
        self.augmentations = augmentations
        self._post_norm = post_norm if post_norm is not None else []
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

        self.post_transform = None
        self.transform = None
        self._set_transforms()

    def _transform(self, img, stratify_list=None):
        img = self.transform(image=img.astype(np.uint8))['image'].astype(self.dtype) / 255
        if self.norm_cols is not None:
            norm_dict = utils.deep_get(self._normalize, *stratify_list)
            img = (img - norm_dict['mean']) / self._normalize['std']
        elif self._normalize is not None:
            img = (img - self._normalize['mean']) / self._normalize['std']
        return self.post_transform(image=img)['image']

    def _set_transforms(self):
        transforms = []
        post_transforms = []
        for key, value in self._augmentations.items():
            if key == 'oneof':
                transforms.append(
                    albumentations.OneOf([
                        melanoma_config.AUGMENTATION_MAP[k](**v) for k, v in value['transforms'].items()
                    ], p=value['p']),
                )
            elif key in self._post_norm:
                post_transforms.append(melanoma_config.AUGMENTATION_MAP[key](**value))
            else:
                transforms.append(melanoma_config.AUGMENTATION_MAP[key](**value))
        self.post_transform = albumentations.Compose(post_transforms)
        self.transform = albumentations.Compose(transforms)

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


class AdvancedHairAugmentation(DualTransform):

    def __init__(self, root, num_hairs=4, max_hairs=None, p=0.5, always_apply=False):
        super(AdvancedHairAugmentation, self).__init__(always_apply, p)
        self.root = root
        self.num_hairs = num_hairs
        self.max_hairs = self.num_hairs if max_hairs is None else max_hairs
        self.filenames = os.listdir(self.root)

    def apply(self, img, **params):
        img = img.copy()
        num_hairs = random.randint(self.num_hairs, self.max_hairs)

        height, width, _ = img.shape  # target image width and height
        for _ in range(num_hairs):
            hair = cv2.imread(os.path.join(self.root, random.choice(self.filenames)))
            hair = cv2.flip(hair, random.choice([-1, 0, 1]))
            hair = cv2.rotate(hair, random.choice([0, 1, 2]))

            h_height, h_width, _ = hair.shape  # hair image width and height
            roi_ho = random.randint(0, img.shape[0] - hair.shape[0])
            roi_wo = random.randint(0, img.shape[1] - hair.shape[1])
            roi = img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width]

            img2gray = cv2.cvtColor(hair, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            hair_fg = cv2.bitwise_and(hair, hair, mask=mask)

            dst = cv2.add(img_bg, hair_fg)
            img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width] = dst

        return img


class ColorConstancy(DualTransform):

    def __init__(self,
                 power=6,
                 gamma=None,
                 always_apply=False,
                 p=0.5):
        self._power = power
        self._gamma = gamma
        super(ColorConstancy, self).__init__(p, always_apply)

    def apply(self, img, **params):
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_dtype = img.dtype

        if self._gamma is not None:
            img = img.astype('uint8')
            look_up_table = np.ones((256 ,1), dtype='uint8') * 0
            for i in range(256):
                look_up_table[i][0] = 255*pow(i/255, 1/self._gamma)
            img = cv2.LUT(img, look_up_table)

        img = img.astype('float32')
        img_power = np.power(img, self._power)
        rgb_vec = np.power(np.mean(img_power, (0,1)), 1/self._power)
        rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
        rgb_vec = rgb_vec/rgb_norm
        rgb_vec = 1/(rgb_vec*np.sqrt(3))
        img = np.multiply(img, rgb_vec)

        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        return img.astype(img_dtype)

    def get_transform_init_args_names(self):
        return ("power", "gamma")
