import albumentations
import numpy as np


class MelanomaAugmentor:
    """Panda augmentor."""

    def __init__(self,
                 augmentations=None,
                 norm_by=None,
                 train_mode=True,
                 dtype='float32'):
        self.augmentations = augmentations
        self.norm_by = norm_by if isinstance(norm_by, list) else [norm_by]
        self.train_mode = train_mode
        self.dtype = dtype
        if self.augmentations is not None:
            self._augmentations = self.augmentations.copy()
            self._normalize = self._augmentations.pop('normalize', None)
        else:
            self._augmentations = None
            self._normalize = None

        # check if normalizing by a stratified factor
        self._norm_by_provider = False
        if self._normalize is not None:
            if self.norm_by is None:
                self._normalize['mean'] = np.array(self._normalize['mean'], dtype=self.dtype)
                self._normalize['std'] = np.array(self._normalize['std'], dtype=self.dtype)
            else:
                for col in self.norm_by:
                    for k in self._normalize[col].keys():
                        self._normalize[col][k]['mean'] = np.array(self._normalize[col][k]['mean'], dtype=self.dtype)
                        self._normalize[col][k]['std'] = np.array(self._normalize[col][k]['std'], dtype=self.dtype)

        self.transform = self._set_transforms()

    def _transform(self, img, stratify_col=None, stratify_value=None):
        if self.norm_by is not None:
            img = (img - self._normalize[stratify_col][stratify_value]['mean']) / self._normalize[stratify_col][stratify_value]['std']
        elif self._normalize is not None:
            img = (img - self._normalize['mean']) / self._normalize['std']
        return self.transform(image=img.astype(self.dtype))['image']

    def _set_transforms(self):
        transforms = []
        for k, v in self._augmentations.items():
            transforms.append(panda_config.AUGMENTATION_MAP[k](**v))
        return albumentations.Compose(transforms)

    def __call__(self, img, **kwargs):
        return self._transform(img, **kwargs)
