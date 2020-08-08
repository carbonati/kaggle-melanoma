import os
import skimage.io
import numpy as np
import torch
from copy import deepcopy
from collections import defaultdict
from torch.utils.data import Dataset

from utils import data_utils


class MelanomaDataset(Dataset):
    """Melanoma dataset."""
    def __init__(self,
                 df,
                 image_dir='',
                 target_col='target',
                 img_format='jpg',
                 augmentor=None,
                 norm_cols=None,
                 fp_16=False,
                 seed=None):
        self.df = df
        self.image_dir = image_dir
        self.augmentor = augmentor
        self.target_col = target_col
        self.img_format = img_format
        self.norm_cols = norm_cols
        self._seed = seed
        self._fp_16 = fp_16
        self._dtype = 'float16' if self._fp_16 else 'float32'
        self._dtype_torch = torch.float16 if self._fp_16 else torch.float32
        self._random_state = np.random.RandomState(self._seed)
        self.image_ids = self.df['image_name'].values.tolist()
        if self.target_col is not None:
            self.labels = df[self.target_col].values.tolist()
            self.training = True
        else:
            self.labels = None
            self.training = False

        self.files = None
        self.image_id_to_filepath = None

        self._set_image_id_to_filepaths()
        self._prepare_args()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        df_index = self.df.iloc[index]
        img = data_utils.load_image(os.path.join(df_index['image_dir'], self.image_dir),
                                    self.image_ids[index],
                                    self.img_format)

        stratify_list = []
        if self.norm_cols is not None:
            for c in self.norm_cols:
                stratify_list.append(c)
                stratify_list.append(df_index[c])

        img = self.preprocess(img, stratify_list=stratify_list)
        img = torch.tensor(img, dtype=self._dtype_torch).permute(2, 0, 1)

        if self.training:
            return img, self.labels[index]
        else:
            return img

    def _prepare_args(self):
        pass

    def _set_image_id_to_filepaths(self):
        self.image_id_to_filepath = defaultdict(list)
        for image_id in self.image_ids:
            self.image_id_to_filepath[image_id] = os.path.join(self.image_dir, f'{image_id}.{self.img_format}')

    def _set_files(self):
        self.files = [fp for fps in self.image_id_to_filepaths.values() for fp in fps]

    def preprocess(self, img, **kwargs):
        if self.augmentor is None:
            return np.asarray(img) / 255
        else:
            return self.augmentor(img, **kwargs)

    def get_labels(self):
        return self.labels

    def get_image_ids(self):
        return self.image_ids

    def reset_state(self, seed=None):
        """Reset dataset random state for bagging + mutltiprocessing"""
        self._random_state = np.random.RandomState(seed)
