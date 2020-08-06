import torch
import numpy as np
from operator import itemgetter
from torch.utils.data import Sampler, DistributedSampler, Dataset
from utils import data_utils


class BatchStratifiedSampler(Sampler):
    """Stratified batch sampler."""
    def __init__(self,
                 data_source,
                 indices,
                 labels,
                 batch_size,
                 drop_last=False,
                 random_state=6969):
        self.data_source = data_source
        self.indices = indices
        self.labels = labels
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.random_state = random_state

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        indices = data_utils.stratify_batches(
            self.indices,
            self.labels,
            self.batch_size,
            drop_last=self.drop_last,
            random_state=self.random_state
        )
        return iter(indices)


class DatasetFromSampler(Dataset):
    """Dataset indices from a sampler."""
    def __init__(self, sampler: Sampler):
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """https://github.com/catalyst-team/catalyst/blob/master/catalyst/data/sampler.py"""
    def __init__(self, sampler, **kwargs):
        super(DistributedSamplerWrapper, self).__init__(DatasetFromSampler(sampler),
                                                        **kwargs)
        self.sampler = sampler

    def __iter__(self):
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        return iter(itemgetter(*indexes_of_indexes)(self.dataset))



class ImbalancedSampler(Sampler):

    def __init__(self,
                 dataset,
                 indices=None,
                 num_samples=None,
                 callback_get_label=None):
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices)if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.get_labels()[idx]

    def __iter__(self):
        return (
            self.indices[i]
            for i
            in torch.multinomial(self.weights, self.num_samples, replacement=True)
        )

    def __len__(self):
        return self.num_samples


class OverSampler(Sampler):
    """Oversampler class to balance positive and negative samples."""

    def __init__(self, dataset, indices=None, labels=None):
        self.dataset = dataset
        self.num_samples = len(dataset)
        self.indices = list(range(self.num_samples)) if indices is None else indices
        self.labels = self.dataset.get_labels() if labels is None else labels
        self._label_count = {}
        self._label_indices = {}
        for c in np.unique(self.labels):
            label_indices = [i for i in self.indices if self.labels[i] == c]
            self._label_indices[c] = label_indices
            self._label_count[c] = len(label_indices)
        self._num_oversampled = self._label_count[0] * 2

    def __len__(self):
        return self._num_oversampled

    def __iter__(self):
        sampled_pos_indices = np.random.choice(self._label_indices[1],
                                               self._label_count[0],
                                               replace=True)
        indices = np.concatenate((self._label_indices[0], sampled_pos_indices))
        return (
            indices[i]
            for i
            in np.random.choice(range(len(self)), len(self), replace=False)
        )
