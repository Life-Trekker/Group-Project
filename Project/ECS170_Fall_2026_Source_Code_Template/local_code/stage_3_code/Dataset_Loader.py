'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import pickle
import numpy as np

from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None   # one pickle holds both train and test
    dataset_name = None               # 'ORL', 'MNIST', or 'CIFAR'
    normalize = True                  # scale pixels to [0, 1]

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    # per-dataset preprocessing of a list of instances -> (X, y) numpy arrays
    def _process_split(self, instances):
        X = np.stack([np.array(inst['image']) for inst in instances]).astype(np.float32)
        y = np.array([inst['label'] for inst in instances], dtype=np.int64)

        if self.dataset_name == 'ORL':
            X = X[:, :, :, 0][:, None, :, :]    # (N,112,92,3) -> (N,1,112,92), one channel
            y = y - 1                           # labels 1..40 -> 0..39
        elif self.dataset_name == 'MNIST':
            X = X[:, None, :, :]                # (N,28,28) -> (N,1,28,28)
        elif self.dataset_name == 'CIFAR':
            X = X.transpose(0, 3, 1, 2)         # (N,32,32,3) -> (N,3,32,32)
        else:
            raise ValueError(f"Unknown dataset_name: {self.dataset_name}")

        if self.normalize:
            X = X / 255.0

        return {'X': X, 'y': y}

    def load(self):
        print('loading data...')

        with open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb') as f:
            raw = pickle.load(f)

        train = self._process_split(raw['train'])
        test = self._process_split(raw['test'])

        return {'train': train, 'test': test}