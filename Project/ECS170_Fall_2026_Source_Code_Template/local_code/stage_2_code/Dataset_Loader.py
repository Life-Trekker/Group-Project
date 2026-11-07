'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_train_source_file_name = None
    dataset_test_source_file_name = None
    label_column = 0    # -1 = last column, 0 = first column
    has_header = False
    
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    # since loading multiple files, make that process a seperate helper function
    def _load_file(self, file_name):
        X, y = [], []
        with open(self.dataset_source_folder_path + file_name, 'r') as f:
            if self.has_header:
                next(f)

            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = [int(v) for v in line.split(',')]
                y.append(row.pop(self.label_column))
                X.append(row)

        return {'X': X, 'y': y}

    # now load calls load_files to get both the train and test data
    def load(self):
        print('loading data...')

        train = self._load_file(self.dataset_train_source_file_name)
        test = self._load_file(self.dataset_test_source_file_name)

        # print('train length:', len(train['X']), 'test length:', len(test['X']))
        return {'train': train, 'test': test}
    