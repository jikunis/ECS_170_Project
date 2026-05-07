import pickle
import numpy as np
from local_code.base_class.dataset import dataset

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')

        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        raw = pickle.load(f)
        f.close()

        X_train, y_train = [], []
        for instance in raw['train']:
            image = np.array(instance['image'])
            if image.ndim == 3 and self.dataset_source_file_name == 'ORL':
                image = image[:, :, 0]  # take R channel only
            X_train.append(image)
            y_train.append(instance['label'])

        X_test, y_test = [], []
        for instance in raw['test']:
            image = np.array(instance['image'])
            if image.ndim == 3 and self.dataset_source_file_name == 'ORL':
                image = image[:, :, 0]
            X_test.append(image)
            y_test.append(instance['label'])

        return {
            'train': {'X': X_train, 'y': y_train},
            'test':  {'X': X_test,  'y': y_test}
        }