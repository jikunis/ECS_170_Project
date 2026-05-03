'''
Concrete IO class for a specific dataset
'''

from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')
        import pandas as pd

        train = pd.read_csv(self.dataset_source_folder_path + 'train.csv', header=None, sep=',')
        test = pd.read_csv(self.dataset_source_folder_path + 'test.csv', header=None, sep=',')

        X_train = train.iloc[:, 1:].values / 255.0
        y_train = train.iloc[:, 0].values
        X_test = test.iloc[:, 1:].values / 255.0
        y_test = test.iloc[:, 0].values

        return {'train': {'X': X_train, 'y': y_train}, 'test':  {'X': X_test,  'y': y_test}}
