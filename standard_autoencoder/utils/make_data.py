import torch
from torch.utils.data import Dataset, DataLoader
import logging


class JetDataset(Dataset):
    """
    PyTorch dataset.
    """

    def __init__(self, data, vec_dims=3, num_pts=-1, shuffle=True):

        self.data = data
        self.p4 = data['p4'] if vec_dims == 4 else data['p4'][..., 1:]

        if num_pts < 0:
            self.num_pts = len(data['Nobj'])
        else:
            if num_pts > len(data['Nobj']):
                logging.warn(f'Desired number of points ({num_pts}) is greater than '
                             f'the number of data points ({len(data)}) available in the dataset!')
                self.num_pts = len(data['Nobj'])
            else:
                self.num_pts = num_pts

        if shuffle:
            self.perm = torch.randperm(len(data['Nobj']))[:self.num_pts]
        else:
            self.perm = None

    def __len__(self):
        return self.num_pts

    def __getitem__(self, idx):
        if self.perm is not None:
            idx = self.perm[idx]
        return self.p4[idx]


def initialize_data(path, batch_size, train_fraction, vec_dims=3, num_val=None):
    data = torch.load(path)

    jet_data = JetDataset(data, vec_dims=vec_dims, shuffle=True)  # The original data is not shuffled yet

    if train_fraction > 1:
        num_train = int(train_fraction)
        if num_val is None:
            num_jets = len(data['Nobj'])
            num_val = num_jets - num_train
        else:
            num_others = len(data['Nobj']) - num_train - num_val
            train_set, val_set, _ = torch.utils.data.random_split(jet_data, [num_train, num_val, num_others])
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            valid_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
            return train_loader, valid_loader
    else:
        if train_fraction < 0:
            train_fraction = 0.8
        num_jets = len(data['Nobj'])
        num_train = int(num_jets * train_fraction)
        num_val = num_jets - num_train

    # split into training and validation set
    train_set, val_set = torch.utils.data.random_split(jet_data, [num_train, num_val])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    logging.info('Data loaded')

    return train_loader, valid_loader


def initialize_test_data(path, batch_size, vec_dims=3):
    data = torch.load(path)
    jet_data = JetDataset(data, vec_dims=vec_dims, shuffle=False)
    return DataLoader(jet_data, batch_size=batch_size, shuffle=True)
