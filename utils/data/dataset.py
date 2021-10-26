from torch.utils.data import Dataset
import torch
import logging


class JetDataset(Dataset):
    """
    PyTorch dataset.
    """

    def __init__(self, data, num_pts=-1, shuffle=True):

        self.data = data
        if 'Nobj' not in data.keys():
            try:
                data['Nobj'] = data['labels'].sum(dim=-1)
            except KeyError:
                data['Nobj'] = data['masks'].sum(dim=-1)

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
        return {key: val[idx] for key, val in self.data.items()}
