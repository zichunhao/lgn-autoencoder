from torch.utils.data import Dataset
import torch
import logging
from typing import Dict, Union


class JetDataset(Dataset):
    """
    PyTorch dataset.
    """

    def __init__(self, data: Dict[str, torch.Tensor], num_pts: Union[int, float] = -1, shuffle: bool = True):

        self.data = data
        self.shuffle = shuffle
        if 'Nobj' not in data.keys():
            try:
                data['Nobj'] = data['labels'].sum(dim=-1)
            except KeyError:
                data['Nobj'] = data['masks'].sum(dim=-1)

        total_pts = len(data['Nobj'])
        if num_pts < 0:
            self.num_pts = total_pts
        elif num_pts <= 1:
            # num_pts is a fraction of the total number of data points
            self.num_pts = int(num_pts * total_pts)
        else:
            # num_pts is an absolute number of data points
            if num_pts > total_pts:
                logging.warn(f'Desired number of points ({num_pts}) is greater than '
                             f'the number of data points ({len(data)}) available in the dataset!')
                self.num_pts = total_pts
            else:
                self.num_pts = num_pts
        
        logging.info(f'Using {self.num_pts} data points out of {len(data["Nobj"])} available.')

        if shuffle:
            self.perm = torch.randperm(total_pts)[:self.num_pts]
        else:
            self.perm = torch.arange(self.num_pts)

    def __len__(self):
        return self.num_pts

    def __getitem__(self, idx):
        if self.shuffle:
            idx = self.perm[idx]
        return {key: val[idx] for key, val in self.data.items()}

    def add(self, data):
        data = {
            key: torch.cat([self.data[key], data[key]], dim=0) 
            for key in self.data.keys()
        }
        self.__init__(self.data, num_pts=self.num_pts, shuffle=self.shuffle)