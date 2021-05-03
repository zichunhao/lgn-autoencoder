import torch
from torch.utils.data import Dataset, DataLoader
import logging
from lgn.cg_lib import normsq4

class JetDataset(Dataset):
    """
    PyTorch dataset.
    """
    def __init__(self, data, num_pts=-1, shuffle=True):

        self.data = data

        if num_pts < 0:
            self.num_pts = len(data['Nobj'])
        else:
            if num_pts > len(data['Nobj']):
                logging.warn(f'Desired number of points ({num_pts}) is greater than the number of data points ({len(data)}) available in the dataset!')
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

def initialize_data(path, batch_size, num_train, num_test=-1, num_val=-1):
    data = torch.load(path)

    # Calculate node masks and edge masks
    if 'labels' in data:
        node_mask = data['labels']
        node_mask = node_mask.to(torch.uint8)
    else:
        node_mask = data['p4'][...,0] != 0
        node_mask = node_mask.to(torch.uint8)
    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    data['node_mask'] = node_mask
    data['edge_mask'] = edge_mask

    jet_data = JetDataset(data, shuffle=True) # The original data is not shuffled yet

    if not (num_test < 0 or num_val < 0): # Specified num_test and num_val
        assert num_train + num_test + num_val <= len(jet_data), f"num_train + num_test + num_val = {num_train + num_test + num_val} \
                                                                is larger than the data size {len(jet_data)}!"

        # split into training, testing, and valid set
        jet_data = JetDataset(jet_data[0: num_train + num_test + num_val], shuffle = True)
        train_set, test_set, valid_set = torch.utils.data.random_split(jet_data, [num_train, num_test, num_val])
        train_loader = DataLoader(jet_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(jet_data, batch_size=batch_size, shuffle=False)
        valid_loader = DataLoader(jet_data, batch_size=batch_size, shuffle=False)

    # Unspecified num_test and num_val -> Choose training data and then divide the rest in half into testing and validation datasets
    else:
        assert num_train <= len(jet_data), f"num_train = {num_train} is larger than the data size {len(jet_data)}!"

        # split into training, testing, and valid sets
        # split the rest in half
        num_test = int((len(jet_data) - num_train) / 2)
        num_val = len(jet_data) - num_train - num_test
        train_set, test_set, valid_set = torch.utils.data.random_split(jet_data, [num_train, num_test, num_val])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    print('Data loaded')

    return train_loader, test_loader, valid_loader

"""
Extract input from data.

Parameters
----------
data : `dict`
    The jet data.

Returns
-------
node_scalars : `torch.Tensor`
    Tensor of scalars for each node.
node_ps: : `torch.Tensor`
    Momenta of the nodes
node_mask : `torch.Tensor`
    Node mask used for batching data.
edge_mask: `torch.Tensor`
    Edge mask used for batching data.
"""
def prepare_input(data, scale, cg_levels=True, device=None, dtype=None):


    node_ps = data['p4'].to(device=device, dtype=dtype) * scale

    data['p4'].requires_grad_(True)

    node_mask = data['node_mask'].to(device=device, dtype=torch.uint8)
    edge_mask = data['edge_mask'].to(device=device, dtype=torch.uint8)

    scalars = torch.ones_like(node_ps[:,:, 0]).unsqueeze(-1)
    scalars = normsq4(node_ps).abs().sqrt().unsqueeze(-1)

    if 'scalars' in data.keys():
        scalars = torch.cat([scalars, data['scalars'].to(device=device, dtype=dtype)], dim=-1)

    if not cg_levels:
        scalars = torch.stack(tuple(scalars for _ in range(scalars.shape[-1])), -2).to(device=device, dtype=dtype)

    return scalars, node_ps, node_mask, edge_mask
