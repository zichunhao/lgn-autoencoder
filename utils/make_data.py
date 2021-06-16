import os
import os.path as osp
import numpy as np
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
                logging.warn(
                    f'Desired number of points ({num_pts}) is greater than the number of data points ({len(data)}) available in the dataset!')
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


def initialize_data(path, batch_size, num_train, num_test=-1, num_val=-1, test_batch_size=None):
    data = torch.load(path)

    jet_data = JetDataset(data, shuffle=True)  # The original data is not shuffled yet

    if not (num_test < 0 or num_val < 0):  # Specified num_test and num_val
        assert num_train + num_test + num_val <= len(jet_data), f"num_train + num_test + num_val = {num_train + num_test + num_val}" \
                                                                f"is larger than the data size {len(jet_data)}!"

        # split into training, testing, and valid set
        jet_data = JetDataset(jet_data[0: num_train + num_test + num_val], shuffle=True)
        train_set, test_set, valid_set = torch.utils.data.random_split(jet_data,
                                                                       [num_train, num_test, num_val])
        train_loader = DataLoader(jet_data, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(jet_data, batch_size=batch_size, shuffle=True)
        if test_batch_size is None:
            test_loader = DataLoader(jet_data, batch_size=batch_size, shuffle=False)
        else:
            test_loader = DataLoader(jet_data, batch_size=test_batch_size, shuffle=False)

    # Unspecified num_test and num_val -> Choose training data and then divide the rest in half into testing and validation datasets
    else:
        assert num_train <= len(jet_data), f"num_train = {num_train} is larger than the data size {len(jet_data)}!"

        # split into training, testing, and valid sets
        # split the rest in half
        num_test = int((len(jet_data) - num_train) / 2)
        num_val = len(jet_data) - num_train - num_test
        train_set, test_set, valid_set = torch.utils.data.random_split(jet_data,
                                                                       [num_train, num_test, num_val])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
        if test_batch_size is None:
            test_loader = DataLoader(jet_data, batch_size=batch_size, shuffle=False)
        else:
            test_loader = DataLoader(jet_data, batch_size=test_batch_size, shuffle=False)

    print('Data loaded')

    return train_loader, valid_loader, test_loader


def prepare_input(data, scale, cg_levels=True, device=None, dtype=None):
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

    node_ps = data['p4'].to(device=device, dtype=dtype) * scale

    data['p4'].requires_grad_(True)

    node_mask = data['node_mask'].to(device=device, dtype=torch.uint8)
    edge_mask = data['edge_mask'].to(device=device, dtype=torch.uint8)

    scalars = torch.ones_like(node_ps[:, :, 0]).unsqueeze(-1)
    scalars = normsq4(node_ps).abs().sqrt().unsqueeze(-1)

    if 'scalars' in data.keys():
        scalars = torch.cat([scalars, data['scalars'].to(device=device, dtype=dtype)], dim=-1)

    if not cg_levels:
        scalars = torch.stack(tuple(scalars for _ in range(
            scalars.shape[-1])), -2).to(device=device, dtype=dtype)

    return scalars, node_ps, node_mask, edge_mask


######################################## Data preprocessing ########################################
def load_pt_file(filename, path='./hls4ml/150p/'):
    path = path + filename
    print(f"loading {path}...")
    data = torch.load(path)
    return data


def cartesian(p_list):
    """
    [eta, phi, pt, tag] -> [E, px, py, pz, tag]
    """
    eta, phi, pt, tag = p_list
    pt /= 1000  # Convert to TeV
    if tag > 0:  # real data
        tag = 1
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        E = np.sqrt(2) * pt * np.cosh(eta)
    else:  # padded data
        tag = 0
        px = py = pz = E = 0
    return [E, px, py, pz, tag]


def convert_to_cartesian(jet_data, name_str, save=False):
    print(f"preprocessing {name_str}...")

    shape = list(jet_data.shape)
    shape[-1] += 1  # [eta, phi, pt, tag] -> [E, px, py, pz, tag]
    shape = tuple(shape)  # (_,_,4)

    p_cartesian_tag = np.zeros(shape)

    print("basis conversion...")
    for jet in range(len(jet_data)):
        for particle in range(len(jet_data[jet])):
            p_cartesian_tag[jet][particle] = cartesian(jet_data[jet][particle])

    p_cartesian = torch.from_numpy(p_cartesian_tag[:, :, :4])
    labels = torch.from_numpy(p_cartesian_tag[:, :, -1])
    Nobj = labels.sum(dim=-1)
    jet_data_cartesian = {'p4': p_cartesian, 'labels': labels, 'Nobj': Nobj}

    if save:
        print(f"saving {name_str}...")
        filename = name_str + '_cartesian.pt'
        path = './150p/cartesian/'
        if not osp.isdir(path):
            os.makedirs(path)
        path += filename
        torch.save(jet_data_cartesian, path)
        print(f"{name_str} saved as {path}")

    return jet_data_cartesian


if __name__ == "__main__":
    # data loading
    dir = '../hls4ml'
    # jet = ['g', 'q', 't', 'w', 'z']
    jet = ['g']

    for type in jet:
        polarrel_mask = load_pt_file(f'all_{type}_jets_30p_cartesian.pt', path=dir).numpy()
        convert_to_cartesian(polarrel_mask, f"{type}_jets_150p", save=True)
