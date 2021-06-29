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


def initialize_data(path, batch_size, train_fraction):
    data = torch.load(path)

    jet_data = JetDataset(data, shuffle=True)  # The original data is not shuffled yet

    if (train_fraction > 1) or (train_fraction < 0):
        train_fraction = 0.8

    num_jets = len(data['Nobj'])
    num_train = int(num_jets * train_fraction)
    num_val = num_jets - num_train
    print(f'{num_jets = }')
    print(f'{num_train = }')
    print(f'{num_val = }')
    print(f'{num_train + num_val = }')

    # split into training and validation set
    train_set, test_set = torch.utils.data.random_split(jet_data, [num_train, num_val])
    train_loader = DataLoader(jet_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(jet_data, batch_size=batch_size, shuffle=True)

    print('Data loaded')

    return train_loader, valid_loader


def initialize_test_data(path, batch_size):
    data = torch.load(path)
    jet_data = JetDataset(data, shuffle=False)
    return DataLoader(jet_data, batch_size=batch_size, shuffle=True)


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


def convert_to_cartesian(jet_data, save_path, file_name, test_fraction=0.2, save=False):
    print(f"preprocessing {file_name}...")

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
    num_jets = len(Nobj)
    num_test = int(num_jets * test_fraction)

    jet_data_cartesian = {'p4': p_cartesian, 'labels': labels, 'Nobj': Nobj}
    jet_data_cartesian_test = {key: val[:num_test] for key, val in jet_data_cartesian.items()}
    jet_data_cartesian_train = {key: val[num_test:] for key, val in jet_data_cartesian.items()}

    if save:
        print(f"saving {file_name}...")
        if not osp.isdir(save_path):
            os.makedirs(save_path)
        torch.save(jet_data_cartesian, osp.join(save_path, f'{file_name}_full.pt'))
        torch.save(jet_data_cartesian_train, osp.join(save_path, f'{file_name}.pt'))
        torch.save(jet_data_cartesian_test, osp.join(save_path, f'{file_name}_test.pt'))
        print(f"{file_name} saved as {save_path}")

    return jet_data_cartesian


if __name__ == "__main__":
    # data loading
    dir = '../hls4ml'
    # jet = ['g', 'q', 't', 'w', 'z']
    jet = ['g']

    for type in jet:
        polarrel_mask = load_pt_file(f'all_{type}_jets_30p_cartesian.pt', path=dir).numpy()
        convert_to_cartesian(jet_data=polarrel_mask, save_path=dir, file_name=f"{type}_jets_150p", save=True)
