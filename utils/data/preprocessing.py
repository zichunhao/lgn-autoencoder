import torch
import numpy as np
import os
import os.path as osp


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
        E = pt * np.cosh(eta)
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
        os.makedirs(save_path, exist_ok=True)
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
