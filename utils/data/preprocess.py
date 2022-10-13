import argparse
import logging
from pathlib import Path
from typing import Union
import jetnet
import numpy as np
import torch

def prepare(
    jet_type: str, 
    save_dir: Union[str, Path],
    test_portion: float = 0.2
):
    logging.info(f"Downloading data ({jet_type=}) from JetNet.")
    data = jetnet.datasets.JetNet(
        jet_type=jet_type,
        data_dir=save_dir / "hdf5"
    )
    logging.info(f"Preparing data ({jet_type=}).")
    
    if isinstance(save_dir, Path):
        pass
    elif isinstance(save_dir, str):
        save_dir = Path(save_dir)
    else:
        raise TypeError(
            "save_path must be of type a str or pathlib.Path. "
            f"Got: {type(save_dir)}."
        )
    save_dir.mkdir(parents=True, exist_ok=True)
    
    jet = data.jet_data
    p = data.particle_data
    
    # jet momenta components
    Pt, Eta, Mass = jet[..., 1], jet[..., 2], jet[..., 3]
    Phi = np.random.random(Eta.shape) * 2 * np.pi  # [0, 2pi]
    
    # particle momenta components (relative coordinates)
    eta_rel, phi_rel, pt_rel, mask = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    
    # particle momenta components (polar coordinates)
    pt = pt_rel * Pt.reshape(-1, 1)
    eta = eta_rel + Eta.reshape(-1, 1)
    phi = phi_rel + Phi.reshape(-1, 1)
    phi = ((phi + np.pi) % (2 * np.pi)) - np.pi  # [-pi, pi]
    mask = torch.from_numpy(mask)
    
    # Cartesian coordinates
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    m = np.random.random(eta.shape) * 1e-3  # O(1e-4 GeV)
    p0 = np.sqrt((pt * np.cosh(eta))**2 + m**2)
    
    p4 = torch.from_numpy(np.stack([p0, px, py, pz], axis=-1))
    p4 = p4 * mask.unsqueeze(-1) / 1000  # tev
    
    p4_data = {
        'p4': p4,
        'labels': mask,
        'Nobj': mask.sum(dim=-1)
    }
    
    torch.save(p4_data, save_dir / f"{jet_type}_jets_30p_p4_all.pt")
    
    # training-test split
    split_idx = int(len(data) * (1 - test_portion))
    torch.save(
        {k: v[:split_idx] for k, v in p4_data.items()}, 
        save_dir / f"{jet_type}_jets_30p_p4.pt"
    )
    torch.save(
        {k: v[split_idx:] for k, v in p4_data.items()}, 
        save_dir / f"{jet_type}_jets_30p_p4_test.pt"
    )
    logging.info(
        f"Data saved in {save_dir} as {jet_type}_jets_30p_p4_all.pt, {jet_type}_jets_30p_p4.pt, {jet_type}_jets_30p_p4_test.pt."
    )
    
    return

if __name__ == "__main__":
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    # args
    parser = argparse.ArgumentParser(
        description='Prepare dataset for LGN Autoencoder'
    )
    parser.add_argument(
        '-j', '--jet_types',
        nargs='+', type=str, default=['g', 'q', 't', 'w', 'z'],
        help='List of jet types to download and preprocess.'
    )
    parser.add_argument(
        '-s', '--save-dir',
        type=str, required=True,
        help='Directory to save preprocessed data.'
    )
    parser.add_argument(
        '-t', '--test-portion',
        type=float, default=0.2,
        help="Test portion of the data."
    )
    args = parser.parse_args()
    logging.info(f"{args=}")
    for jet_type in args.jet_types:
        prepare(
            jet_type=jet_type,
            save_dir=Path(args.save_dir),
            test_portion=args.test_portion
        )
