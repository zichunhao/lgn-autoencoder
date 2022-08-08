from argparse import Namespace
from typing import Optional, Tuple
from lgn.models import LGNEncoder, LGNDecoder
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from utils.data.dataset import JetDataset
from utils.utils import get_eps

import logging
import torch


def initialize_data(
    path: str, 
    batch_size: int, 
    train_fraction: float, 
    num_val: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    data = torch.load(path)

    jet_data = JetDataset(data, shuffle=True)  # The original data is not shuffled yet

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


def initialize_test_data(
    path: str, 
    batch_size: int
) -> DataLoader:
    data = torch.load(path)
    jet_data = JetDataset(data, shuffle=False)
    return DataLoader(jet_data, batch_size=batch_size, shuffle=False)


def initialize_autoencoder(
    args: Namespace,
    print_models: bool = True
) -> Tuple[LGNEncoder, LGNDecoder]:
    encoder = LGNEncoder(
        num_input_particles=args.num_jet_particles,
        tau_input_scalars=args.tau_jet_scalars,
        tau_input_vectors=args.tau_jet_vectors,
        map_to_latent=args.map_to_latent,
        tau_latent_scalars=args.tau_latent_scalars,
        tau_latent_vectors=args.tau_latent_vectors,
        maxdim=args.maxdim, max_zf=[1],
        num_channels=args.encoder_num_channels,
        weight_init=args.weight_init, level_gain=args.level_gain,
        num_basis_fn=args.num_basis_fn, activation=args.activation, scale=args.scale,
        mlp=args.mlp, mlp_depth=args.mlp_depth, mlp_width=args.mlp_width,
        device=args.device, dtype=args.dtype
    )
    logging.info(f"LGNEncoder initialized with {encoder.num_learnable_parameters} learnable parameters.")
    
    # multiplicity gets larger when concatenation is used
    mult = len(args.map_to_latent.split('&')) if '&' in args.map_to_latent else 1    
    tau_latent_scalars = args.tau_latent_scalars * mult
    tau_latent_vectors = args.tau_latent_vectors * mult
    
    decoder = LGNDecoder(
        tau_latent_scalars=tau_latent_scalars,
        tau_latent_vectors=tau_latent_vectors,
        num_output_particles=args.num_jet_particles,
        tau_output_scalars=args.tau_jet_scalars,
        tau_output_vectors=args.tau_jet_vectors,
        maxdim=args.maxdim, max_zf=[1],
        num_channels=args.decoder_num_channels,
        weight_init=args.weight_init, level_gain=args.level_gain,
        num_basis_fn=args.num_basis_fn, activation=args.activation,
        mlp=args.mlp, mlp_depth=args.mlp_depth, mlp_width=args.mlp_width,
        cg_dict=encoder.cg_dict, device=args.device, dtype=args.dtype
    )
    logging.info(f"LGNDecoder initialized with {encoder.num_learnable_parameters} learnable parameters.")
    
    if print_models:
        logging.info(f"{encoder=}")
        logging.info(f"{decoder=}")

    return encoder, decoder


def initialize_optimizers(
    args: Namespace, 
    encoder: LGNEncoder, 
    decoder: LGNDecoder
) -> Tuple[Optimizer, Optimizer]:
    if args.optimizer.lower() == 'adam':
        optimizer_encoder = torch.optim.Adam(encoder.parameters(), args.lr)
        optimizer_decoder = torch.optim.Adam(decoder.parameters(), args.lr)
    elif args.optimizer.lower() == 'rmsprop':
        optimizer_encoder = torch.optim.RMSprop(encoder.parameters(), lr=args.lr, eps=get_eps(args), momentum=0.9)
        optimizer_decoder = torch.optim.RMSprop(decoder.parameters(), lr=args.lr, eps=get_eps(args), momentum=0.9)
    else:
        raise NotImplementedError(
            f"""
            Other choices of optimizer are not implemented. 
            Available choices are 'Adam' and 'RMSprop'. Found: {args.optimizer}.
            """
        )
    return optimizer_encoder, optimizer_decoder
