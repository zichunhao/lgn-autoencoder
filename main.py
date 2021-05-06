import os.path as osp
import sys
sys.path.insert(1, 'lgn/')
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

import torch
import matplotlib.pyplot as plt

from args import setup_argparse
from utils.make_pytorch_data import initialize_data
from lgn.models.autotest import lgn_tests
from utils.utils import create_model_folder

from lgn.models.lgn_encoder import LGNEncoder
from lgn.models.lgn_decoder import LGNDecoder

if __name__ == "__main__":
    args = setup_argparse()

    data_path = osp.join(args.file_path, f"{args.jet_type}_{args.file_suffix}")

    train_loader, test_loader, valid_loader = initialize_data(path=data_path,
                                                              batch_size=args.batch_size,
                                                              num_train=args.num_train,
                                                              num_val=args.num_val,
                                                              num_test=args.num_test)

    encoder = LGNEncoder(num_input_particles=args.num_jet_particles,
                         tau_input_scalars=args.tau_jet_scalars,
                         tau_input_vectors=args.tau_jet_vectors,
                         num_latent_particles=args.num_latent_particles,
                         tau_latent_scalars=args.tau_latent_scalars,
                         tau_latent_vectors=args.tau_latent_vectors,
                         maxdim=args.maxdim, max_zf=args.max_zf,
                         num_channels=args.encoder_num_channels,
                         weight_init=args.weight_init, level_gain=args.level_gain,
                         num_basis_fn=args.num_basis_fn, activation=args.activation, scale=args.scale,
                         mlp=args.mlp, mlp_depth=args.mlp_depth, mlp_width=args.mlp_width,
                         device=args.device, dtype=args.dtype)
    decoder = LGNDecoder(num_latent_particles=args.num_latent_particles,
                         tau_latent_scalars=args.tau_latent_scalars,
                         tau_latent_vectors=args.tau_latent_vectors,
                         num_output_particles=args.num_jet_particles,
                         tau_output_scalars=args.tau_jet_scalars,
                         tau_output_vectors=args.tau_jet_vectors,
                         maxdim=args.maxdim, max_zf=args.max_zf,
                         num_channels=args.decoder_num_channels,
                         weight_init=args.weight_init, level_gain=args.level_gain,
                         num_basis_fn=args.num_basis_fn, activation=args.activation,
                         mlp=args.mlp, mlp_depth=args.mlp_depth, mlp_width=args.mlp_width,
                         device=args.device, dtype=args.dtype)
