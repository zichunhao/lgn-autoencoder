import os.path as osp
import sys
sys.path.insert(1, 'lgn/')
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

import torch
import matplotlib.pyplot as plt

from args import setup_argparse
from utils.make_data import initialize_data
from utils.utils import create_model_folder
from utils.train import train_loop

from lgn.models.lgn_encoder import LGNEncoder
from lgn.models.lgn_decoder import LGNDecoder

torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    args = setup_argparse()

    data_path = osp.join(args.file_path, f"{args.jet_type}_{args.file_suffix}")

    train_loader, valid_loader, test_loader = initialize_data(path=data_path,
                                                              batch_size=args.batch_size,
                                                              num_train=args.num_train,
                                                              num_val=args.num_val,
                                                              num_test=args.num_test)

    """Initializations"""
    encoder = LGNEncoder(num_input_particles=args.num_jet_particles,
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
                         device=args.device, dtype=args.dtype)
    decoder = LGNDecoder(tau_latent_scalars=args.tau_latent_scalars,
                         tau_latent_vectors=args.tau_latent_vectors,
                         num_output_particles=args.num_jet_particles,
                         tau_output_scalars=args.tau_jet_scalars,
                         tau_output_vectors=args.tau_jet_vectors,
                         maxdim=args.maxdim, max_zf=[1],
                         num_channels=args.decoder_num_channels,
                         weight_init=args.weight_init, level_gain=args.level_gain,
                         num_basis_fn=args.num_basis_fn, activation=args.activation,
                         mlp=args.mlp, mlp_depth=args.mlp_depth, mlp_width=args.mlp_width,
                         device=args.device, dtype=args.dtype)

    optimizer_encoder = torch.optim.Adam(encoder.parameters(), args.lr)
    optimizer_decoder = torch.optim.Adam(decoder.parameters(), args.lr)

    # Both on gpu
    if (next(encoder.parameters()).is_cuda and next(encoder.parameters()).is_cuda):
        print('The models are initialized on GPU...')
    # One on cpu and the other on gpu
    elif (next(encoder.parameters()).is_cuda or next(encoder.parameters()).is_cuda):
        raise AssertionError("The encoder and decoder are not trained on the same device!")
    # Both on cpu
    else:
        print('The models are initialized on CPU...')

    print(f'Training over {args.num_epochs} epochs...')

    '''Training'''
    # Load existing model
    if args.load_to_train:
        outpath = args.load_path
        encoder.load_state_dict(torch.load(osp.join(outpath, f'weights_encoder/epoch_{args.load_epoch}_encoder_weights.pth'), map_location=args.device))
        decoder.load_state_dict(torch.load(osp.join(outpath, f'weights_decoder/epoch_{args.load_epoch}_encoder_weights.pth'), map_location=args.device))
    # Create new model
    else:
        outpath = create_model_folder(args)

    outpath = create_model_folder(args)
    train_loop(args, train_loader, valid_loader, encoder, decoder, optimizer_encoder, optimizer_decoder, outpath, args.device)

    print("Training completed!")
