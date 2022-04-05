import torch
import os.path as osp
from utils.utils import create_model_folder, eps, latest_epoch
from args import setup_argparse
from utils.make_data import initialize_data
from utils.train import train_loop
from models import Encoder
from models import Decoder

import logging
logging.basicConfig(level=logging.INFO)

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def main(args):
    if args.load_to_train and args.load_epoch < 0:
        args.load_epoch = latest_epoch(args.load_path, num=args.load_epoch)
        
    if args.patience <= 0:
        import math
        args.patience = math.inf

    logging.info(args)
    
    # Loading data and initializing models
    train_loader, valid_loader = initialize_data(path=args.file_path,
                                                 batch_size=args.batch_size,
                                                 vec_dims=args.vec_dims,
                                                 train_fraction=args.train_fraction,
                                                 num_val=args.num_valid)

    encoder, decoder, outpath = initialize_models(args)

    # trainings
    if args.optimizer.lower() == 'adam':
        optimizer_encoder = torch.optim.Adam(encoder.parameters(), args.lr)
        optimizer_decoder = torch.optim.Adam(decoder.parameters(), args.lr)
    elif args.optimizer.lower() == 'rmsprop':
        optimizer_encoder = torch.optim.RMSprop(encoder.parameters(), lr=args.lr, eps=eps(args), momentum=0.9)
        optimizer_decoder = torch.optim.RMSprop(decoder.parameters(), lr=args.lr, eps=eps(args), momentum=0.9)
    else:
        raise NotImplementedError("Other choices of optimizer are not implemented. "
                                  f"Available choices are 'Adam' and 'RMSprop'. Found: {args.optimizer}.")

    # Both on gpu
    if (next(encoder.parameters()).is_cuda and next(encoder.parameters()).is_cuda):
        logging.info('The models are initialized on GPU...')
    # One on cpu and the other on gpu
    elif (next(encoder.parameters()).is_cuda or next(encoder.parameters()).is_cuda):
        raise AssertionError(
            "The encoder and decoder are not trained on the same device!"
        )
    # Both on cpu
    else:
        logging.info('The models are initialized on CPU...')

    logging.info(f'Training over {args.num_epochs} epochs...')

    '''Training'''
    train_loop(args, train_loader, valid_loader, encoder, decoder,
               optimizer_encoder, optimizer_decoder, outpath, args.device)


def initialize_models(args):
    encoder = Encoder(num_nodes=args.num_jet_particles,
                      input_node_size=args.vec_dims,
                      latent_node_size=args.latent_node_size,
                      node_sizes=args.encoder_node_sizes,
                      edge_sizes=args.encoder_edge_sizes,
                      num_mps=args.encoder_num_mps,
                      dropout=args.encoder_dropout,
                      alphas=args.encoder_alphas,
                      batch_norm=args.encoder_batch_norm,
                      latent_map=args.latent_map,
                      dtype=args.dtype, device=args.device)

    decoder = Decoder(num_nodes=args.num_jet_particles,
                      latent_node_size=args.latent_node_size,
                      output_node_size=args.vec_dims,
                      node_sizes=args.decoder_node_sizes,
                      edge_sizes=args.decoder_edge_sizes,
                      num_mps=args.decoder_num_mps,
                      dropout=args.decoder_dropout,
                      alphas=args.decoder_alphas,
                      latent_map=args.latent_map,
                      normalize_output=args.normalized,
                      batch_norm=args.decoder_batch_norm,
                      dtype=args.dtype, device=args.device)

    if args.load_to_train:
        outpath = args.load_path
        encoder.load_state_dict(torch.load(
            osp.join(outpath, f'weights_encoder/epoch_{args.load_epoch}_encoder_weights.pth'),
            map_location=args.device
        ))
        decoder.load_state_dict(torch.load(
            osp.join(outpath, f'weights_decoder/epoch_{args.load_epoch}_decoder_weights.pth'),
            map_location=args.device
        ))
    # Create new model
    else:
        import json
        outpath = create_model_folder(args)
        args_dir = osp.join(outpath, "args_cache.json")
        with open(args_dir, "w") as f:
            json.dump({k: str(v) for k, v in vars(args).items()}, f)

    logging.info(f"{encoder=}")
    logging.info(f"{decoder=}")
    logging.info(f'Latent space size: {encoder.latent_space_size}')

    return encoder, decoder, outpath


if __name__ == '__main__':
    import sys
    torch.autograd.set_detect_anomaly(True)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    args = setup_argparse()
    main(args)
