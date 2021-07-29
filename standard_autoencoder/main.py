import torch
import os.path as osp
from utils.utils import create_model_folder, eps
from args import setup_argparse
from utils.make_data import initialize_data
from utils.train import train_loop
from models.encoder import Encoder
from models.decoder import Decoder

import logging
logging.basicConfig(level=logging.INFO)


def main(args):
    logging.info(args)

    # Loading data and initializing models
    train_data_path = osp.join(args.file_path, f"{args.jet_type}_{args.file_suffix}.pt")

    train_loader, valid_loader = initialize_data(path=train_data_path,
                                                 batch_size=args.batch_size,
                                                 train_fraction=args.train_fraction,
                                                 num_val=args.num_valid)

    encoder, decoder = initialize_models(args)

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
            "The encoder and decoder are not trained on the same device!")
    # Both on cpu
    else:
        logging.info('The models are initialized on CPU...')

    logging.info(f'Training over {args.num_epochs} epochs...')

    '''Training'''
    # Load existing model
    if args.load_to_train:
        outpath = args.load_path
        encoder.load_state_dict(torch.load(osp.join(outpath, f'weights_encoder/epoch_{args.load_epoch}_encoder_weights.pth'),
                                           map_location=args.device))
        decoder.load_state_dict(torch.load(osp.join(outpath, f'weights_decoder/epoch_{args.load_epoch}_decoder_weights.pth'),
                                           map_location=args.device))
    # Create new model
    else:
        outpath = create_model_folder(args)

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
                      dtype=args.dtype, device=args.device)

    decoder = Decoder(num_nodes=args.num_jet_particles,
                      latent_node_size=args.latent_node_size,
                      output_node_size=args.vec_dims,
                      node_sizes=args.decoder_node_sizes,
                      edge_sizes=args.decoder_edge_sizes,
                      num_mps=args.decoder_num_mps,
                      dropout=args.decoder_dropout,
                      alphas=args.decoder_alphas,
                      batch_norm=args.decoder_batch_norm,
                      dtype=args.dtype, device=args.device)

    logging.info(f"{encoder = }")
    logging.info(f"{decoder = }")

    return encoder, decoder


if __name__ == '__main__':
    args = setup_argparse()
    main(args)