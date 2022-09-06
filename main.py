from utils.argparse_utils import get_bool, get_device, get_dtype
from utils.argparse_utils import parse_model_settings, parse_plot_settings, parse_covariance_test_settings, parse_data_settings
from utils.utils import create_model_folder, best_epoch, get_compression_rate
from utils.train import train_loop
from utils.initialize import initialize_autoencoder, initialize_data, initialize_test_data, initialize_optimizers

from lgn.models.autotest.lgn_tests import lgn_tests
from lgn.models.autotest.utils import plot_all_dev

import torch
import os.path as osp
import logging
import argparse


def main(args):
    if args.load_to_train and args.load_epoch < 0:
        args.load_epoch = best_epoch(args.load_path, num=args.load_epoch)
    logging.info(f'{args=}')
    compression_rate = get_compression_rate(
        ls=args.tau_latent_scalars,
        lv=args.tau_latent_vectors,
        map_to_latent=args.map_to_latent,
        vec_dim=4,
        num_particles=args.num_jet_particles
    )
    logging.info(f"compression rate: {compression_rate}")

    train_loader, valid_loader = initialize_data(path=args.data_path,
                                                 batch_size=args.batch_size,
                                                 train_fraction=args.train_fraction,
                                                 num_val=args.num_valid)
    test_loader = initialize_test_data(path=args.test_data_path, batch_size=args.test_batch_size)

    """Initializations"""
    encoder, decoder = initialize_autoencoder(args)
    optimizer_encoder, optimizer_decoder = initialize_optimizers(args, encoder, decoder)

    # Both on gpu
    if (next(encoder.parameters()).is_cuda and next(encoder.parameters()).is_cuda):
        logging.info('The models are initialized on GPU...')
    # One on cpu and the other on gpu
    elif (next(encoder.parameters()).is_cuda or next(encoder.parameters()).is_cuda):
        raise RuntimeError("The encoder and decoder are not trained on the same device!")
    # Both on cpu
    else:
        logging.info('The models are initialized on CPU...')

    '''Training'''
    logging.info(f'Training over {args.num_epochs} epochs...')
    # Load existing model
    if args.load_to_train:
        outpath = args.load_path
        try:
            encoder.load_state_dict(torch.load(osp.join(outpath, f'weights_encoder/best_encoder_weights.pth'),
                                            map_location=args.device))
            decoder.load_state_dict(torch.load(osp.join(outpath, f'weights_decoder/best_decoder_weights.pth'),
                                            map_location=args.device))
        except FileNotFoundError:
            encoder.load_state_dict(torch.load(osp.join(outpath, f'weights_encoder/epoch_{args.load_epoch}_encoder_weights.pth'),
                                            map_location=args.device))
            decoder.load_state_dict(torch.load(osp.join(outpath, f'weights_decoder/epoch_{args.load_epoch}_decoder_weights.pth'),
                                            map_location=args.device))
    # Create new model
    else:
        import json
        outpath = create_model_folder(args)
        args_dir = osp.join(outpath, "args_cache.json")
        with open(args_dir, "w") as f:
            json.dump({k: str(v) for k, v in vars(args).items()}, f)
            
    if args.equivariance_test:
        dev = lgn_tests(args, encoder, decoder, test_loader, alpha_max=args.alpha_max, theta_max=args.theta_max,
                        cg_dict=encoder.cg_dict, unit=args.unit)
        plot_all_dev(dev, osp.join(outpath, 'model_evaluations/equivariance_tests/initial'))

    best_epoch = train_loop(
        args, train_loader, valid_loader, 
        encoder, decoder, 
        optimizer_encoder, optimizer_decoder, 
        outpath, args.device
    )
    logging.info(f"Training completed! Best epoch: {best_epoch}")

    # Equivariance tests
    if args.equivariance_test:
        try:
            encoder.load_state_dict(torch.load(osp.join(outpath, 'weights_encoder/best_encoder_weights.pth'),
                                               map_location=args.test_device))
            decoder.load_state_dict(torch.load(osp.join(outpath, 'weights_decoder/best_encoder_weights.pth'),
                                               map_location=args.test_device))
        except FileNotFoundError:
            encoder.load_state_dict(torch.load(osp.join(outpath, f'weights_encoder/epoch_{best_epoch}_encoder_weights.pth'),
                                               map_location=args.test_device))
            decoder.load_state_dict(torch.load(osp.join(outpath, f'weights_decoder/epoch_{best_epoch}_decoder_weights.pth'),
                                               map_location=args.test_device))

        dev = lgn_tests(args, encoder, decoder, test_loader, alpha_max=args.alpha_max, theta_max=args.theta_max,
                        cg_dict=encoder.cg_dict, unit=args.unit)
        plot_all_dev(dev, osp.join(outpath, 'model_evaluations/equivariance_tests'))

    if args.test_best_model:
        args.load_epoch = best_epoch
        args.model_path = outpath
        args.equivariance_test = False  # Already ran on the best model
        logging.info(f'Running test on model from epoch {best_epoch}...')
        from test import test
        test(args)

    logging.info("Done!")


def setup_argparse():
    parser = argparse.ArgumentParser(description='LGN Autoencoder Options')

    # Data
    parse_data_settings(parser, training=True)

    parse_model_settings(parser)

    # Training
    parser.add_argument('--device', type=get_device, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), metavar='',
                        help="Device to which the model is initialized. Options: ('gpu', 'cpu', 'cuda', '-1')."
                        "Default: -1, which means deciding device based on whether gpu is available.")
    parser.add_argument('--dtype', type=get_dtype, default=torch.float64, metavar='',
                        help="Data type to which the model is initialized. Options: ('float', 'float64', 'double'). Default: float64")
    parser.add_argument('--lr', type=float, default=1e-5, metavar='',
                        help='Learning rate of the backpropagation.')
    parser.add_argument('--optimizer', type=str, default="adam", metavar='',
                        help="The optimizer to use. Options: ('adam', 'rmsprop') Default: 'adam'")
    parser.add_argument('-bs', '--batch-size', type=int, default=16, metavar='',
                        help='Batch size.')
    parser.add_argument('-e', '--num-epochs', type=int, default=64, metavar='',
                        help='Number of epochs for training.')
    parser.add_argument('-p', '--patience', type=int, default=-1, metavar='',
                        help='Patience for early stopping. Use -1 for no early stopping.')
    parser.add_argument('--loss-choice', type=str, default='ChamferLoss', metavar='',
                        help="Choice of loss function. Options: ('ChamferLoss', 'EMDLoss', 'hybrid')")
    # chamfer loss options
    parser.add_argument('--chamfer-loss-weight', type=float, default=1.0, metavar='',
                        help="The weight for the chamfer loss, only relevant if loss-choice is 'hybrid'. Default: 1.0.")
    parser.add_argument('--chamfer-loss-norm-choice', type=str, default='p3', metavar='',
                        help="Choice of calculating the norms of 4-vectors when calculating the loss. "
                        "Options: ['canonical', 'real', 'cplx', 'p3', 'polar']. "
                        "'canonical': Write p in the basis of zonal functions, take the dot product, and find the norm out of the complex scalar. "
                        "'real': Find the norm of each component and then take the dot product. "
                        "'cplx': Take the dot product and then find the norm out the the complex scalar. "
                        "'p3': Find the norm of each component and find the norm square of the 3-momenta part of p4. "
                        "'polar': Find the distance of real component in polar coordinate: ΔpT^2 + Δphi^2 + Δeta^2"
                        "Default: 'p3.'")
    parser.add_argument('--chamfer-jet-features', type=get_bool, default=False,
                        help="Whether to take into the jet features.")
    parser.add_argument('--chamfer-im', type=get_bool, default=False,
                        help="Whether to take into imaginary component of the reconstructed jet into account if using the chamfer loss."
                        "Only used when --loss-norm-choice is in ['p3', 'polar']"
                        "If set to True, the target will be complexified with 0 imaginary components.")

    # Hungarian loss options
    parser.add_argument('--hungarian-abs-coord', type=get_bool, default=True,
                        help="Whether to use absolute coordinate when calculating the Hungarian MSE loss.")
    parser.add_argument('--hungarian-polar-coord', type=get_bool, default=False,
                        help="Whether to use polar coordinate when calculating the Hungarian MSE loss.")

    parser.add_argument('--save-dir', type=str, default='autoencoder-trained-models', metavar='',
                        help='The directory to save trained models and figures.')
    parser.add_argument('--custom-suffix', type=str, default=None, metavar='',
                        help='Custom suffix of the saving directory.')
    parser.add_argument('--save-freq', type=int, default=500, metavar='',
                        help='How frequent the model weights are saved in each epoch (EMD only). Default: 500.')

    # regularizations
    parser.add_argument('--l1-lambda', type=float, default=1e-8, metavar='',
                        help='penalty for L1 regularization.')
    parser.add_argument('--l2-lambda', type=float, default=0, metavar='',
                        help='penalty for L2 regularization.')

    # Loading existing models
    parser.add_argument('--load-to-train', default=False, action='store_true',
                        help='Whether to load existing (trained) model for training.')
    parser.add_argument('--load-path', type=str, default=None, metavar='',
                        help='Path of the trained model to load.')
    parser.add_argument('--load-epoch', type=int, default=-1, metavar='',
                        help='Epoch number of the trained model to load. '
                        'Set to -1 for loading weights in the latest model.')

    parse_plot_settings(parser)
    parse_covariance_test_settings(parser)

    parser.add_argument('--test-best-model', default=False, action='store_true',
                        help='Whether to test the best model with the test dataset.')

    args = parser.parse_args()

    if args.load_to_train and ((args.load_path is None) or (args.load_epoch is None)):
        raise ValueError("--load-to-train requires --load-model-path and --load-epoch.")
    if args.patience < 0:
        import math
        args.patience = math.inf

    return args


if __name__ == "__main__":
    import sys
    torch.autograd.set_detect_anomaly(True)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    args = setup_argparse()
    main(args)
