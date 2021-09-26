from lgn.models.autotest.lgn_tests import lgn_tests
from lgn.models.autotest.utils import plot_all_dev

from utils.argparse_utils import get_device, get_dtype, parse_model_settings, parse_data_settings
from utils.initialize import initialize_autoencoder, initialize_test_data
from utils.utils import latest_epoch

import argparse
import torch
import numpy as np
import os.path as osp


def covariance_test(args):
    # Data
    test_data_path = osp.join(args.file_path, f"{args.jet_type}_{args.file_suffix}_test.pt")
    test_loader = initialize_test_data(path=test_data_path, batch_size=args.test_batch_size)

    # Load models
    encoder, decoder = initialize_autoencoder(args)

    encoder_path = osp.join(args.model_path, f'weights_encoder/epoch_{args.load_epoch}_encoder_weights.pth')
    decoder_path = osp.join(args.model_path, f'weights_decoder/epoch_{args.load_epoch}_decoder_weights.pth')
    encoder.load_state_dict(torch.load(encoder_path, map_location=args.device))
    decoder.load_state_dict(torch.load(decoder_path, map_location=args.device))

    # Evaluations
    dev = lgn_tests(args, encoder, decoder, test_loader, alpha_max=args.alpha_max, theta_max=args.theta_max,
                    cg_dict=encoder.cg_dict, unit=args.unit)
    plot_all_dev(dev, osp.join(args.model_path, 'model_evaluations/equivariance_tests'))


def setup_argparse():
    parser = argparse.ArgumentParser(description='LGN Model Convariance Test Options')

    # Test data
    parse_data_settings(parser, training=False)

    parse_model_settings(parser)

    # Load models
    parser.add_argument('--model-path', type=str, default=None, metavar='',
                        help='Path of the trained model to load and test.')
    parser.add_argument('--load-epoch', type=int, default=-1, metavar='',
                        help='Epoch number of the trained model to load. -1 for loading weights in the lastest model.')

    # Covariance test options
    parser.add_argument('--device', type=get_device, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), metavar='',
                        help="Device to for testing. Options: ('gpu', 'cpu', 'cuda', '-1')."
                        "Default: -1, which means deciding device based on whether gpu is available.")
    parser.add_argument('--dtype', type=get_dtype, default=torch.float64, metavar='',
                        help="Data type to which the model is initialized. Options: ('float', 'float64', 'double'). Default: float64")
    parser.add_argument('--alpha-max', type=float, default=10., metavar='',
                        help='The maximum alpha value of equivariance test, where gamma = cosh(alpha).'
                        'Default: 10., at which gamma = 11013.2.')
    parser.add_argument('--theta-max', type=float, default=2*np.pi, metavar='',
                        help='The maximum theta value of equivariance test.'
                        'Default: 2 pi.')

    args = parser.parse_args()

    if args.load_epoch < 0:
        args.load_epoch = latest_epoch(args.model_path, num=args.load_epoch)
    if args.model_path is None:
        raise ValueError('--model-path needs to be specified.')

    return args


if __name__ == "__main__":
    import sys
    import logging
    torch.autograd.set_detect_anomaly(True)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    args = setup_argparse()
    logging.info(f'{args=}')
    covariance_test(args)
