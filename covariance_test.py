from args import get_bool, get_dtype, get_device
from lgn.models.autotest.lgn_tests import lgn_tests
from lgn.models.autotest.utils import plot_all_dev
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
    parser.add_argument('-j', '--jet-type', type=str, required=True, metavar='',
                        help="The jet type to train. Options: ('g', 'q', 't', 'w', 'z').")
    parser.add_argument('--file-path', type=str, default='hls4ml/', metavar='',
                        help='The path of the data.')
    parser.add_argument('--file-suffix', type=str, default='jets_30p_p4', metavar='',
                        help="The suffix of the file. Default: 'jets_30p_p4'")
    parser.add_argument('--unit', type=str, default='TeV',
                        help="The unit of momenta. Choices: ('GeV', 'TeV'). Default: TeV. ")
    parser.add_argument('-tbs', '--test-batch-size', type=int, default=4, metavar='',
                        help='The batch size for equivariance test.')
    parser.add_argument('--num-test-batch', type=int, default=-1, metavar='',
                        help='The number of batches used for equivariance test. For full test set, use -1.')
    parser.add_argument('--scale', type=float, default=1., metavar='',
                        help='The rescaling factor of the input 4-momenta. Default: 1.')

    # Model
    parser.add_argument('--model-path', default=None, type=str,
                        help='Directory of the trained model.')
    parser.add_argument('--load-epoch', type=int, default=-1, metavar='',
                        help='Epoch number of the trained model to load. -1 for loading weights in the lastest model.')

    parser.add_argument('--num-jet-particles', type=int, default=30, metavar='',
                        help='Number of particles per jet (batch) in the input. Default: 30 for the hls4ml 30-p data.')
    parser.add_argument('--tau-jet-scalars', type=int, default=1, metavar='',
                        help='Multiplicity of scalars per particle in a jet. Default: 1 for the hls4ml 150p data.')
    parser.add_argument('--tau-jet-vectors', type=int, default=1, metavar='',
                        help='Multiplicity of 4-vectors per particle in the jet. Default: 1 for the hls4ml 150p data.')

    parser.add_argument('--map-to-latent', type=str, default='mean', metavar='',
                        help="The method to map to latent space. Choice: ('sum', 'mix', 'mean')")
    parser.add_argument('--tau-latent-scalars', type=int, default=3, metavar='',
                        help='Multiplicity of scalars per particle in the latent space.')
    parser.add_argument('--tau-latent-vectors', type=int, default=2, metavar='',
                        help='Multiplicity of 4-vectors per particle the latent space.')

    parser.add_argument('--encoder-num-channels', nargs="+", type=int, default=[2, 3, 2, 3], metavar='',
                        help='Number of channels (multiplicity of all irreps) in each CG layer in the encoder.')
    parser.add_argument('--decoder-num-channels', nargs="+", type=int, default=[2, 3, 2, 3], metavar='',
                        help='Number of channels (multiplicity of all irreps) in each CG layer in the decoder.')

    parser.add_argument('--maxdim', nargs="+", type=int, default=[3], metavar='',
                        help='Maximum weights in the model (exclusive).')
    parser.add_argument('--num-basis-fn', type=int, default=10, metavar='',
                        help='Number of basis function to express edge features. Default: [10].')

    parser.add_argument('--weight-init', type=str, default='randn', metavar='',
                        help="Weight initialization distribution to use. Options: ['randn', 'rand']. Default: 'randn'.")
    parser.add_argument('--level-gain', nargs="+", type=float, default=[1.], metavar='',
                        help="Gain at each level. Default: [1.].")
    parser.add_argument('--activation', type=str, default='leakyrelu', metavar='',
                        help="Activation function used in MLP layers. Options: ['relu', 'elu', 'leakyrelu', 'sigmoid', 'logsigmoid']. Default: 'leakyrelu'.")
    parser.add_argument('--mlp', type=get_bool, default=True,
                        help='Whether to insert a perceptron acting on invariant scalars inside each CG level. Default: True')
    parser.add_argument('--mlp-depth', type=int, default=3, metavar='N',
                        help='Number of hidden layers in each MLP layer. Default: 3')
    parser.add_argument('--mlp-width', type=int, default=2, metavar='N',
                        help='Width of hidden layers in each MLP layer in units of the number of inputs. Default: 2')

    # Testing options
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
    return args


if __name__ == "__main__":
    import sys
    import logging
    torch.autograd.set_detect_anomaly(True)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    args = setup_argparse()
    if args.load_epoch < 0:
        args.load_epoch = latest_epoch(args.model_path, num=args.load_epoch)
    if args.model_path is None:
        raise ValueError('--model-path needs to be specified.')

    covariance_test(args)
