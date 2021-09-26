import argparse
import numpy as np
import torch


def get_bool(arg):
    """Parse boolean from input string.
    Adapted from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(arg, bool):
        return arg
    if arg.lower() in ('true', 't', '1'):
        return True
    elif arg.lower() in ('false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected. "
                                         "Options: For True, anything in ('true', 't', '1') can be used. "
                                         "For False, anything in ('false', 'f', '0') can be used. "
                                         "Cases are ignored.")


def get_device(arg):
    """Parse torch.device from input string"""
    if arg is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if arg.lower() == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def get_dtype(arg):
    """Parse torch.dtype from input string"""
    if arg is None:
        return torch.float64

    if arg.lower() == 'float':
        dtype = torch.float
    elif arg.lower() == 'double':
        dtype = torch.double
    else:
        dtype = torch.float64
    return dtype


def parse_model_settings(parser):
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
    return parser


def parse_plot_settings(parser):
    parser.add_argument('--plot-freq', type=int, default=10, metavar='',
                        help='How frequent to plot. Used when --loss-choice is not EMD. Default: 10.')
    parser.add_argument('--cutoff', type=float, default=1e-7, metavar='',
                        help='Cutoff value of (3-)momenta magnitude to be included in the historgram. Default: 1e-7.')
    parser.add_argument('--fill', default=False, action='store_true',
                        help='Whether to plot filled histograms as well. True only if called in the command line.')

    parser.add_argument('--jet-image-npix', type=int, default=24,
                        help='The number of pixels for the jet image')
    parser.add_argument('--jet-image-vmin', type=float, default=1e-10,
                        help='vmin for LogNorm')
    parser.add_argument('--num-jet-images', type=int, default=15,
                        help='Number of one-to-one jet images to plot.')
    return parser


def parse_covariance_test_settings(parser):
    parser.add_argument('--equivariance-test', default=False, action='store_true',
                        help='Whether to take the equivariance test after all trainings on the last model. True only when it is called.'
                        'Default: False.')
    parser.add_argument('--equivariance-test-only', default=False, action='store_true',
                        help='Whether to take the equivariance test only (i.e. no training).')
    parser.add_argument('--test-device', type=get_device, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), metavar='',
                        help="Device to for testing. Options: ('gpu', 'cpu', 'cuda', '-1')."
                        "Default: -1, which means deciding device based on whether gpu is available.")
    parser.add_argument('--alpha-max', type=float, default=10., metavar='',
                        help='The maximum alpha value of equivariance test, where gamma = cosh(alpha).'
                        'Default: 10., at which gamma = 11013.2.')
    parser.add_argument('--theta-max', type=float, default=2*np.pi, metavar='',
                        help='The maximum theta value of equivariance test.'
                        'Default: 2 pi.')
    return parser


def parse_data_settings(parser, training=True):
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
    parser.add_argument('--abs-coord', type=get_bool, default=True, metavar='',
                        help='Whether the data is in absolute coordinates. False when relative coordinates are used.')
    parser.add_argument('--scale', type=float, default=1., metavar='',
                        help='The rescaling factor of the input 4-momenta. Default: 1.')
    parser.add_argument('--num-test-batch', type=int, default=-1, metavar='',
                        help='The number of batches used for equivariance test. For full test set, use -1.')

    if training:
        parser.add_argument('--train-fraction', type=float, default=0.8,
                            help='The fraction (or number) of data used for training.')
        parser.add_argument('--num-valid', type=int, default=None,
                            help='The number of data used for validation. Used only if train-fraction is greater than 1.'
                            'Useful for test runs.')

    return parser
