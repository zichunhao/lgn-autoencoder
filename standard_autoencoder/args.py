import argparse
import numpy as np
import torch


def setup_argparse():
    parser = argparse.ArgumentParser(description='HLS4ML Graph autoencoder options')

    ######################################### Data options #########################################
    parser.add_argument('-j', '--jet-type', type=str, required=True, metavar='',
                        help="Jet type to train. Options: ('g', 'q', 't', 'w', 'z').")
    parser.add_argument('--file-path', type=str, default='../hls4ml/', metavar='',
                        help='Path of the data.')
    parser.add_argument('--file-suffix', type=str, default='jets_30p_p4', metavar='',
                        help="The suffix of the file. Default: 'jets_30p_p4'")
    parser.add_argument('--train-fraction', type=float, default=10,
                        help='The fraction (or number) of data used for training.')
    parser.add_argument('--num-valid', type=int, default=10,
                        help='The number of data used for validation. Used only if train-fraction is int (for test runs).')

    ######################################## Model options ########################################
    parser.add_argument('--num-jet-particles', type=int, default=30, metavar='',
                        help='Number of particles per jet (batch) in the input. Default: 30 for the hls4ml 30p data.')
    parser.add_argument('--vec-dims', type=int, default=4, metavar='',
                        help='Dimension of vectors. Default: 4 for 4-vectors.')
    parser.add_argument('--latent-node-size', type=int, default=20, metavar='',
                        help='Dimension of latent vectors (shape: N x 1).')

    # encoder
    parser.add_argument('--encoder-edge-sizes', type=int, nargs='+', action='append',
                        default=[[32, 128, 64, 16]], metavar='',
                        help='Edge convolution layer width in each message passing step in encoder.')
    parser.add_argument('--encoder-node-sizes', type=int, nargs='+', action='append',
                        default=[[16], [32], [8]], metavar='',
                        help='Edge convolution layer width in each message passing step in encoder.')
    parser.add_argument('--encoder-num-mps', type=int, default=3, metavar='',
                        help='Number of message passing steps in encoder.')
    parser.add_argument('--encoder-dropout', type=float, default=0.2, metavar='',
                        help='Dropout value for edge features in encoder.')
    parser.add_argument('--encoder-alphas', type=float, default=0.2, metavar='',
                        help='Alpha values for the leaky relu layers in encoder.')
    parser.add_argument('--encoder-batch-norm', type=get_bool, default=True, metavar='',
                        help='Whether to include batch normalizations in encoder. Default: True.')
    parser.add_argument('--encoder-metric', type=get_bool, default=True, metavar='',
                        help="The metric for distance in encoder. Options: ('minkoskian', 'cartesian'). Default: 'cartesian'.")

    # decoder
    parser.add_argument('--decoder-edge-sizes', type=int, nargs='+', action='append',
                        default=[[32, 128, 64, 16]], metavar='',
                        help='Edge convolution layer width in each message passing step in decoder.')
    parser.add_argument('--decoder-node-sizes', type=int, nargs='+', action='append',
                        default=[[16], [32], [8]], metavar='',
                        help='Edge convolution layer width in each message passing step in decoder.')
    parser.add_argument('--decoder-num-mps', type=int, default=3, metavar='',
                        help='Number of message passing steps in decoder.')
    parser.add_argument('--decoder-dropout', type=float, default=0.1, metavar='',
                        help='Dropout value for edge features in decoder.')
    parser.add_argument('--decoder-alphas', type=float, default=0.2, metavar='',
                        help='Alpha value for the leaky relu layers in decoder.')
    parser.add_argument('--decoder-batch-norm', type=get_bool, default=True, metavar='',
                        help='Whether to include batch normalizations in decoder. Default: True.')
    parser.add_argument('--decoder-metric', type=get_bool, default=True, metavar='',
                        help="The metric for distance in decoder. Options: ('minkoskian', 'cartesian'). Default: 'cartesian'.")

    ####################################### Training options #######################################
    parser.add_argument('--device', type=get_device, default=get_device('-1'), metavar='',
                        help="Device to which the model is initialized. Options: ('gpu', 'cpu', 'cuda', '-1')."
                        "Default: -1, which means deciding device based on whether gpu is available.")
    parser.add_argument('--dtype', type=get_dtype, default=torch.float, metavar='',
                        help="Data type to which the model is initialized. Options: ('float', 'float64', 'double'). Default: torch.float")
    parser.add_argument('--lr', type=float, default=1e-5, metavar='',
                        help='Learning rate of the backpropagation.')
    parser.add_argument('--optimizer', type=str, default="adam", metavar='',
                        help="The optimizer to use. Options:('adam', 'rmprop') Default: 'adam'")
    parser.add_argument('-bs', '--batch-size', type=int, default=16, metavar='',
                        help='Batch size.')
    parser.add_argument('-e', '--num-epochs', type=int, default=64, metavar='',
                        help='Number of epochs for training.')
    parser.add_argument('--loss-choice', type=str, default='ChamferLoss', metavar='',
                        help="Choice of loss function. Options: ('ChamferLoss', 'EMDLoss', 'hybrid')")
    parser.add_argument('--loss-norm-choice', type=str, default='cartesian', metavar='',
                        help="Choice of calculating the norms of 4-vectors when calculating the loss. "
                        "Options: ['cartesian', 'minkowskian', 'polar']. "
                        "'cartesian': (+, +, +, +). \n"
                        "'minkowskian': (+, -, -, -) \n"
                        "'polar': convert to (E, pt, eta, phi) paired with metric (+, +, +, +) \n"
                        "Default: 'cartesian.'")
    parser.add_argument('--chamfer-jet-features', type=get_bool, default=True,
                        help="Whether to take into the jet features.")

    parser.add_argument('--save-dir', type=str, default='standard-autoencoder-trained-models', metavar='',
                        help='The directory to save trained models and figures.')
    parser.add_argument('--save-freq', type=int, default=500, metavar='',
                        help='How frequent the model weights are saved in each epoch. Default: 500.')
    parser.add_argument('--custom-suffix', type=str, default=None, metavar='',
                        help='Custom suffix of the saving directory.')

    # Loading existing models
    parser.add_argument('--load-to-train', default=False, action='store_true',
                        help='Whether to load existing (trained) model for training.')
    parser.add_argument('--load-path', type=str, default=None, metavar='',
                        help='Path of the trained model to load.')
    parser.add_argument('--load-epoch', type=int, default=None, metavar='',
                        help='Epoch number of the trained model to load.')

    ################################### Model evaluation options ###################################
    parser.add_argument('--unit', type=str, default='TeV',
                        help="The unit of momenta. Choices: ('GeV', 'TeV'). Default: TeV. ")
    parser.add_argument('--polar-max', nargs="+", type=float, default=[200, 2, np.pi], metavar='',
                        help='List of maximum values of (pt, eta, phi) in the histogram. Default: [200, np.pi, 2].')
    parser.add_argument('--cartesian-max', nargs="+", type=float, default=[100, 100, 100], metavar='',
                        help='List of maximum values of (px, py, pz) in the histogram. Default: [100, 100, 100].')
    parser.add_argument('--jet-polar-max', nargs="+", type=float, default=[250, 4000, 2, np.pi], metavar='',
                        help='List of maximum values of jet features (m, pt, eta, phi) in the histogram. Default: [200, 4000, 2, np.pi].')
    parser.add_argument('--jet-cartesian-max', nargs="+", type=float, default=[250, 2000, 2000, 4000], metavar='',
                        help='List of maximum values of jet features (m, px, py, pz) in the histogram. Default: [200, 2000, 2000, 4000].')
    parser.add_argument('--num_bins', type=int, default=81, metavar='',
                        help='Number of bins in the histogram Default: 81.')
    parser.add_argument('--cutoff', type=float, default=1e-7, metavar='',
                        help='Cutoff value of (3-)momenta magnitude to be included in the historgram. Default: 1e-7.')
    parser.add_argument('--fill', default=False, action='store_true',
                        help='Whether to plot filled histograms as well. True only if called in the command line.')

    args = parser.parse_args()

    return args


def get_bool(arg):
    """
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
    if arg is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if arg.lower() == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def get_dtype(arg):
    if arg is None:
        return torch.float64

    if arg.lower() == 'float':
        dtype = torch.float
    elif arg.lower() == 'double':
        dtype = torch.double
    else:
        dtype = torch.float64
    return dtype
