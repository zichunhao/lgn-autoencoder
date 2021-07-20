import argparse
import torch
import numpy as np


def setup_argparse():
    parser = argparse.ArgumentParser(description='LGN Autoencoder Options')

    ######################################### Data options #########################################
    parser.add_argument('-j', '--jet-type', type=str, required=True, metavar='',
                        help="The jet type to train. Options: ('g', 'q', 't', 'w', 'z').")
    parser.add_argument('--file-path', type=str, default='hls4ml/', metavar='',
                        help='The path of the data.')
    parser.add_argument('--file-suffix', type=str, default='jets_30p_p4', metavar='',
                        help="The suffix of the file. Default: 'jets_30p_p4'")
    parser.add_argument('--train-fraction', type=float, default=0.8,
                        help='The fraction (or number) of data used for training.')
    parser.add_argument('--num-valid', type=int, default=None,
                        help='The number of data used for validation. Used only if train-fraction is greater than 1.'
                        'Useful for test runs.')
    parser.add_argument('--scale', type=float, default=1., metavar='',
                        help='The rescaling factor of the input 4-momenta. Default: 1.')

    ######################################## Model options ########################################
    parser.add_argument('--num-jet-particles', type=int, default=30, metavar='',
                        help='Number of particles per jet (batch) in the input. Default: 30 for the hls4ml 30-p data.')
    parser.add_argument('--tau-jet-scalars', type=int, default=1, metavar='',
                        help='Multiplicity of scalars per particle in a jet. Default: 1 for the hls4ml 150p data.')
    parser.add_argument('--tau-jet-vectors', type=int, default=1, metavar='',
                        help='Multiplicity of 4-vectors per particle in the jet. Default: 1 for the hls4ml 150p data.')
    parser.add_argument('--jet-features', type=get_bool, default=False,
                        help='Whether to incorporate jet features in the message passing step. Default: True')

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

    ####################################### Training options #######################################
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
    parser.add_argument('--loss-choice', type=str, default='ChamferLoss', metavar='',
                        help="Choice of loss function. Options: ('ChamferLoss', 'EMDLoss', 'hybrid')")
    # chamfer loss options
    parser.add_argument('--chamfer-loss-weight', type=float, default=1.0, metavar='',
                        help="The weight for the chamfer loss, only relevant if loss-choice is 'hybrid'. Default: 1.0.")
    parser.add_argument('--loss-norm-choice', type=str, default='p3', metavar='',
                        help="Choice of calculating the norms of 4-vectors when calculating the loss. "
                        "Options: ['canonical', 'real', 'cplx', 'p3', 'polar']. "
                        "'canonical': Write p in the basis of zonal functions, take the dot product, and find the norm out of the complex scalar. "
                        "'real': Find the norm of each component and then take the dot product. "
                        "'cplx': Take the dot product and then find the norm out the the complex scalar. "
                        "'p3': Find the norm of each component and find the norm square of the 3-momenta part of p4. "
                        "'polar': Find the distance of real component in polar coordinate: ΔpT^2 + Δphi^2 + Δeta^2"
                        "Default: 'p3.'")
    parser.add_argument('--chamfer-jet-features', type=get_bool, default=True,
                        help="Whether to take into the jet features.")
    parser.add_argument('--im', type=get_bool, default=True,
                        help="Whether to take into imaginary component of the reconstructed jet into account if using the chamfer loss."
                        "Only used when --loss-norm-choice is in ['p3', 'polar']"
                        "If set to True, the target will be complexified with 0 imaginary components.")

    parser.add_argument('--save-dir', type=str, default='autoencoder-trained-models', metavar='',
                        help='The directory to save trained models and figures.')
    parser.add_argument('--custom-suffix', type=str, default=None, metavar='',
                        help='Custom suffix of the saving directory.')
    parser.add_argument('--save-freq', type=int, default=500, metavar='',
                        help='How frequent the model weights are saved in each epoch. Default: 500.')

    # Loading existing models
    parser.add_argument('--load-to-train', default=False, action='store_true',
                        help='Whether to load existing (trained) model for training.')
    parser.add_argument('--load-path', type=str, default=None, metavar='',
                        help='Path of the trained model to load.')
    parser.add_argument('--load-epoch', type=int, default=-1, metavar='',
                        help='Epoch number of the trained model to load. -1 for loading weights in the lastest model.')

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

    ################################## Equivariance test options ##################################
    parser.add_argument('--equivariance-test', default=False, action='store_true',
                        help='Whether to take the equivariance test after all trainings on the last model. True only when it is called.'
                        'Default: False.')
    parser.add_argument('--equivariance-test-only', default=False, action='store_true',
                        help='Whether to take the equivariance test only (i.e. no training).')
    parser.add_argument('-tbs', '--test-batch-size', type=int, default=4, metavar='',
                        help='The batch size for equivariance test.')
    parser.add_argument('--num-test-batch', type=int, default=-1, metavar='',
                        help='The number of batches used for equivariance test. For full test set, use -1.')
    parser.add_argument('--test-device', type=get_device, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), metavar='',
                        help="Device to for testing. Options: ('gpu', 'cpu', 'cuda', '-1')."
                        "Default: -1, which means deciding device based on whether gpu is available.")
    parser.add_argument('--alpha-max', type=float, default=10., metavar='',
                        help='The maximum alpha value of equivariance test, where gamma = cosh(alpha).'
                        'Default: 10., at which gamma = 11013.2.')
    parser.add_argument('--theta-max', type=float, default=2*np.pi, metavar='',
                        help='The maximum theta value of equivariance test.'
                        'Default: 2 pi.')

    args = parser.parse_args()

    if args.load_to_train and ((args.load_path is None) or (args.load_epoch is None)):
        raise ValueError("--load-to-train requires --load-model-path and --load-epoch.")

    return args


####################################### Parsing from strings #######################################
# Adapted from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def get_bool(arg):
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
