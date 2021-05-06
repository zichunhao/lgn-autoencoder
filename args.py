import argparse
import torch

def setup_argparse():
    parser = argparse.ArgumentParser(description='LGN Autoencoder Options')

    ######################################### Data options #########################################
    parser.add_argument('-j', '--jet-type', type=str, required=True, metavar='',
                        help="The jet type to train. Options: ('g', 'q', 't', 'w', 'z').")
    parser.add_argument('--file-path', type=str, default='hls4ml/cartesian', metavar='',
                        help='The path of the data.')
    parser.add_argument('--file-suffix', type=str, default='jets_150p_cartesian.pt', metavar='',
                        help="The suffix of the file. Default: 'jets_150p_cartesian.pt'")
    parser.add_argument('--num-train', type=int, default=5, metavar='',
                        help='Number of samples to train on. Default: 528000')
    parser.add_argument('--num-val', type=int, default=5, metavar='',
                        help='Number of samples to validate on. Default: -1.')
    parser.add_argument('--num-test', type=int, default=5, metavar='',
                        help='Number of samples to test eqvuivariance on. Default: -1.')
    parser.add_argument('--scale', type=float, default=1., metavar='',
                        help='The rescaling factor of the input 4-momenta. Default: 1.')


    ######################################## Model options ########################################
    parser.add_argument('--num-jet-particles', type=int, default=150, metavar='',
                        help='Number of particles per jet (batch) in the input. Default: 150 for the hls4ml 150p data.')
    parser.add_argument('--tau-jet-scalars', type=int, default=1, metavar='',
                        help='Multiplicity of scalars per particle in a jet. Default: 1 for the hls4ml 150p data.')
    parser.add_argument('--tau-jet-vectors', type=int, default=1, metavar='',
                        help='Multiplicity of 4-vectors per particle in the jet. Default: 1 for the hls4ml 150p data.')

    parser.add_argument('--num-latent-particles', type=int, default=150, metavar='',
                        help='Number of particles per jet in the latent space. Default: 150 for the hls4ml 150p data.')
    parser.add_argument('--tau-latent-scalars', type=int, default=3, metavar='',
                        help='Multiplicity of scalars per particle in the latent space.')
    parser.add_argument('--tau-latent-vectors', type=int, default=2, metavar='',
                        help='Multiplicity of 4-vectors per particle the latent space.')

    parser.add_argument('--encoder-num-channels', nargs="+", type=int, default=[2, 3, 2, 1], metavar='',
                        help='Number of channels (or multiplicity or all irreps) in each CG layer in the encoder.')
    parser.add_argument('--decoder-num-channels', nargs="+", type=int, default=[2, 3, 2, 1], metavar='',
                        help='Number of channels (or multiplicity or all irreps) in each CG layer in the decoder.')

    parser.add_argument('--maxdim', nargs="+", type=int, default=[2], metavar='',
                        help='Maximum weights in the model. Each element in maxdim will be capped to 1 because we only want 4-momentum, ' \
                        'i.e. the (1/2,1/2) representation, in the end, while irreps with different weights are mixed independently.')
    parser.add_argument('--max-zf', nargs="+", type=int, default=[2], metavar='',
                        help='Maximum dimensions of zonal functions. Default: [2].')
    parser.add_argument('--num-basis-fn', type=int, default=10, metavar='',
                        help='Number of basis function to express edge features. Default: [2].')

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
                        help="Device to which the model is initialized. Options: ('gpu', 'cpu', 'cuda', '-1')." \
                        "Default: -1, which means deciding device based on whether gpu is available.")
    parser.add_argument('--dtype', type=get_dtype, default=torch.float64, metavar='',
                        help="Data type to which the model is initialized. Options: ('float', 'float64', 'double'). Default: float64")
    parser.add_argument('--lr', type=float, default=1e-4, metavar='',
                        help='Learning rate of the backpropagation.')
    parser.add_argument('--batch-size', type=int, default=8, metavar='',
                        help='Batch size.')
    parser.add_argument('--num-epochs', type=int, default=64, metavar='',
                        help='Number of epochs for training.')
    parser.add_argument('--loss-norm-choice', type=str, default='canonical', metavar='',
                        help="Choice of calculating the norms of 4-vectors when calculating the loss. " \
                        "Options: ('canonical', 'real', 'cplx'). " \
                        "'canonical': Write p in the basis of zonal functions, take the dot product, and find the norm out of the complex scalar. " \
                        "'real': Find the norm of each component and then take the dot product. " \
                        "'cplx': Take the dot product and then find the norm out the the complex scalar. " \
                        "Default: 'canonical.'")

    parser.add_argument('--save-dir', type=str, default='autoencoder_trained_models', metavar='',
                        help='The directory to save trained models and figures.')
    parser.add_argument('--custom-suffix', type=str, default=None, metavar='',
                        help='Custom suffix of the saving directory.')

    # Loading existing models
    parser.add_argument('--load-to-train', type=get_bool, default=False, metavar='',
                        help='Whether to load existing (trained) model for training.')
    parser.add_argument('--load-path', type=str, default=None, metavar='',
                        help='Path of the trained model to load.')
    parser.add_argument('--load-epoch', type=int, default=None, metavar='',
                        help='Epoch number of the trained model to load.')

    args = parser.parse_args()

    if args.load_to_train and ((args.load_model_path is None) or (args.load_epoch is None)):
        parser.error("--load-to-train requires --load-model-path and --load-epoch.")

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
        raise argparse.ArgumentTypeError("Boolean value expected. "\
                                         "Options: For True, anything in ('true', 't', '1') can be used. "\
                                         "For False, anything in ('false', 'f', '0') can be used. "\
                                         "Cases are ignored.")


def get_device(arg):
    if arg is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if arg.lower() in ('gpu', 'cuda'):
        arg.device = torch.device('cuda')
    elif arg.lower() == 'cpu':
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
