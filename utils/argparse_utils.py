import argparse
from email.policy import default
import numpy as np
import torch


def get_bool(arg: argparse.Namespace) -> bool:
    """Parse boolean from input string.
    Adapted from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(arg, bool):
        return arg
    if arg.lower() in ("true", "t", "1"):
        return True
    elif arg.lower() in ("false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(
            "Boolean value expected. "
            "Options: For True, anything in ('true', 't', '1') can be used. "
            "For False, anything in ('false', 'f', '0') can be used. "
            "Cases are ignored."
        )


def get_device(arg: argparse.Namespace) -> torch.dtype:
    """Parse torch.device from input string"""
    if arg is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if arg.lower() == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def get_dtype(arg: argparse.Namespace) -> torch.dtype:
    """Parse torch.dtype from input string"""
    if arg is None:
        return torch.float64

    if arg.lower() == "float":
        dtype = torch.float
    elif arg.lower() == "double":
        dtype = torch.double
    else:
        dtype = torch.float64
    return dtype


def parse_model_settings(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--num-jet-particles",
        type=int,
        default=30,
        metavar="",
        help="Number of particles per jet (batch) in the input. Default: 30 for the JetNet data.",
    )
    parser.add_argument(
        "--tau-jet-scalars",
        type=int,
        default=1,
        metavar="",
        help="Multiplicity of scalars per particle in a jet. Default: 1 for the JetNet data.",
    )
    parser.add_argument(
        "--tau-jet-vectors",
        type=int,
        default=1,
        metavar="",
        help="Multiplicity of 4-vectors per particle in the jet. Default: 1 for the JetNet data.",
    )

    parser.add_argument(
        "--map-to-latent",
        type=str,
        default="min&max",
        metavar="",
        help="The method to map to latent space. "
        "Choice: ('sum', 'mix', 'mean', 'min', 'max') "
        "or any combinations with '+' (for addition) or '&' (for concatenation).",
    )
    parser.add_argument(
        "--tau-latent-scalars",
        type=int,
        default=1,
        metavar="",
        help="Multiplicity of scalars per particle in the latent space.",
    )
    parser.add_argument(
        "--tau-latent-vectors",
        type=int,
        default=1,
        metavar="",
        help="Multiplicity of 4-vectors per particle the latent space.",
    )
    parser.add_argument(
        "--jet-features",
        default=False,
        action="store_true",
        help="Include jet momentum as an additional vector and jet invariant mass as an additional scalar to the input.",
    )

    parser.add_argument(
        "--encoder-num-channels",
        nargs="+",
        type=int,
        default=[3, 3, 4, 4],
        metavar="",
        help="Number of channels (multiplicity of all irreps) in each CG layer in the encoder.",
    )
    parser.add_argument(
        "--decoder-num-channels",
        nargs="+",
        type=int,
        default=[4, 4, 3, 3],
        metavar="",
        help="Number of channels (multiplicity of all irreps) in each CG layer in the decoder.",
    )

    parser.add_argument(
        "--maxdim",
        nargs="+",
        type=int,
        default=[2],
        metavar="",
        help="Maximum weights in the model (exclusive).",
    )
    parser.add_argument(
        "--num-basis-fn",
        type=int,
        default=10,
        metavar="",
        help="Number of basis function to express edge features.",
    )

    parser.add_argument(
        "--weight-init",
        type=str,
        default="randn",
        metavar="",
        help="Weight initialization distribution to use. Options: ['randn', 'rand']. Default: 'randn'.",
    )
    parser.add_argument(
        "--level-gain",
        nargs="+",
        type=float,
        default=[1.0],
        metavar="",
        help="Gain at each level. Default: [1.].",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="leakyrelu",
        metavar="",
        help="Activation function used in MLP layers. Options: ['relu', 'elu', 'leakyrelu', 'sigmoid', 'logsigmoid'].",
    )
    parser.add_argument(
        "--mlp",
        type=get_bool,
        default=True,
        help="Whether to insert a perceptron acting on invariant scalars inside each CG level.",
    )
    parser.add_argument(
        "--mlp-depth",
        type=int,
        default=6,
        metavar="N",
        help="Number of hidden layers in each MLP layer.",
    )
    parser.add_argument(
        "--mlp-width",
        type=int,
        default=6,
        metavar="N",
        help="Width of hidden layers in each MLP layer in units of the number of inputs.",
    )
    return parser


def parse_plot_settings(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--plot-freq",
        type=int,
        default=10,
        metavar="",
        help="How frequent to plot. Used when --loss-choice is not EMD. Default: 10.",
    )
    parser.add_argument(
        "--plot-start-epoch",
        type=int,
        default=50,
        metavar="",
        help="The epoch to start plotting. Default: 50.",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=1e-7,
        metavar="",
        help="Cutoff value of (3-)momenta magnitude to be included in the historgram. Default: 1e-7.",
    )
    parser.add_argument(
        "--fill",
        default=False,
        action="store_true",
        help="Whether to plot filled histograms as well. True only if called in the command line.",
    )

    parser.add_argument(
        "--jet-image-npix",
        type=int,
        default=40,
        help="The number of pixels for the jet image",
    )
    parser.add_argument(
        "--jet-image-maxR", type=float, default=0.5, help="The maxR of the jet image"
    )
    parser.add_argument(
        "--jet-image-vmin", type=float, default=1e-10, help="vmin for LogNorm"
    )
    parser.add_argument(
        "--num-jet-images",
        type=int,
        default=15,
        help="Number of one-to-one jet images to plot.",
    )

    _parse_particle_recons_err_settings(parser)
    _parse_jet_recons_err_settings(parser)

    return parser


def _parse_particle_recons_err_settings(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--custom-particle-recons-ranges",
        default=False,
        action="store_true",
        help="Whether to manually set the ranges of particle reconstruction errors when plotting the histograms. "
        "Call --custom-particle-recons-ranges to set true.",
    )
    parser.add_argument(
        "--particle-rel-err-min-cartesian",
        nargs="+",
        type=float,
        default=[-1, -1, -1],
        metavar="",
        help="xmin of histogram for particle reconstruction relative errors in Cartesian coordinates.",
    )
    parser.add_argument(
        "--particle-rel-err-max-cartesian",
        nargs="+",
        type=float,
        default=[1, 1, 1],
        metavar="",
        help="xmax of histogram for particle reconstruction relative errors in Cartesian coordinates.",
    )
    parser.add_argument(
        "--particle-padded-recons-min-cartesian",
        nargs="+",
        type=float,
        default=[-100, -100, -100],
        metavar="",
        help="xmin of histogram for reconstructed padded particless in Cartesian coordinates.",
    )
    parser.add_argument(
        "--particle-padded-recons-max-cartesian",
        nargs="+",
        type=float,
        default=[100, 100, 100],
        metavar="",
        help="xmax of histogram for reconstructed padded particless in Cartesian coordinates.",
    )

    parser.add_argument(
        "--particle-rel-err-min-polar",
        nargs="+",
        type=float,
        default=[-1, -1, -1],
        metavar="",
        help="xmin of histogram for particle reconstruction relative errors in polar coordinates (pt, eta, phi).",
    )
    parser.add_argument(
        "--particle-rel-err-max-polar",
        nargs="+",
        type=float,
        default=[1, 1, 1],
        metavar="",
        help="xmax of histogram for particle reconstruction relative errors in polar coordinates (pt, eta, phi).",
    )
    parser.add_argument(
        "--particle-padded-recons-min-polar",
        nargs="+",
        type=float,
        default=[-100, -1, -np.pi],
        metavar="",
        help="xmin of histogram for reconstructed padded particless in polar coordinates (pt, eta, phi).",
    )
    parser.add_argument(
        "--particle-padded-recons-max-polar",
        nargs="+",
        type=float,
        default=[100, 1, np.pi],
        metavar="",
        help="xmax of histogram for reconstructed padded particless in polar coordinates.",
    )
    return parser


def _parse_jet_recons_err_settings(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--custom-jet-recons-ranges",
        default=False,
        action="store_true",
        help="Whether to manually set the ranges of jet reconstruction errors when plotting the histograms. "
        "Call --custom-jet-recons-ranges to set true.",
    )
    parser.add_argument(
        "--jet-rel-err-min-cartesian",
        nargs="+",
        type=float,
        default=[-1, -1, -1, -1],
        metavar="",
        help="xmin of histogram for jet reconstruction relative errors in Cartesian coordinates.",
    )
    parser.add_argument(
        "--jet-rel-err-max-cartesian",
        nargs="+",
        type=float,
        default=[1, 1, 1, 1],
        metavar="",
        help="xmax of histogram for jet reconstruction relative errors in Cartesian coordinates.",
    )
    parser.add_argument(
        "--jet-rel-err-min-polar",
        nargs="+",
        type=float,
        default=[-1, -1, -1, -1],
        metavar="",
        help="xmin of histogram for jet reconstruction relative errors in polar coordinates (pt, eta, phi).",
    )
    parser.add_argument(
        "--jet-rel-err-max-polar",
        nargs="+",
        type=float,
        default=[1, 1, 1, 1],
        metavar="",
        help="xmax of histogram for jet reconstruction relative errors in polar coordinates (pt, eta, phi).",
    )
    return parser


def parse_covariance_test_settings(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--equivariance-test",
        default=False,
        action="store_true",
        help="Whether to take the equivariance test after all trainings on the last model. True only when it is called."
        "Default: False.",
    )
    parser.add_argument(
        "--test-device",
        type=get_device,
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        metavar="",
        help="Device to for testing. Options: ('gpu', 'cpu', 'cuda', '-1')."
        "Default: -1, which means deciding device based on whether gpu is available.",
    )
    parser.add_argument(
        "--alpha-max",
        type=float,
        default=10.0,
        metavar="",
        help="The maximum alpha value of equivariance test, where gamma = cosh(alpha)."
        "Default: 10., at which gamma = 11013.2.",
    )
    parser.add_argument(
        "--theta-max",
        type=float,
        default=2 * np.pi,
        metavar="",
        help="The maximum theta value of equivariance test." "Default: 2 pi.",
    )
    return parser


def parse_data_settings(
    parser: argparse.ArgumentParser, training: bool = True
) -> argparse.ArgumentParser:
    parser.add_argument(
        "-j",
        "--jet-type",
        type=str,
        required=True,
        metavar="",
        help="The jet type to train. Options: ('g', 'q', 't', 'w', 'z', 'QCD').",
    )
    parser.add_argument(
        "--data-paths",
        nargs="+",
        type=str,
        default=["./data/g_jets_30p_p4.pt", "./data/q_jets_30p_p4.pt"],
        metavar="",
        help="The paths of the training data.",
    )
    parser.add_argument(
        "--train-set-portion",
        type=float,
        default=-1,
        metavar="",
        help="The portion of the training-validation set to be used as training set. "
        "Default: -1, which means using all the data as training set."
        "If in the range (0, 1], it will be used as the portion of the training set. "
        "If in the range (1, infinity), it will be used as the number of events in the training set. ",
    )
    parser.add_argument(
        "--test-data-paths",
        nargs="+",
        type=str,
        default=["./data/g_jets_30p_p4_test.pt", "./data/q_jets_30p_p4_test.pt"],
        metavar="",
        help="The paths of the test data.",
    )
    parser.add_argument(
        "--unit",
        type=str,
        default="TeV",
        help="The unit of momenta. Choices: ('GeV', 'TeV'). Default: TeV. ",
    )
    parser.add_argument(
        "-tbs",
        "--test-batch-size",
        type=int,
        default=4,
        metavar="",
        help="The batch size for equivariance test.",
    )
    parser.add_argument(
        "--abs-coord",
        type=get_bool,
        default=True,
        metavar="",
        help="Whether the data is in absolute coordinates. False when relative coordinates are used.",
    )
    parser.add_argument(
        "--polar-coord",
        type=get_bool,
        default=False,
        metavar="",
        help="Whether the data is in polar coordinates (pt, eta, phi). False when Cartesian coordinates are used.",
    )
    parser.add_argument(
        "--normalize",
        type=get_bool,
        default=True,
        metavar="",
        help="Whether to normalize the features before passing into the NN.",
    )
    parser.add_argument(
        "--normalize-method",
        type=str,
        default="overall_max",
        metavar="",
        help='Method to normalize the features. Choices: ("overall_max", "component_max", "jet_E"). Default: "overall_max". '
        "overall_max: normalize by the overall maximum of all features within a jet. "
        "component_max: normalize each component by its maximum (absolute value) within a jet. "
        "jet_E: normalize each component by the jet energy.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        metavar="",
        help="The rescaling factor of the input 4-momenta. Default: 1.",
    )
    parser.add_argument(
        "--num-test-batch",
        type=int,
        default=-1,
        metavar="",
        help="The number of batches used for equivariance test. For full test set, use -1.",
    )

    if training:
        parser.add_argument(
            "--train-fraction",
            type=float,
            default=0.8,
            help="The fraction (or number) of data used for training.",
        )
        parser.add_argument(
            "--num-valid",
            type=int,
            default=None,
            help="The number of data used for validation. Used only if train-fraction is greater than 1."
            "Useful for test runs.",
        )

    return parser
