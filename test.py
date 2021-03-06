import os.path as osp
import argparse
import logging
import torch
from utils.argparse_utils import get_bool, get_device, get_dtype
from utils.argparse_utils import parse_model_settings, parse_plot_settings, parse_covariance_test_settings, parse_data_settings
from utils.jet_analysis.plot import plot_p
from lgn.models.autotest.lgn_tests import lgn_tests
from lgn.models.autotest.utils import plot_all_dev
from utils.initialize import initialize_autoencoder, initialize_test_data
from utils.utils import make_dir, latest_epoch
from utils.train import validate


def test(args):
    test_loader = initialize_test_data(path=args.test_data_path, batch_size=args.test_batch_size)

    # Load models
    encoder, decoder = initialize_autoencoder(args)

    encoder_path = osp.join(args.model_path, f'weights_encoder/epoch_{args.load_epoch}_encoder_weights.pth')
    decoder_path = osp.join(args.model_path, f'weights_decoder/epoch_{args.load_epoch}_decoder_weights.pth')
    encoder.load_state_dict(torch.load(encoder_path, map_location=args.device))
    decoder.load_state_dict(torch.load(decoder_path, map_location=args.device))
    
    if args.plot_only:
        test_path = osp.join(args.model_path, f'test_{args.jet_type}_jets')
        try:
            # we do not need to load latent space or normalization factors
            recons = torch.load(osp.join(test_path, 'reconstructed.pt')).to(args.device)
            target = torch.load(osp.join(test_path, 'target.pt')).to(args.device)
        except FileNotFoundError:
            logging.warning('Inference results not found. Run inference first.')
            recons, target, latent, norm_factors = validate(
                args, test_loader, encoder, decoder, args.load_epoch,
                args.model_path, args.device, for_test=True
            )
            test_path = make_dir(osp.join(args.model_path, f'test_{args.jet_type}_jets'))
            torch.save(target, osp.join(test_path, 'target.pt'))
            torch.save(recons, osp.join(test_path, 'reconstructed.pt'))
            torch.save(latent, osp.join(test_path, 'latent.pt'))
            torch.save(norm_factors, osp.join(test_path, 'norm_factors.pt'))
            logging.info(f'Data saved exported to {test_path}.')
    else:
        recons, target, latent, norm_factors = validate(
            args, test_loader, encoder, decoder, args.load_epoch,
            args.model_path, args.device, for_test=True
        )
        test_path = make_dir(osp.join(args.model_path, f'test_{args.jet_type}_jets'))
        torch.save(target, osp.join(test_path, 'target.pt'))
        torch.save(recons, osp.join(test_path, 'reconstructed.pt'))
        torch.save(latent, osp.join(test_path, 'latent.pt'))
        torch.save(norm_factors, osp.join(test_path, 'norm_factors.pt'))
        logging.info(f'Data saved exported to {test_path}.')

    fig_path = make_dir(osp.join(test_path, 'jet_plots'))
    if args.abs_coord and (args.unit.lower() == 'tev'):
        # Convert to GeV for plotting
        recons *= 1000
        target *= 1000

    jet_images_same_norm, jet_images = plot_p(args, target, recons, fig_path)
    torch.save(jet_images_same_norm, osp.join(test_path, 'jet_images_same_norm.pt'))
    torch.save(jet_images, osp.join(test_path, 'jet_images.pt'))
    logging.info('Plots finished.')

    if args.equivariance_test:
        dev = lgn_tests(args, encoder, decoder, test_loader, alpha_max=args.alpha_max, theta_max=args.theta_max,
                        cg_dict=encoder.cg_dict, unit=args.unit)
        plot_all_dev(dev, osp.join(test_path, 'equivariance_tests'))


def setup_argparse():
    parser = argparse.ArgumentParser(description='LGN Autoencoder on Test Dataset')

    # Data
    parse_data_settings(parser)

    # Model
    parse_model_settings(parser)

    # Test
    parser.add_argument('--plot-only', action='store_true', default=False,
                        help='Only plot the results without the inference. If inference results are not found, run inference first.')
    parser.add_argument('--loss-choice', type=str, default='ChamferLoss', metavar='',
                        help="Choice of loss function. Options: ('ChamferLoss', 'EMDLoss', 'hybrid')")
    parser.add_argument('--loss-norm-choice', type=str, default='p3', metavar='',
                        help="Choice of calculating the norms of 4-vectors when calculating the loss. "
                        "Options: ['canonical', 'real', 'cplx', 'p3', 'polar']. "
                        "'canonical': Write p in the basis of zonal functions, take the dot product, and find the norm out of the complex scalar. "
                        "'real': Find the norm of each component and then take the dot product. "
                        "'cplx': Take the dot product and then find the norm out the the complex scalar. "
                        "'p3': Find the norm of each component and find the norm square of the 3-momenta part of p4. "
                        "'polar': Find the distance of real component in polar coordinate: ??pT^2 + ??phi^2 + ??eta^2"
                        "Default: 'p3.'")
    parser.add_argument('--chamfer-jet-features', type=get_bool, default=True,
                        help="Whether to take into the jet features.")
    parser.add_argument('--im', type=get_bool, default=True,
                        help="Whether to take into imaginary component of the reconstructed jet into account if using the chamfer loss."
                        "Only used when --loss-norm-choice is in ['p3', 'polar']"
                        "If set to True, the target will be complexified with 0 imaginary components.")
    parser.add_argument('--device', type=get_device, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), metavar='',
                        help="Device to which the model is initialized. Options: ('gpu', 'cpu', 'cuda', '-1')."
                        "Default: -1, which means deciding device based on whether gpu is available.")
    parser.add_argument('--dtype', type=get_dtype, default=torch.float64, metavar='',
                        help="Data type to which the model is initialized. Options: ('float', 'float64', 'double'). Default: float64")

    # Load models
    parser.add_argument('--model-path', type=str, default=None, metavar='',
                        help='Path of the trained model to load and test.')
    parser.add_argument('--load-epoch', type=int, default=-1, metavar='',
                        help='Epoch number of the trained model to load. -1 for loading weights in the lastest model.')

    # Plots
    parse_plot_settings(parser)

    # Convariance tests
    parse_covariance_test_settings(parser)

    args = parser.parse_args()

    if args.load_epoch < 0:
        args.load_epoch = latest_epoch(args.model_path, num=args.load_epoch)
    if args.model_path is None:
        raise ValueError('--model-path needs to be specified.')

    return args


if __name__ == '__main__':
    import sys
    torch.autograd.set_detect_anomaly(True)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    args = setup_argparse()
    logging.info(f'{args=}')
    test(args)
