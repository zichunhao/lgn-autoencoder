import os.path as osp
import argparse
import logging
from pathlib import Path
import torch
import numpy as np
from utils.argparse_utils import get_bool, get_device, get_dtype
from utils.argparse_utils import (
    parse_model_settings,
    parse_plot_settings,
    parse_covariance_test_settings,
    parse_data_settings,
)
from utils.jet_analysis import plot_p, get_ROC_AUC, anomaly_scores_sig_bkg
from lgn.models.autotest.lgn_tests import lgn_tests
from lgn.models.autotest.utils import plot_all_dev
from utils.initialize import initialize_autoencoder, initialize_test_data
from utils.utils import make_dir, best_epoch
from utils.train import validate


def test(args):
    test_loader = initialize_test_data(
        paths=args.test_data_paths, batch_size=args.test_batch_size
    )

    # Load models
    encoder, decoder = initialize_autoencoder(args)
    try:
        encoder_path = osp.join(
            args.model_path, "weights_encoder/best_encoder_weights.pth"
        )
        decoder_path = osp.join(
            args.model_path, "weights_decoder/best_decoder_weights.pth"
        )
        encoder.load_state_dict(torch.load(encoder_path, map_location=args.device))
        decoder.load_state_dict(torch.load(decoder_path, map_location=args.device))
    except FileNotFoundError:
        encoder_path = osp.join(
            args.model_path,
            f"weights_encoder/epoch_{args.load_epoch}_encoder_weights.pth",
        )
        decoder_path = osp.join(
            args.model_path,
            f"weights_decoder/epoch_{args.load_epoch}_decoder_weights.pth",
        )
        encoder.load_state_dict(torch.load(encoder_path, map_location=args.device))
        decoder.load_state_dict(torch.load(decoder_path, map_location=args.device))

    if args.plot_only:
        test_path = osp.join(
            args.model_path, f"test_{args.jet_type}_jets_{args.load_epoch}"
        )
        try:
            # we do not need to load latent space or normalization factors
            recons = torch.load(osp.join(test_path, "reconstructed.pt")).to(args.device)
            target = torch.load(osp.join(test_path, "target.pt")).to(args.device)
        except FileNotFoundError:
            logging.warning("Inference results not found. Run inference first.")
            recons, target, latent, norm_factors = validate(
                args,
                test_loader,
                encoder,
                decoder,
                args.load_epoch,
                args.model_path,
                args.device,
                for_test=True,
            )
            test_path = make_dir(
                osp.join(
                    args.model_path, f"test_{args.jet_type}_jets_{args.load_epoch}"
                )
            )
            torch.save(target, osp.join(test_path, "target.pt"))
            torch.save(recons, osp.join(test_path, "reconstructed.pt"))
            torch.save(latent, osp.join(test_path, "latent.pt"))
            torch.save(norm_factors, osp.join(test_path, "norm_factors.pt"))
            logging.info(f"Data saved exported to {test_path}.")
    else:
        recons, target, latent, norm_factors = validate(
            args,
            test_loader,
            encoder,
            decoder,
            args.load_epoch,
            args.model_path,
            args.device,
            for_test=True,
        )
        test_path = make_dir(
            osp.join(args.model_path, f"test_{args.jet_type}_jets_{args.load_epoch}")
        )
        torch.save(target, osp.join(test_path, "target.pt"))
        torch.save(recons, osp.join(test_path, "reconstructed.pt"))
        torch.save(latent, osp.join(test_path, "latent.pt"))
        torch.save(norm_factors, osp.join(test_path, "norm_factors.pt"))
        logging.info(f"Data saved exported to {test_path}.")

    fig_path = make_dir(osp.join(test_path, "jet_plots"))
    if args.abs_coord and (args.unit.lower() == "tev"):
        # Convert to GeV for plotting
        recons *= 1000
        target *= 1000

    jet_images_same_norm, jet_images = plot_p(args, target, recons, fig_path)
    torch.save(jet_images_same_norm, osp.join(test_path, "jet_images_same_norm.pt"))
    torch.save(jet_images, osp.join(test_path, "jet_images.pt"))
    logging.info("Plots finished.")
    
    # Lorentz group equivariance tests
    if args.equivariance_test:
        logging.info("Running equivariance tests.")
        dev = lgn_tests(
            args,
            encoder,
            decoder,
            test_loader,
            alpha_max=args.alpha_max,
            theta_max=args.theta_max,
            cg_dict=encoder.cg_dict,
            unit=args.unit,
        )
        plot_all_dev(dev, osp.join(test_path, "equivariance_tests"))

    # anomaly detection
    if (args.anomaly_detection) and (len(args.signal_paths) > 0):
        logging.info(f"Anomaly detection started. Signal paths: {args.signal_paths}")
        path_ad = Path(make_dir(osp.join(test_path, "anomaly_detection")))
        eps = 1e-16
        bkg_recons, bkg_target, bkg_norms = recons, target, norm_factors
        if args.abs_coord and (args.unit.lower() == "tev"):
            # convert back for consistent unit
            bkg_recons = bkg_recons / 1000
            bkg_target = bkg_target / 1000
        bkg_recons_normalized = bkg_recons / (bkg_norms + eps)
        bkg_target_normalized = bkg_target / (bkg_norms + eps)

        torch.save(bkg_recons, path_ad / f"{args.jet_type}_recons.pt")
        torch.save(bkg_target, path_ad / f"{args.jet_type}_target.pt")
        torch.save(bkg_norms, path_ad / f"{args.jet_type}_norms.pt")
        torch.save(latent, path_ad / f"{args.jet_type}_latent.pt")

        sig_recons_list = []
        sig_target_list = []
        sig_norms_list = []
        sig_recons_normalized_list = []
        sig_target_normalized_list = []
        sig_scores_list = []

        # background vs single signal
        for signal_path, signal_type in zip(args.signal_paths, args.signal_types):
            logging.info(f"Anomaly detection: {args.jet_type} vs {signal_type}.")
            path_ad_single = path_ad / f"single_signals/{signal_type}"
            sig_loader = initialize_test_data(
                paths=signal_path, batch_size=args.test_batch_size
            )
            sig_recons, sig_target, sig_latent, sig_norms = validate(
                args,
                sig_loader,
                encoder,
                decoder,
                args.load_epoch,
                args.model_path,
                args.device,
                for_test=True,
            )

            sig_recons_normalized = sig_recons / (sig_norms + eps)
            sig_target_normalized = sig_target / (sig_norms + eps)

            scores_dict, true_labels, sig_scores, bkg_scores = anomaly_scores_sig_bkg(
                sig_recons,
                sig_target,
                sig_recons_normalized,
                sig_target_normalized,
                bkg_recons,
                bkg_target,
                bkg_recons_normalized,
                bkg_target_normalized,
                include_emd=True,
                batch_size=args.test_batch_size,
            )
            get_ROC_AUC(scores_dict, true_labels, save_path=path_ad_single)
            plot_p(
                args,
                sig_target * 1000
                if args.abs_coord and (args.unit.lower() == "tev")
                else sig_target,
                sig_recons * 1000
                if args.abs_coord and (args.unit.lower() == "tev")
                else sig_recons,
                save_dir=path_ad_single,
                jet_type=signal_type,
            )

            # add to list
            sig_recons_list.append(sig_recons)
            sig_target_list.append(sig_target)
            sig_norms_list.append(sig_norms)
            sig_recons_normalized_list.append(sig_recons_normalized)
            sig_target_normalized_list.append(sig_target_normalized)
            sig_scores_list.append(sig_scores)

            # save results
            torch.save(sig_recons, path_ad_single / f"{signal_type}_recons.pt")
            torch.save(sig_target, path_ad_single / f"{signal_type}_target.pt")
            torch.save(sig_norms, path_ad_single / f"{signal_type}_norms.pt")
            torch.save(sig_latent, path_ad_single / f"{signal_type}_latent.pt")

        # background vs. all signals
        logging.info(f"Anomaly detection: {args.jet_type} vs {args.signal_types}.")
        sig_recons = torch.cat(sig_recons_list, dim=0)
        sig_target = torch.cat(sig_target_list, dim=0)
        sig_norms = torch.cat(sig_norms_list, dim=0)
        sig_recons_normalized = torch.cat(sig_recons_normalized_list, dim=0)
        sig_target_normalized = torch.cat(sig_target_normalized_list, dim=0)

        # concatenate all signal scores
        sig_scores = {
            k: np.concatenate([v[k] for v in sig_scores_list], axis=0)
            for k in sig_scores_list[0].keys()
        }
        # signals and backgrounds
        scores_dict = {
            k: np.concatenate([sig_scores[k], bkg_scores[k]]) for k in sig_scores.keys()
        }
        true_labels = np.concatenate(
            [
                np.ones_like(sig_scores[list(sig_scores.keys())[0]]),
                -np.ones_like(bkg_scores[list(sig_scores.keys())[0]]),
            ]
        )
        get_ROC_AUC(scores_dict, true_labels, save_path=path_ad)

    elif (args.anomaly_detection) and (len(args.signal_paths) > 0):
        logging.error("No signal paths given for anomaly detection.")




def setup_argparse():
    parser = argparse.ArgumentParser(description="LGN Autoencoder on Test Dataset")

    # Data
    parse_data_settings(parser)

    # Model
    parse_model_settings(parser)

    # Test
    parser.add_argument(
        "--plot-only",
        action="store_true",
        default=False,
        help="Only plot the results without the inference. If inference results are not found, run inference first.",
    )
    parser.add_argument(
        "--loss-choice",
        type=str,
        default="ChamferLoss",
        metavar="",
        help="Choice of loss function. Options: ('ChamferLoss', 'EMDLoss', 'hybrid')",
    )
    parser.add_argument(
        "--get-real-method",
        type=str,
        default="real",
        metavar="",
        help="Method to map complexified vectors to real ones. \n"
        "Supported type: \n"
        "  - real: real component is taken (default).\n"
        "  - imag: imaginary component is taken. \n"
        "  - sum : sum of real and imaginary components is taken. \n"
        "  - norm: norm of real and imaginary components is taken. \n"
        "  - mean: mean of real and imaginary components is taken.",
    )
    parser.add_argument(
        "--chamfer-jet-features",
        type=get_bool,
        default=True,
        help="Whether to take into the jet features.",
    )
    parser.add_argument(
        "--device",
        type=get_device,
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        metavar="",
        help="Device to which the model is initialized. Options: ('gpu', 'cpu', 'cuda', '-1')."
        "Default: -1, which means deciding device based on whether gpu is available.",
    )
    parser.add_argument(
        "--dtype",
        type=get_dtype,
        default=torch.float64,
        metavar="",
        help="Data type to which the model is initialized. Options: ('float', 'float64', 'double'). Default: float64",
    )

    # Load models
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        metavar="",
        help="Path of the trained model to load and test.",
    )
    parser.add_argument(
        "--load-epoch",
        type=int,
        default=-1,
        metavar="",
        help="Epoch number of the trained model to load. -1 for loading weights in the best model.",
    )

    # Plots
    parse_plot_settings(parser)

    # Convariance tests
    parse_covariance_test_settings(parser)
    parser.add_argument(
        "--anomaly-detection",
        action="store_true",
        default=False,
        help="Whether to run anomaly detection.",
    )
    parser.add_argument(
        "--anomaly-scores-batch-size",
        type=int,
        default=-1,
        metavar="",
        help="Batch size when computing anomaly scores. Used for calculating chamfer distances. "
        "Default: -1, which means not using batch size.",
    )
    parser.add_argument(
        "--signal-paths",
        nargs="+",
        type=str,
        metavar="",
        default=[],
        help="Paths to all signal files",
    )
    parser.add_argument(
        "--signal-types",
        nargs="+",
        type=str,
        metavar="",
        default=[],
        help="Types of jets in the signal files",
    )
    parser.add_argument(
        "--plot-num-rocs",
        type=int,
        metavar="",
        default=-1,
        help="Number of ROC curves to keep when plotting (after sorted by AUC). "
        "If the value takes one of {0, -1}, all ROC curves are kept.",
    )

    args = parser.parse_args()

    if args.load_epoch < 0:
        args.load_epoch = best_epoch(args.model_path, num=args.load_epoch)
    if args.model_path is None:
        raise ValueError("--model-path needs to be specified.")

    return args


if __name__ == "__main__":
    import sys

    torch.autograd.set_detect_anomaly(True)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    args = setup_argparse()
    logging.info(f"{args=}")
    test(args)
