import argparse
import logging
from pathlib import Path
import torch
import numpy as np
from utils.jet_analysis import get_ROC_AUC, anomaly_scores_sig_bkg
from utils.utils import get_eps

def main(args: argparse.Namespace) -> None:
    # background
    bkg_type = args.bkg_type
    bkg_path = Path(args.bkg_dir)
    bkg_recons = torch.load(bkg_path / f'{bkg_type}_recons.pt')
    bkg_target = torch.load(bkg_path / f'{bkg_type}_target.pt')
    bkg_norms = torch.load(bkg_path / f'{bkg_type}_norms.pt')
    
    eps = get_eps(bkg_recons.dtype)
    bkg_recons_normalized = bkg_recons / (bkg_norms + eps)
    bkg_target_normalized = bkg_target / (bkg_norms + eps)
    
    # signals
    sig_recons_list = []
    sig_target_list = []
    sig_recons_normalized_list = []
    sig_target_normalized_list = []
    sig_scores_list = []
    
    # background vs. single signal
    for sig_type, sig_path in zip(args.sig_types, args.sig_dirs):
        logging.info(f'Anomaly detection: {bkg_type} vs. {sig_type}.')
        sig_path = Path(sig_path)
        
        sig_recons = torch.load(sig_path / f'{sig_type}_recons.pt')
        sig_target = torch.load(sig_path / f'{sig_type}_target.pt')
        sig_norms = torch.load(sig_path / f'{sig_type}_norms.pt')
        
        sig_recons_normalized = sig_recons / (sig_norms + eps)
        sig_target_normalized = sig_target / (sig_norms + eps)
        
        scores_dict, true_labels, sig_scores, bkg_scores = anomaly_scores_sig_bkg(
            sig_recons, sig_target, sig_recons_normalized, sig_target_normalized,
            bkg_recons, bkg_target, bkg_recons_normalized, bkg_target_normalized,
            include_emd=args.include_emd, batch_size=args.batch_size,
        )
        get_ROC_AUC(scores_dict, true_labels, save_path=sig_path)
        
        # append to lists
        sig_recons_list.append(sig_recons)
        sig_target_list.append(sig_target)
        sig_recons_normalized_list.append(sig_recons_normalized)
        sig_target_normalized_list.append(sig_target_normalized)
        sig_scores_list.append(sig_scores)
        
    # background vs. all signals
    logging.info(f"Anomaly detection: {args.bkg_type} vs. {args.sig_types}.")
    sig_recons = torch.cat(sig_recons_list, dim=0)
    sig_target = torch.cat(sig_target_list, dim=0)
    sig_recons_normalized = torch.cat(sig_recons_normalized_list, dim=0)
    sig_target_normalized = torch.cat(sig_target_normalized_list, dim=0)
    
    # concatenate all signal scores
    sig_scores = {
        k: np.concatenate([v[k] for v in sig_scores_list], axis=0)
        for k in sig_scores_list[0].keys()
    }
    # signals and backgrounds
    scores_dict = {
        k: np.concatenate([sig_scores[k], bkg_scores[k]])
        for k in sig_scores.keys()
    }
    true_labels = np.concatenate([
        np.ones_like(sig_scores[list(sig_scores.keys())[0]]),
        -np.ones_like(bkg_scores[list(sig_scores.keys())[0]])
    ])
    get_ROC_AUC(scores_dict, true_labels, save_path=bkg_path)
    
    return

        
def setup_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='LGN Autoencoder Anomaly Detection')
    
    parser.add_argument("--include-emd", default=False, action="store_true",
                        help="Include EMD as a score for anomaly detection.")
    
    parser.add_argument('--batch-size', '-bs', type=int, default=-1,
                        help='Batch size for ROC AUC computation. If -1, use all data.')
    
    # background
    parser.add_argument('--bkg-dir', type=str, required=True,
                        help='Directory that contains background reconstruction files.')
    parser.add_argument('--bkg-type', type=str, required=True,
                        help='Jet type for background.')
    
    # signals
    parser.add_argument('--sig-dirs', type=str, nargs='+', required=True,
                        help='Directories that contain signal reconstruction files.')
    parser.add_argument('--sig-types', type=str, nargs='+', required=True,
                        help='Jet types for signals.')
    
    return parser.parse_args()

if __name__ == '__main__':
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    args = setup_argparse()
    logging.info(f'{args=}')
    main(args)
    