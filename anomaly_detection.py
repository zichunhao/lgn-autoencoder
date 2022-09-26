import argparse
import logging
from pathlib import Path
import torch
from utils.jet_analysis import anomaly_detection_ROC_AUC
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
    sig_recons_all = []
    sig_target_all = []
    sig_recons_normalized_all = []
    sig_target_normalized_all = []
    
    
    # background vs. single signal
    for sig_type, sig_path in zip(args.sig_types, args.sig_dirs):
        logging.info(f'Anomaly detection: {sig_type} vs. {bkg_type}.')
        sig_path = Path(sig_path)
        
        sig_recons = torch.load(sig_path / f'{sig_type}_recons.pt')
        sig_target = torch.load(sig_path / f'{sig_type}_target.pt')
        sig_norms = torch.load(sig_path / f'{sig_type}_norms.pt')
        
        sig_recons_normalized = sig_recons / (sig_norms + eps)
        sig_target_normalized = sig_target / (sig_norms + eps)
        
        anomaly_detection_ROC_AUC(
            sig_recons, sig_target, sig_recons_normalized, sig_target_normalized,
            bkg_recons, bkg_target, bkg_recons_normalized, bkg_target_normalized,
            include_emd=True, batch_size=args.batch_size, save_path=sig_path
        )
        
        # append to lists
        sig_recons_all.append(sig_recons)
        sig_target_all.append(sig_target)
        sig_recons_normalized_all.append(sig_recons_normalized)
        sig_target_normalized_all.append(sig_target_normalized)
        
    # background vs. all signals
    logging.info(f"Anomaly detection: {args.jet_type} vs. {args.sig_types}.")
    sig_recons = torch.cat(sig_recons_all, dim=0)
    sig_target = torch.cat(sig_target_all, dim=0)
    sig_recons_normalized = torch.cat(sig_recons_normalized_all, dim=0)
    sig_target_normalized = torch.cat(sig_target_normalized_all, dim=0)
    
    anomaly_detection_ROC_AUC(
        sig_recons, sig_target, sig_recons_normalized, sig_target_normalized,
        bkg_recons, bkg_target, bkg_recons_normalized, bkg_target_normalized,
        include_emd=True, batch_size=args.batch_size, save_path=bkg_path
    )
    
    return

        
def setup_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='LGN Autoencoder Anomaly Detection')
    
    parser.add_argument('--batch-size', '-bs', type=int, default=-1,
                        help='Batch size for ROC AUC computation. If -1, use all data.')
    
    # background
    parser.add_argument('--bkg-dir', type=str, nargs='+', required=True,
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
    