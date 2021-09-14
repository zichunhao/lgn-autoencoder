from args import setup_argparse
from utils.utils import create_model_folder, latest_epoch
from utils.train import train_loop
from utils.initialize import initialize_autoencoder, initialize_data, initialize_test_data, initialize_optimizers

from lgn.models.autotest.lgn_tests import lgn_tests
from lgn.models.autotest.utils import plot_all_dev

import torch
import os.path as osp
import logging


def main(args):
    if args.load_to_train and args.load_epoch < 0:
        args.load_epoch = latest_epoch(args.load_path, num=args.load_epoch)
    logging.info(args)

    train_data_path = osp.join(args.file_path, f"{args.jet_type}_{args.file_suffix}.pt")
    test_data_path = osp.join(args.file_path, f"{args.jet_type}_{args.file_suffix}_test.pt")

    train_loader, valid_loader = initialize_data(path=train_data_path,
                                                 batch_size=args.batch_size,
                                                 train_fraction=args.train_fraction,
                                                 num_val=args.num_valid)
    test_loader = initialize_test_data(path=test_data_path, batch_size=args.test_batch_size)

    """Initializations"""
    encoder, decoder = initialize_autoencoder(args)

    # Training and equivariance test
    if not args.equivariance_test_only:
        optimizer_encoder, optimizer_decoder = initialize_optimizers(args, encoder, decoder)

        # Both on gpu
        if (next(encoder.parameters()).is_cuda and next(encoder.parameters()).is_cuda):
            logging.info('The models are initialized on GPU...')
        # One on cpu and the other on gpu
        elif (next(encoder.parameters()).is_cuda or next(encoder.parameters()).is_cuda):
            raise AssertionError("The encoder and decoder are not trained on the same device!")
        # Both on cpu
        else:
            logging.info('The models are initialized on CPU...')

        logging.info(f'Training over {args.num_epochs} epochs...')

        '''Training'''
        # Load existing model
        if args.load_to_train:
            outpath = args.load_path
            encoder.load_state_dict(torch.load(osp.join(outpath, f'weights_encoder/epoch_{args.load_epoch}_encoder_weights.pth'),
                                               map_location=args.device))
            decoder.load_state_dict(torch.load(osp.join(outpath, f'weights_decoder/epoch_{args.load_epoch}_decoder_weights.pth'),
                                               map_location=args.device))
        # Create new model
        else:
            import json
            outpath = create_model_folder(args)
            args_dir = osp.join(outpath, "args_cache.json")
            with open(args_dir, "w") as f:
                json.dump({k: str(v) for k, v in vars(args).items()}, f)

        train_loop(args, train_loader, valid_loader, encoder, decoder, optimizer_encoder, optimizer_decoder, outpath, args.device)

        # Equivariance tests
        if args.equivariance_test:
            encoder.load_state_dict(torch.load(osp.join(outpath, f'weights_encoder/epoch_{args.num_epochs}_encoder_weights.pth'),
                                               map_location=args.test_device))
            decoder.load_state_dict(torch.load(osp.join(outpath, f'weights_decoder/epoch_{args.num_epochs}_decoder_weights.pth'),
                                               map_location=args.test_device))
            dev = lgn_tests(encoder, decoder, test_loader, args, alpha_max=args.alpha_max,
                            theta_max=args.theta_max, epoch=args.num_epochs, cg_dict=encoder.cg_dict)
            plot_all_dev(dev, osp.join(outpath, 'model_evaluations/equivariance_tests'))

        logging.info("Training completed!")
    # equivariance test only
    else:
        if args.load_path is not None:
            loadpath = args.load_path
        else:
            raise RuntimeError("load-path cannot be None if equivariance-test-only is True!")
        load_epoch = args.load_epoch if args.load_epoch is not None else args.num_epochs

        encoder.load_state_dict(torch.load(osp.join(loadpath, f'weights_encoder/epoch_{load_epoch}_encoder_weights.pth'),
                                           map_location=args.test_device))
        decoder.load_state_dict(torch.load(osp.join(loadpath, f'weights_decoder/epoch_{load_epoch}_decoder_weights.pth'),
                                           map_location=args.test_device))

        dev = lgn_tests(encoder, decoder, test_loader, args, alpha_max=args.alpha_max,
                        theta_max=args.theta_max, epoch=args.num_epochs, cg_dict=encoder.cg_dict, unit=args.unit)
        plot_all_dev(dev, osp.join(loadpath, 'model_evaluations/equivariance_tests'))

        logging.info("Done!")


if __name__ == "__main__":
    import sys
    torch.autograd.set_detect_anomaly(True)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    args = setup_argparse()
    main(args)
