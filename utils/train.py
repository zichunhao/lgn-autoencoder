import logging
from os import mkdir
import numpy as np
from utils.jet_analysis import plot_p
from utils.utils import make_dir, save_data, plot_eval_results, get_eps
import time
import os.path as osp
import warnings
import torch
import torch.nn as nn
import sys
import math
from tqdm import tqdm
sys.path.insert(1, 'utils/')
sys.path.insert(1, 'lgn/')
if not sys.warnoptions:
    warnings.simplefilter("ignore")

BLOW_UP_THRESHOLD = 1e8


def train_loop(args, train_loader, valid_loader, encoder, decoder,
               optimizer_encoder, optimizer_decoder, outpath, device=None):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.save_dir is None:
        raise ValueError('Please specify directory of saving the models!')

    make_dir(args.save_dir)

    train_avg_losses = []
    valid_avg_losses = []
    dts = []

    outpath_train_jet_plots = make_dir(osp.join(outpath, 'jet_plots/train'))
    outpath_valid_jet_plots = make_dir(osp.join(outpath, 'jet_plots/valid'))
    best_loss = math.inf
    best_epoch = 1
    num_stale_epochs = 0

    total_epoch = args.num_epochs if not args.load_to_train else args.num_epochs + args.load_epoch

    for ep in range(args.num_epochs):
        epoch = args.load_epoch + ep if args.load_to_train else ep

        # Training
        start = time.time()
        train_avg_loss, train_recons, train_target = train(
            args, train_loader, encoder, decoder,
            optimizer_encoder, optimizer_decoder, epoch,
            outpath, is_train=True, device=device
        )

        # Validation
        valid_avg_loss, valid_recons, valid_target = validate(args, valid_loader, encoder, decoder,
                                                           epoch, outpath, device=device)
        if (abs(valid_avg_loss) < best_loss):
            best_loss = valid_avg_loss
            num_stale_epochs = 0
            best_epoch = epoch + 1
            torch.save(encoder.state_dict(), osp.join(outpath, "weights_encoder/best_encoder_weights.pth"))
            torch.save(decoder.state_dict(), osp.join(outpath, "weights_decoder/best_decoder_weights.pth"))
        else:
            num_stale_epochs += 1

        if args.abs_coord and (args.unit.lower() == 'tev'):
            # Convert to GeV for plotting
            train_target *= 1000
            train_recons *= 1000
            valid_target *= 1000
            valid_recons *= 1000

        # EMD: Plot every epoch because model trains slowly with the EMD loss.
        # Others (MSE and chamfer losses): Plot every args.plot_freq epoch or the best epoch.
        is_emd = 'emd' in args.loss_choice.lower()
        if args.plot_freq > 0:
            if (epoch >= args.plot_start_epoch):
                plot_epoch = ((epoch + 1) % args.plot_freq == 0) or (num_stale_epochs == 0)
            else:
                plot_epoch = False
        else:
            plot_epoch = (num_stale_epochs == 0)
        to_plot = is_emd or plot_epoch

        if to_plot:
            for target, recons, dir in zip(
                (train_target, valid_target),
                (train_recons, valid_recons),
                (outpath_train_jet_plots, outpath_valid_jet_plots)
            ):
                plot_p(args, target, recons, save_dir=dir, cutoff=args.cutoff, epoch=epoch)

        dt = time.time() - start

        dts.append(dt)
        train_avg_losses.append(train_avg_loss)
        valid_avg_losses.append(valid_avg_loss)

        np.savetxt(osp.join(outpath, 'model_evaluations/losses_training.txt'), train_avg_losses)
        np.savetxt(osp.join(outpath, 'model_evaluations/losses_validation.txt'), valid_avg_losses)
        np.savetxt(osp.join(outpath, 'model_evaluations/dts.txt'), dts)

        logging.info(f'epoch={epoch+1}/{total_epoch}, train_loss={train_avg_loss}, valid_loss={valid_avg_loss}, '
                     f'{dt=}s, {num_stale_epochs=}, {best_epoch=}')

        if (epoch > 0) and ((epoch + 1) % 10 == 0):
            plot_eval_results(args, data=(train_avg_losses[-10:], valid_avg_losses[-10:]),
                              data_name=f"losses from {epoch + 1 - 10} to {epoch + 1}", outpath=outpath)
        if (epoch > 0) and ((epoch + 1) % 100 == 0):
            plot_eval_results(args, data=(train_avg_losses, valid_avg_losses),
                              data_name='Losses', outpath=outpath)

        if num_stale_epochs > args.patience:
            logging.info(f'Number of stale epochs reached the set patience ({args.patience}). Training breaks.')
            return best_epoch

        if (abs(valid_avg_loss) > BLOW_UP_THRESHOLD) or (abs(train_avg_loss) > BLOW_UP_THRESHOLD):
            logging.error('Loss blows up. Training breaks.')
            
            error_path = make_dir(osp.join(outpath, 'errors'))
            torch.save(train_target, osp.join(error_path, 'p4_target_train.pt'))
            torch.save(train_recons, osp.join(error_path, 'p4_recons_train.pt'))
            torch.save(valid_target, osp.join(error_path, 'p4_target_valid.pt'))
            torch.save(valid_recons, osp.join(error_path, 'p4_recons_valid.pt'))
            torch.save(encoder.state_dict(), osp.join(error_path, 'encoder_weights.pt'))
            torch.save(decoder.state_dict(), osp.join(error_path, 'decoder_weights.pt'))
            
            logging.error('Saved error data to: ' + error_path)
            return best_epoch

    # Save global data
    save_data(data=train_avg_losses, data_name='losses', is_train=True, outpath=outpath, epoch=-1)
    save_data(data=valid_avg_losses, data_name='losses', is_train=False, outpath=outpath, epoch=-1)
    save_data(data=dts, data_name='dts', is_train=None, outpath=outpath, epoch=-1)

    plot_eval_results(args, data=(train_avg_losses, valid_avg_losses), data_name='Losses', outpath=outpath)
    plot_eval_results(args, data=dts, data_name='Durations', outpath=outpath)

    return best_epoch


def train(args, loader, encoder, decoder, optimizer_encoder, optimizer_decoder,
          epoch, outpath, is_train=True, for_test=False, device=None):
    
    if args.normalize:
        eps = get_eps(args)

    if is_train:
        assert (optimizer_encoder is not None) and (optimizer_decoder is not None), "Please specify the optimizers."
        encoder.train()
        decoder.train()
        encoder_weight_path = make_dir(osp.join(outpath, "weights_encoder"))
        decoder_weight_path = make_dir(osp.join(outpath, "weights_decoder"))
    else:
        encoder.eval()
        decoder.eval()

    target_data = []
    reconstructed_data = []
    if for_test:  # Save latent space for later analysis
        latent_spaces = []
        if args.normalize:
            norm_factors = []
    else:
        epoch_total_loss = 0

    for i, batch in enumerate(tqdm(loader)):

        if args.normalize:
            norm_factor = torch.abs(batch['p4']).amax(dim=-2, keepdim=True).to(batch['p4'].device)
            batch['p4'] /= (norm_factor + eps)

        latent_features = encoder(batch, covariance_test=False)
        p4_recons = decoder(latent_features, covariance_test=False)
        p4_target = batch['p4']
        if device is not None:
            p4_target = p4_target.to(device=device)

        if args.normalize:
            norm_factor = norm_factor.to(p4_recons.device)
            reconstructed_data.append((p4_recons[0]*norm_factor).detach().cpu())
            target_data.append((p4_target*norm_factor).detach().cpu())
        else:
            reconstructed_data.append(p4_recons[0].cpu().detach())
            target_data.append(p4_target.cpu().detach())

        if for_test:
            latent_spaces.append({k: latent_features[k].squeeze(dim=2) for k in latent_features.keys()})
            if args.normalize:
                norm_factors.append(norm_factor.squeeze(dim=2))
        else:
            batch_loss = get_loss(args, p4_recons, p4_target)
            epoch_total_loss += batch_loss.item()

            # Backward propagation
            if is_train:
                optimizer_encoder.zero_grad()
                optimizer_decoder.zero_grad()
                try:
                    batch_loss.backward()
                except RuntimeError as e:
                    import os
                    error_path = osp.join(outpath, 'errors')
                    os.makedirs(error_path, exist_ok=True)
                    torch.save(p4_recons, osp.join(error_path, 'p4_recons.pt'))
                    torch.save(p4_target, osp.join(error_path, 'p4_target.pt'))
                    torch.save(encoder.state_dict(), osp.join(error_path, 'encoder_weights.pt'))
                    torch.save(decoder.state_dict(), osp.join(error_path, 'decoder_weights.pt'))
                    raise e
                optimizer_encoder.step()
                optimizer_decoder.step()
                
                # save model
                if ('emd' in args.loss_choice.lower()) and ((i % args.save_freq) == 0 and i > 0):
                    torch.save(encoder.state_dict(), osp.join(encoder_weight_path, f"epoch_{epoch+1}_encoder_weights.pth"))
                    torch.save(decoder.state_dict(), osp.join(decoder_weight_path, f"epoch_{epoch+1}_decoder_weights.pth"))

    reconstructed_data = torch.cat(reconstructed_data, dim=0)
    target_data = torch.cat(target_data, dim=0)
    if for_test:
        latent_dict = {
            k: [latent_spaces[i][k] for i in range(len(latent_spaces))]
            for k in latent_features.keys()
        }
        return reconstructed_data, target_data, latent_dict, norm_factors

    else:
        epoch_avg_loss = epoch_total_loss / len(loader)
        save_data(data=epoch_avg_loss, data_name='loss',
                  is_train=is_train, outpath=outpath, epoch=epoch)
        # Save weights
        if is_train:
            torch.save(encoder.state_dict(), osp.join(encoder_weight_path, f"epoch_{epoch+1}_encoder_weights.pth"))
            torch.save(decoder.state_dict(), osp.join(decoder_weight_path, f"epoch_{epoch+1}_decoder_weights.pth"))

        return epoch_avg_loss, reconstructed_data, target_data


@torch.no_grad()
def validate(args, loader, encoder, decoder, epoch, outpath, device, for_test=False):
    return train(
        args, loader=loader, encoder=encoder, decoder=decoder,
        optimizer_encoder=None, optimizer_decoder=None,
        epoch=epoch, outpath=outpath, is_train=False, device=device, for_test=for_test
    )


def get_loss(args, p4_recons, p4_target):
    loss_choice = args.loss_choice.lower()

    # Chamfer loss
    if 'chamfer' in loss_choice:
        from utils.losses import ChamferLoss
        chamferloss = ChamferLoss(loss_norm_choice=args.chamfer_loss_norm_choice, im=args.chamfer_im)
        batch_loss = chamferloss(p4_recons, p4_target, jet_features=args.chamfer_jet_features)  # output, target

    # EMD loss
    elif 'emd' in loss_choice:
        from utils.losses import EMDLoss
        emd_loss = EMDLoss(eps=get_eps(args))
        batch_loss = emd_loss(p4_recons, p4_target)  # true, output

    # Hybrid of Chamfer loss and EMD loss
    elif loss_choice in ('hybrid', 'combined', 'mix'):
        from utils.losses import ChamferLoss, EMDLoss
        chamfer_loss = ChamferLoss(loss_norm_choice=args.chamfer_loss_norm_choice, im=args.chamfer_im)
        emd_loss = EMDLoss(eps=get_eps(args))
        batch_loss = args.chamfer_loss_weight * chamfer_loss(p4_recons, p4_target, jet_features=args.chamfer_jet_features)
        batch_loss += emd_loss(p4_recons, p4_target)

    # MSE loss
    elif 'mse' in loss_choice:
        mseloss = nn.MSELoss()
        batch_loss = mseloss(p4_recons[0], p4_target)  # output, target

    # Hungarian loss
    elif 'jet' in loss_choice or 'hungarian' in loss_choice:
        from utils.losses import HungarianMSELoss
        hungarian_mse = HungarianMSELoss()
        # TODO: implement different metric for calculating distance
        batch_loss = hungarian_mse(p4_recons[0], p4_target, abs_coord=args.hungarian_abs_coord, polar_coord=args.hungarian_polar_coord)

    else:
        err_msg = f'Current loss choice ({args.loss_choice}) is not implemented. '
        err_msg += "The current available options are ('chamfer', 'emd', 'hybrid', 'mse', 'hungarian')"
        raise NotImplementedError(err_msg)

    return batch_loss
