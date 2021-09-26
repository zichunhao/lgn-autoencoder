import logging
import numpy as np
from utils.jet_analysis.plot import plot_p
from utils.utils import make_dir, save_data, plot_eval_results, get_eps
from utils.chamfer_loss import ChamferLoss
from utils.emd_loss.emd_loss import emd_loss
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
    best_epoch = 0
    num_stale_epochs = 0

    for ep in range(args.num_epochs):
        epoch = args.load_epoch + ep if args.load_to_train else ep

        # Training
        start = time.time()
        train_avg_loss, train_gen, train_target = train(args, train_loader, encoder, decoder,
                                                        optimizer_encoder, optimizer_decoder, epoch,
                                                        outpath, is_train=True, device=device)
        # Validation
        valid_avg_loss, valid_gen, valid_target = validate(args, valid_loader, encoder, decoder,
                                                           epoch, outpath, device=device)
        if (abs(valid_avg_loss) < best_loss):
            best_loss = valid_avg_loss
            num_stale_epochs = 0
            best_epoch = epoch
            torch.save(encoder.state_dict(), osp.join(outpath, "weights_encoder/best_encoder_weights.pth"))
            torch.save(decoder.state_dict(), osp.join(outpath, "weights_decoder/best_decoder_weights.pth"))
        else:
            num_stale_epochs += 1

        train_end = time.time()
        save_data(data=train_end - start, data_name='dts', is_train=None, outpath=outpath, epoch=epoch)
        save_data(data=train_avg_loss, data_name='losses', is_train=True, outpath=outpath, epoch=epoch)
        save_data(data=valid_avg_loss, data_name='losses', is_train=False, outpath=outpath, epoch=epoch)

        if args.abs_coord and (args.unit.lower() == 'tev'):
            # Convert to GeV for plotting
            train_target *= 1000
            train_gen *= 1000
            valid_target *= 1000
            valid_gen *= 1000

        # EMD: Plot every epoch because model trains slowly with the EMD loss.
        # Others (MSE and chamfer losses): Plot every args.plot_freq epoch.
        to_plot = ('emd' in args.loss_choice.lower()) or ('emd' not in args.loss_choice.lower() and ((epoch + 1) % args.plot_freq == 0)) or (ep == 0)
        if to_plot:
            for target, gen, dir in zip((train_target, valid_target),
                                        (train_gen, valid_gen),
                                        (outpath_train_jet_plots, outpath_valid_jet_plots)):
                plot_p(args, target, gen, save_dir=dir,
                       particle_recon_err=('mse' in args.loss_choice.lower()),
                       cutoff=args.cutoff, epoch=epoch)

        dts.append(train_end-start)
        train_avg_losses.append(train_avg_loss)
        valid_avg_losses.append(valid_avg_loss)

        np.savetxt(osp.join(outpath, 'model_evaluations/losses_training.txt'), train_avg_losses)
        np.savetxt(osp.join(outpath, 'model_evaluations/losses_validation.txt'), valid_avg_losses)

        plot_end = time.time()

        logging.info(f'epoch={epoch+1}/{args.num_epochs if not args.load_to_train else args.num_epochs + args.load_epoch}, '
                     f'train_loss={train_avg_loss}, valid_loss={valid_avg_loss}, dt_train={train_end-start}s, dt_plot={plot_end-train_end}s, dt={plot_end-start}s, '
                     f'{num_stale_epochs=}, best_epoch={best_epoch+1}')

        if (epoch > 0) and ((epoch + 1) % 10 == 0):
            plot_eval_results(args, data=(train_avg_losses[-10:], valid_avg_losses[-10:]),
                              data_name=f"losses from {epoch + 1 - 10} to {epoch + 1}", outpath=outpath)
        if (epoch > 0) and ((epoch + 1) % 100 == 0):
            plot_eval_results(args, data=(train_avg_losses, valid_avg_losses),
                              data_name='Losses', outpath=outpath)

        if num_stale_epochs > args.patience:
            logging.info(f'Number of stale epochs reached the set patience ({args.patience}). Training breaks.')
            break

        if (abs(valid_avg_losses) > BLOW_UP_THRESHOLD) or (abs(train_avg_losses) > BLOW_UP_THRESHOLD):
            logging.error('Loss blows up. Training breaks.')
            break

    # Save global data
    save_data(data=train_avg_losses, data_name='losses', is_train=True, outpath=outpath, epoch=-1)
    save_data(data=valid_avg_losses, data_name='losses', is_train=False, outpath=outpath, epoch=-1)
    save_data(data=dts, data_name='dts', is_train=None, outpath=outpath, epoch=-1)

    plot_eval_results(args, data=(train_avg_losses, valid_avg_losses), data_name='Losses', outpath=outpath)
    plot_eval_results(args, data=dts, data_name='Durations', outpath=outpath)

    return train_avg_losses, valid_avg_losses, dts


def train(args, loader, encoder, decoder, optimizer_encoder, optimizer_decoder,
          epoch, outpath, is_train=True, device=None, latent=False):

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
    generated_data = []
    if latent:
        latent_spaces = []
    epoch_total_loss = 0

    for i, batch in enumerate(tqdm(loader)):
        latent_features = encoder(batch, covariance_test=False)
        p4_gen = decoder(latent_features, covariance_test=False)
        generated_data.append(p4_gen[0].cpu().detach())

        p4_target = batch['p4']
        if device is not None:
            p4_target = p4_target.to(device=device)
        target_data.append(p4_target.cpu().detach())
        if latent:
            latent_spaces.append({k: latent_features[k].squeeze(dim=2) for k in latent_features.keys()})

        batch_loss = get_loss(args, p4_gen, p4_target)
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
                torch.save(p4_gen, osp.join(error_path, 'p4_gen.pt'))
                torch.save(p4_target, osp.join(error_path, 'p4_target.pt'))
                torch.save(encoder.state_dict(), osp.join(error_path, 'encoder_weights.pt'))
                torch.save(decoder.state_dict(), osp.join(error_path, 'decoder_weights.pt'))
                raise e
            optimizer_encoder.step()
            optimizer_decoder.step()

            # save model
            if ((i % args.save_freq) == 0 and i > 0):
                torch.save(encoder.state_dict(), osp.join(encoder_weight_path, f"epoch_{epoch+1}_encoder_weights.pth"))
                torch.save(decoder.state_dict(), osp.join(decoder_weight_path, f"epoch_{epoch+1}_decoder_weights.pth"))

    generated_data = torch.cat(generated_data, dim=0)
    target_data = torch.cat(target_data, dim=0)
    if latent:
        latent_dict = {
            k: [latent_spaces[i][k] for i in range(len(latent_spaces))]
            for k in latent_features.keys()
        }

    epoch_avg_loss = epoch_total_loss / len(loader)
    save_data(data=epoch_avg_loss, data_name='loss',
              is_train=is_train, outpath=outpath, epoch=epoch)

    # Save weights
    if is_train:
        torch.save(encoder.state_dict(), osp.join(encoder_weight_path, f"epoch_{epoch+1}_encoder_weights.pth"))
        torch.save(decoder.state_dict(), osp.join(decoder_weight_path, f"epoch_{epoch+1}_decoder_weights.pth"))

    if latent:
        return epoch_avg_loss, generated_data, target_data, latent_dict
    else:
        return epoch_avg_loss, generated_data, target_data


@torch.no_grad()
def validate(args, loader, encoder, decoder, epoch, outpath, device, latent=False):
    return train(
        args, loader=loader, encoder=encoder, decoder=decoder,
        optimizer_encoder=None, optimizer_decoder=None,
        epoch=epoch, outpath=outpath, is_train=False, device=device, latent=latent
    )


def get_loss(args, p4_gen, p4_target):
    if 'chamfer' in args.loss_choice.lower():
        chamferloss = ChamferLoss(loss_norm_choice=args.loss_norm_choice, im=args.im)
        batch_loss = chamferloss(p4_gen, p4_target, jet_features=args.chamfer_jet_features)  # output, target
    elif 'emd' in args.loss_choice.lower():
        batch_loss = emd_loss(p4_target, p4_gen, eps=get_eps(args), device=args.device)  # true, output
    elif 'mse' in args.loss_choice.lower():
        mseloss = nn.MSELoss()
        batch_loss = mseloss(p4_gen[0], p4_target)  # output, target
    elif args.loss_choice.lower() in ('hybrid', 'combined', 'mix'):
        chamferloss = ChamferLoss(loss_norm_choice=args.loss_norm_choice)
        batch_loss = args.chamfer_loss_weight * chamferloss(p4_gen, p4_target, jet_features=args.chamfer_jet_features)
        batch_loss += emd_loss(p4_target, p4_gen, eps=get_eps(args), device=args.device)
    else:
        err_msg = f'Current loss choice ({args.loss_choice}) is not implemented. '
        err_msg += "The current available options are ('chamfer', 'emd', 'mse', 'hybrid')"
        raise NotImplementedError(err_msg)
    return batch_loss
