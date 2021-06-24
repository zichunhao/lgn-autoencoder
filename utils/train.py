import logging
from utils.jet_analysis import plot_p
from utils.utils import make_dir, save_data, plot_eval_results, eps
from utils.chamfer_loss import ChamferLoss
from utils.emd_loss import emd_loss
import time
import os.path as osp
import warnings
import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.insert(1, 'utils/')
sys.path.insert(1, 'lgn/')
if not sys.warnoptions:
    warnings.simplefilter("ignore")


def train(args, loader, encoder, decoder, optimizer_encoder, optimizer_decoder,
          epoch, outpath, is_train=True, device=None):

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
    epoch_total_loss = 0

    for i, batch in enumerate(loader):
        latent_features = encoder(batch, covariance_test=False)
        p4_gen = decoder(latent_features, covariance_test=False)
        if (p4_gen != p4_gen).any():
            raise RuntimeError('NaN data!')
        generated_data.append(p4_gen[0].cpu().detach().numpy())

        p4_target = batch['p4']
        if device is not None:
            p4_target = p4_target.to(device=device)
        target_data.append(p4_target.cpu().detach().numpy())

        if args.loss_choice.lower() in ['chamfer', 'chamferloss', 'chamfer_loss']:
            chamferloss = ChamferLoss(loss_norm_choice=args.loss_norm_choice)
            batch_loss = chamferloss(p4_gen, p4_target, jet_features=True)  # output, target
            epoch_total_loss += batch_loss.item()
        elif args.loss_choice.lower() in ['emd', 'emdloss', 'emd_loss']:
            batch_loss = emd_loss(p4_target, p4_gen, loss_norm_choice=args.loss_norm_choice, eps=eps(args))  # true, output
            epoch_total_loss += batch_loss.item()
        elif args.loss_choice.lower() in ['mse', 'mseloss', 'mse_loss']:
            mseloss = nn.MSELoss()
            batch_loss = mseloss(p4_gen[0], p4_target)  # output, target
            epoch_total_loss += batch_loss

        if (batch_loss != batch_loss).any():
            raise RuntimeError('Batch loss is NaN!')

        # Backward propagation
        if is_train:
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            batch_loss.backward()
            optimizer_encoder.step()
            optimizer_decoder.step()

            # save model
            if ((i % 50) == 0 and i > 0):
                torch.save(encoder.state_dict(), osp.join(encoder_weight_path, f"epoch_{epoch+1}_encoder_weights.pth"))
                torch.save(decoder.state_dict(), osp.join(decoder_weight_path, f"epoch_{epoch+1}_decoder_weights.pth"))

    generated_data = np.concatenate(generated_data, axis=0)
    target_data = np.concatenate(target_data, axis=0)

    epoch_avg_loss = epoch_total_loss / len(loader)
    save_data(data=epoch_avg_loss, data_name='loss',
              is_train=is_train, outpath=outpath, epoch=epoch)

    # Save weights
    if is_train:
        torch.save(encoder.state_dict(), osp.join(encoder_weight_path, f"epoch_{epoch+1}_encoder_weights.pth"))
        torch.save(decoder.state_dict(), osp.join(decoder_weight_path, f"epoch_{epoch+1}_decoder_weights.pth"))

    return epoch_avg_loss, generated_data, target_data


@torch.no_grad()
def validate(args, loader, encoder, decoder, epoch, outpath, device):
    with torch.no_grad():
        epoch_avg_loss, generated_data, target_data = train(args, loader=loader, encoder=encoder, decoder=decoder,
                                                            optimizer_encoder=None, optimizer_decoder=None,
                                                            epoch=epoch, outpath=outpath, is_train=False, device=device)
    return epoch_avg_loss, generated_data, target_data


def train_loop(args, train_loader, valid_loader, encoder, decoder, optimizer_encoder, optimizer_decoder, outpath, device=None):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    assert (args.save_dir is not None), "Please specify directory of saving the models!"
    make_dir(args.save_dir)

    train_avg_losses = []
    valid_avg_losses = []
    dts = []

    outpath_train_jet_plots = make_dir(osp.join(outpath, 'model_evaluations/jet_plots/train'))
    outpath_valid_jet_plots = make_dir(osp.join(outpath, 'model_evaluations/jet_plots/valid'))

    for ep in range(args.num_epochs):
        epoch = args.load_epoch + ep + 1 if args.load_to_train else ep

        # Training
        start = time.time()
        train_avg_loss, train_gen, train_target = train(args, train_loader, encoder, decoder,
                                                        optimizer_encoder, optimizer_decoder, epoch,
                                                        outpath, is_train=True, device=device)
        # Validation
        valid_avg_loss, valid_gen, valid_target = validate(args, valid_loader, encoder, decoder,
                                                           epoch, outpath, device=device)

        dt = time.time() - start
        save_data(data=dt, data_name='dts', is_train=None, outpath=outpath, epoch=epoch)
        save_data(data=train_avg_loss, data_name='losses', is_train=True,
                  outpath=outpath, epoch=epoch)
        save_data(data=valid_avg_loss, data_name='losses', is_train=False,
                  outpath=outpath, epoch=epoch)
        if args.unit.lower() == 'tev':
            # Convert to GeV for plotting
            train_target *= 1000
            train_gen *= 1000
            valid_target *= 1000
            valid_gen *= 1000

        for target, gen, dir in zip((train_target, valid_target),
                                    (train_gen, valid_gen),
                                    (outpath_train_jet_plots, outpath_valid_jet_plots)):
            plot_p(args, target_data=target, gen_data=gen, save_dir=dir,
                   polar_max=args.polar_max, cartesian_max=args.cartesian_max,
                   jet_polar_max=args.jet_polar_max, jet_cartesian_max=args.jet_cartesian_max,
                   num_bins=args.num_bins, cutoff=args.cutoff, epoch=epoch)

        dts.append(dt)
        train_avg_losses.append(train_avg_loss)
        valid_avg_losses.append(train_avg_loss)

        logging.info(f'epoch={epoch+1}/{args.num_epochs if not args.load_to_train else args.num_epochs + args.load_epoch}, '
                     f'train_loss={train_avg_loss}, valid_loss={valid_avg_loss}, dt={dt}')

        if (epoch > 0) and ((epoch + 1) % 10 == 0):
            plot_eval_results(args, data=(train_avg_losses[-50:], valid_avg_losses[-50:]), data_name=f"losses from {epoch + 1 - 50} to {epoch + 1}",
                              outpath=outpath, global_data=False)
        if (epoch > 0) and ((epoch + 1) % 2 == 0):
            plot_eval_results(args, data=(train_avg_losses, valid_avg_losses),
                              data_name='Losses', outpath=outpath, global_data=False)

    # Save global data
    save_data(data=train_avg_losses, data_name='losses', is_train=True, outpath=outpath, epoch=-1)
    save_data(data=valid_avg_losses, data_name='losses', is_train=False, outpath=outpath, epoch=-1)
    save_data(data=dts, data_name='dts', is_train=None, outpath=outpath, epoch=-1)

    plot_eval_results(args, data=(train_avg_losses, valid_avg_losses),
                      data_name='Losses', outpath=outpath, global_data=True)
    plot_eval_results(args, data=dts, data_name='Time durations', outpath=outpath, global_data=True)

    return train_avg_losses, valid_avg_losses, dts
