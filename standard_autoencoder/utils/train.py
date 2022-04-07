import torch.nn as nn
import torch
import numpy as np
import math
import os.path as osp
import time
from utils.utils import make_dir, save_data, plot_eval_results, eps
from utils.jet_analysis import plot_p
import logging
from tqdm import tqdm

BLOW_UP_THRESHOLD = 1e8
# The percentage of total epochs after which the thus-far best result is plotted
PLOT_START_PERCENTAGE = 0.05

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

    for i, data in enumerate(tqdm(loader)):
        p4_target = data.to(args.dtype)
        p4_gen = decoder(
            encoder(p4_target, metric=args.encoder_metric),
            metric=args.decoder_metric
        )
        generated_data.append(p4_gen.cpu().detach())

        if device is not None:
            p4_target = p4_target.to(device=device)
        target_data.append(p4_target.cpu().detach())

        batch_loss = get_loss(args, p4_gen, p4_target.to(args.dtype))
        epoch_total_loss += batch_loss.cpu().item()

        # Backward propagation
        if is_train:
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            batch_loss.backward()
            optimizer_encoder.step()
            optimizer_decoder.step()

            if ('emd' in args.loss_choice.lower()) and ((i % args.save_freq) == 0 and i > 0):
                torch.save(
                    encoder.state_dict(),
                    osp.join(encoder_weight_path, f"epoch_{epoch+1}_encoder_weights.pth")
                )
                torch.save(
                    decoder.state_dict(),
                    osp.join(decoder_weight_path, f"epoch_{epoch+1}_decoder_weights.pth")
                )

    generated_data = torch.cat(generated_data, dim=0)
    target_data = torch.cat(target_data, dim=0)

    epoch_avg_loss = epoch_total_loss / len(loader)

    # Save weights
    if is_train:
        torch.save(encoder.state_dict(), osp.join(encoder_weight_path, f"epoch_{epoch}_encoder_weights.pth"))
        torch.save(decoder.state_dict(), osp.join(decoder_weight_path, f"epoch_{epoch}_decoder_weights.pth"))

    return epoch_avg_loss, generated_data, target_data


@torch.no_grad()
def validate(args, loader, encoder, decoder, epoch, outpath, device):
    with torch.no_grad():
        epoch_avg_loss, generated_data, target_data = train(args, loader=loader, encoder=encoder, decoder=decoder,
                                                            optimizer_encoder=None, optimizer_decoder=None,
                                                            epoch=epoch, outpath=outpath, is_train=False, device=device)
    return epoch_avg_loss, generated_data, target_data


def train_loop(args, train_loader, valid_loader, encoder, decoder,
               optimizer_encoder, optimizer_decoder, outpath, device=None):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    assert (args.save_dir is not None), "Please specify directory of saving the models!"
    make_dir(args.save_dir)

    train_avg_losses = []
    valid_avg_losses = []
    dts = []

    best_epoch = 1
    num_stale_epochs = 0
    best_loss = math.inf

    outpath_train_jet_plots = make_dir(osp.join(outpath, 'model_evaluations/jet_plots/train'))
    outpath_valid_jet_plots = make_dir(osp.join(outpath, 'model_evaluations/jet_plots/valid'))

    total_epoch = args.num_epochs if not args.load_to_train else args.num_epochs + args.load_epoch

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
            best_epoch = epoch + 1
            torch.save(
                encoder.state_dict(),
                osp.join(outpath, "weights_encoder/best_encoder_weights.pth")
            )
            torch.save(
                decoder.state_dict(),
                osp.join(outpath, "weights_decoder/best_decoder_weights.pth")
            )
        else:
            num_stale_epochs += 1

        dt = time.time() - start

        if (args.abs_coord and (args.unit.lower() == 'tev')) and not args.normalized:
            # Convert to GeV for plotting
            train_target *= 1000
            train_gen *= 1000
            valid_target *= 1000
            valid_gen *= 1000

        if args.plot_freq > 0:
            if (epoch >= int(args.num_epochs * PLOT_START_PERCENTAGE)):
                plot_epoch = ((epoch + 1) % args.plot_freq == 0) or (num_stale_epochs == 0)
            else:
                plot_epoch = ((epoch + 1) % args.plot_freq == 0)
        else:
            plot_epoch = (num_stale_epochs == 0)

        is_emd = 'emd' in args.loss_choice.lower()
        to_plot = is_emd or plot_epoch
        if to_plot:
            for target, gen, dir in zip((train_target, valid_target),
                                        (train_gen, valid_gen),
                                        (outpath_train_jet_plots, outpath_valid_jet_plots)):
                logging.debug("plotting")
                plot_p(args, p4_target=target, p4_gen=gen, save_dir=dir, epoch=epoch, show=False)

        dts.append(dt)
        train_avg_losses.append(train_avg_loss)
        valid_avg_losses.append(valid_avg_loss)
        np.savetxt(osp.join(outpath, 'model_evaluations/losses_training.txt'), train_avg_losses)
        np.savetxt(osp.join(outpath, 'model_evaluations/losses_validation.txt'), valid_avg_losses)
        np.savetxt(osp.join(outpath, 'model_evaluations/dts.txt'), dts)

        logging.info(f'epoch={epoch+1}/{total_epoch}, train_loss={train_avg_loss}, valid_loss={valid_avg_loss}, '
                     f'{dt=}s, {num_stale_epochs=}, {best_epoch=}')

        if args.plot_freq > 0:
            if (epoch > 0) and (epoch % int(args.plot_freq) == 0):
                plot_eval_results(
                    args, data=(train_avg_losses, valid_avg_losses),
                    data_name='Losses', outpath=outpath, start=epoch-args.plot_freq
                )

        if num_stale_epochs > args.patience:
            logging.info(
                f'Number of stale epochs reached the set patience ({args.patience}). Training breaks.'
            )
            return best_epoch

        if (abs(valid_avg_loss) > BLOW_UP_THRESHOLD) or (abs(train_avg_loss) > BLOW_UP_THRESHOLD):
            logging.error('Loss blows up. Training breaks.')
            return best_epoch


    # Save global data
    save_data(data=train_avg_losses, data_name='losses', is_train=True, outpath=outpath, epoch=-1)
    save_data(data=valid_avg_losses, data_name='losses', is_train=False, outpath=outpath, epoch=-1)
    save_data(data=dts, data_name='dts', is_train=None, outpath=outpath, epoch=-1)

    plot_eval_results(args, data=(train_avg_losses, valid_avg_losses),
                      data_name='Losses', outpath=outpath)
    plot_eval_results(args, data=dts, data_name='Time durations',
                      outpath=outpath)

    return train_avg_losses, valid_avg_losses, dts


def get_loss(args, p4_gen, p4_target):
    if args.loss_choice.lower() in ['chamfer', 'chamferloss', 'chamfer_loss']:
        from utils.losses import ChamferLoss
        chamferloss = ChamferLoss(loss_norm_choice=args.loss_norm_choice)
        batch_loss = chamferloss(p4_gen, p4_target, jet_features_weight=args.chamfer_jet_features_weight)  # output, target
        return batch_loss

    if args.loss_choice.lower() in ['emd', 'emdloss', 'emd_loss']:
        from utils.losses import emd_loss
        batch_loss = emd_loss(p4_target, p4_gen, eps=eps(args), device=args.device)  # true, output
        return batch_loss

    if args.loss_choice.lower() in ['mse', 'mseloss', 'mse_loss']:
        mseloss = nn.MSELoss()
        batch_loss = mseloss(p4_gen, p4_target)  # output, target
        return batch_loss

    if args.loss_choice.lower() in ['hybrid', 'combined', 'mix']:
        from utils.losses import ChamferLoss
        from utils.losses import emd_loss
        chamferloss = ChamferLoss(loss_norm_choice=args.loss_norm_choice)
        batch_loss = args.chamfer_loss_weight * chamferloss(
            p4_gen, p4_target, jet_features_weight=args.chamfer_jet_features_weight
        ) + emd_loss(
            p4_target, p4_gen, eps=eps(args), device=args.device
        )
        return batch_loss
