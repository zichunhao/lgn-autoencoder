import torch
import sys
sys.path.insert(1, 'utils/')
sys.path.insert(1, 'lgn/')
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

import os.path as osp
import time

from utils.chamfer_loss import ChamferLoss
from utils.utils import make_dir, save_data, plot_eval_results

def train(args, loader, encoder, decoder, optimizer_encoder, optimizer_decoder,
          epoch, outpath, is_train=True, device=None):

    if is_train:
        assert (optimizer_encoder is not None) and (optimizer_decoder is not None), "Please specify the optimizers."
        encoder.train()
        decoder.train()
    else:
        encoder.eval()
        decoder.eval()

    target_data = []
    generated_data = []
    epoch_total_loss = 0

    for i, batch in enumerate(loader):
        print(f"batch = {i+1}")
        latent_features = encoder(batch, covariance_test=False)
        p4_gen = decoder(latent_features, covariance_test=False)
        if (p4_gen != p4_gen).any():
            raise RuntimeError('NaN data!')
        generated_data.append(p4_gen)

        p4_target = batch['p4']
        target_data.append(p4_target)

        loss = torch.nn.MSELoss()
        # loss = ChamferLoss(loss_norm_choice=args.loss_norm_choice)
        batch_loss = loss(p4_gen, p4_target)  # preds, targets
        if (batch_loss != batch_loss).any():
            raise RuntimeError('Batch loss is NaN!')
        epoch_total_loss += batch_loss.item()

        # Backward propagation
        if is_train:
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            batch_loss.backward()
            optimizer_encoder.step()
            optimizer_decoder.step()

    epoch_avg_loss = epoch_total_loss / len(loader)
    save_data(data=epoch_avg_loss, data_name='loss', is_train=is_train, outpath=outpath, epoch=epoch)

    # Save loss
    if is_train:
        make_dir(osp.join(outpath, "weights_encoder"))
        make_dir(osp.join(outpath, "weights_decoder"))
        torch.save(encoder.state_dict(), f"{outpath}/weights_encoder/epoch_{epoch+1}_encoder_weights.pth")
        torch.save(decoder.state_dict(), f"{outpath}/weights_decoder/epoch_{epoch+1}_decoder_weights.pth")

    return epoch_avg_loss, generated_data

@torch.no_grad()
def validate(args, loader, encoder, decoder, epoch, outpath, device):
    with torch.no_grad():
        epoch_avg_loss, generated_data = train(args, loader=loader, encoder=encoder, decoder=decoder,
                                               optimizer_encoder=None, optimizer_decoder=None,
                                               epoch=epoch, outpath=outpath, is_train=False, device=device)
    return generated_data, epoch_avg_loss

def train_loop(args, train_loader, valid_loader, encoder, decoder, optimizer_encoder, optimizer_decoder, outpath, device=None):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    assert (args.save_dir is not None), "Please specify directory of saving the models!"
    make_dir(args.save_dir)

    train_avg_losses = []
    valid_avg_losses = []
    dts = []

    for ep in range(args.num_epochs):
        epoch = args.load_epoch + ep + 1 if args.load_to_train else ep

        # Training
        start = time.time()
        train_avg_loss, train_gen = train(args, train_loader, encoder, decoder, optimizer_encoder, optimizer_decoder,
                                          epoch, outpath, is_train=True, device=device)
        # Validation
        valid_avg_loss, valid_gen = validate(args, valid_loader, encoder, decoder, epoch, outpath, device=device)

        dt = time.time() - start
        dts.append(dt)

        train_avg_losses.append(train_avg_loss)
        valid_avg_losses.append(train_avg_loss)

        save_data(data=dt, data_name='dt', is_train=None, outpath=outpath, epoch=epoch)

        print(f'epoch={epoch+1}/{args.num_epochs if not args.load_to_train else args.num_epochs + args.load_epoch}, ' \
              f'train_loss={train_avg_loss}, valid_loss={valid_avg_loss}, dt={dt}')

        if (epoch > 0) and ((epoch + 1) % 10 == 0):
            plot_eval_results(args, data=(train_avg_losses, valid_avg_losses),
                              data_name=f"losses to up to {epoch+1}", outpath=outpath, global_data=True)

    # Save global data
    save_data(data=train_avg_losses, data_name='losses', is_train=True, outpath=outpath, epoch=-1)
    save_data(data=valid_avg_losses, data_name='losses', is_train=False, outpath=outpath, epoch=-1)
    save_data(data=dts, data_name='dts', is_train=None, outpath=outpath, epoch=-1)

    return train_avg_losses, valid_avg_losses, dts
