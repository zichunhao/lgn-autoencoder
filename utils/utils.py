from argparse import Namespace
import os
import os.path as osp
from typing import List, Optional, Tuple, Union
import torch
import matplotlib.pyplot as plt
import glob


def create_model_folder(args: Namespace) -> str:
    make_dir(args.save_dir)
    return make_dir(osp.join(args.save_dir, get_model_fname(args)))


def get_model_fname(args: Namespace) -> str:
    encoder_cg = ''
    decoder_cg = ''
    for i in range(len(args.encoder_num_channels)):
        encoder_cg += str(args.encoder_num_channels[i])
    for i in range(len(args.decoder_num_channels)):
        decoder_cg += str(args.decoder_num_channels[i])
    model_fname = f"LGNAutoencoder_{args.jet_type}Jet_{args.map_to_latent}_tauLS{args.tau_latent_scalars}_tauLV{args.tau_latent_vectors}_encoder{encoder_cg}_decoder{decoder_cg}"
    if args.custom_suffix is not None:
        model_fname += f'_{args.custom_suffix}'
    return model_fname


def make_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_data(
    data: Union[Tuple[float], List[float]],
    data_name: str, 
    is_train: bool, 
    outpath: str, 
    epoch: int = -1
):
    '''
    Save data like losses and dts. If epoch is -1, the data will be considered a global data, such as
    the losses over all epochs.
    '''
    outpath = make_dir(osp.join(outpath, "model_evaluations/pkl_files"))
    if isinstance(data, torch.Tensor):
        data = data.cpu()

    if is_train is None:
        if epoch >= 0:
            torch.save(data, osp.join(outpath, f'{data_name}_epoch_{epoch+1}.pkl'))
        else:
            torch.save(data, osp.join(outpath, f'{data_name}.pkl'))
        return

    if epoch >= 0:
        if is_train:
            torch.save(data, osp.join(outpath, f'train_{data_name}_epoch_{epoch+1}.pkl'))
        else:
            torch.save(data, osp.join(outpath, f'valid_{data_name}_epoch_{epoch+1}.pkl'))
    else:
        if is_train:
            torch.save(data, osp.join(outpath, f'train_{data_name}.pkl'))
        else:
            torch.save(data, osp.join(outpath, f'valid_{data_name}.pkl'))


def plot_eval_results(
    args: Namespace, 
    data: Union[Tuple[float], List[float]], 
    data_name: str, 
    outpath: str, 
    start: Optional[int] = None
):
    '''
    Plot evaluation results
    '''
    outpath = make_dir(osp.join(outpath, "model_evaluations/evaluation_plots"))
    if args.load_to_train:
        start = args.load_epoch + 1
    else:
        start = 1 if start is None else start
    # (train, label)
    if (type(data) in [tuple, list]) and (len(data) == 2) and (type(data[0]) in [tuple, list]) and (type(data[1]) in [tuple, list]):
        train, valid = data
        x = [start + i for i in range(len(train))]
        if isinstance(train, torch.Tensor):
            train = train.detach().cpu().numpy()
        if isinstance(valid, torch.Tensor):
            valid = valid.detach().cpu().numpy()
        plt.plot(x, train, label='Train', alpha=0.8)
        plt.plot(x, valid, label='Valid', alpha=0.8)
        plt.legend()
    # only one type of data (e.g. dt)
    else:
        x = [start + i for i in range(len(data))]
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        plt.plot(x, data)

    plt.xlabel('Epoch')
    plt.ylabel(data_name)
    plt.title(data_name)
    save_name = "_".join(data_name.lower().split(" "))
    plt.savefig(osp.join(outpath, f"{save_name}.pdf"), bbox_inches='tight')
    plt.close()


def get_eps(args: Namespace) -> float:
    return 1e-16 if args.dtype in (torch.float64, torch.double) else 1e-12


def get_p_polar(
    p: torch.Tensor, 
    eps: float = 1e-16, 
    keep_p0: bool = False
) -> torch.Tensor:
    """
    (E, px, py, pz) -> (eta, phi, pt) or (E, eta, phi, pt)

    keep_p0: bool
        Whether to keep p0.
        Optional, default: False
    """
    px = p[..., 1]
    py = p[..., 2]
    pz = p[..., 3]

    pt = torch.sqrt(px ** 2 + py ** 2 + eps)
    try:
        eta = torch.asinh(pz / (pt + eps))
    except AttributeError:
        eta = arcsinh(pz / (pt + eps))
    phi = torch.atan2(py + eps, px + eps)

    if not keep_p0:
        return torch.stack((eta, phi, pt), dim=-1)
    else:
        E = p[..., 0]
        return torch.stack((E, eta, phi, pt), dim=-1)


def arcsinh(z):
    return torch.log(z + torch.sqrt(1 + torch.pow(z, 2)))


def get_compression_rate(
    ls: int, 
    lv: int, 
    map_to_latent: str, 
    vec_dim: int = 4, 
    num_particles: int = 30
) -> float:
    """
    Get the compression rate based on the multiplicities of scalars and vectors in the latent space.
    """
    input_params = vec_dim * num_particles
    latent_params = 2 * (ls + 4 * lv)  # Complexified
    ratio = latent_params / input_params
    if '&' in map_to_latent:
        return len(map_to_latent.split('&')) * ratio
    return ratio


def latest_epoch(
    model_path: str, 
    num: int = -1
) -> int:
    path = osp.join(model_path, 'weights_decoder/*.pth')
    file_list = glob.glob(path)
    print(f"{path=}, {file_list=}")
    epochs = [[int(s) for s in filename.split('_') if s.isdigit()] for filename in file_list]
    epochs.sort()
    try:
        latest = epochs[num][0]
    except IndexError:
        try:
            latest = epochs[-1][0]
        except IndexError:
            raise RuntimeError(f"Model does not exist in {model_path}")
    return latest
