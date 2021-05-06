import os
import os.path as osp
import torch
import matplotlib.pyplot as plt

def create_model_folder(args, model):
    if not osp.isdir(args.outpath):
        os.makedirs(args.outpath)

    model_fname = get_model_fname(args)
    outpath = osp.join(args.outpath, model_fname)

    return make_dir(outpath)

def get_model_fname(args):
    maxzf = max(args.maxzf) if type(args.maxzf) == list else args.maxzf
    maxdim = max(args.maxdim) if type(args.maxdim) == list else args.maxdim
    model_fname = f"LGNAutoencoder_numLatentScalar_{args.num_latent_scalars}_numLatentVectors_{args.num_latent_vectors}_maxzf_{maxzf}_maxdim_{maxdim}"
    return model_fname

def make_dir(path):
    if not osp.isdir(path):
        os.makedirs(path)
    return path

def save_data(data, data_name, is_train, outpath, epoch=-1):
    '''
    Save data like losses and dts. If epoch is -1, the data will be considered a global data, such as
    the losses over all epochs.
    '''
    outpath = osp.join(outpath, "model_evaluations/pkl_files")
    if isinstance(data, torch.Tensor):
        data = data.cpu()

    if is_train is None:
        if epoch >= 0:
            torch.save(data, f"{outpath}/{data_name}_epoch_{epoch+1}.pkl")
        else:
            torch.save(data, f"{outpath}/{data_name}.pkl")

    if epoch >= 0:
        if is_train:
            torch.save(data, f"{outpath}/train_{data_name}_epoch_{epoch+1}.pkl")
        else:
            torch.save(data, f"{outpath}/valid_{data_name}_epoch_{epoch+1}.pkl")
    else:
        if is_train:
            torch.save(data, f"{outpath}/train_{data_name}.pkl")
        else:
            torch.save(data, f"{outpath}/valid_{data_name}.pkl")


def plot_eval_results(args, data, data_name, outpath, global_data=True):
    '''
    Plot evaluation results
    '''
    outpath = make_dir(osp.join(outpath, "model_evaluations"))
    if args.load_to_train:
        start = args.load_epoch + 1
        end = start + args.num_epochs
    else:
        start = 1
        end = args.num_epochs

    # (train, label)
    if type(data) in [tuple, list] and len(data) == 2:
        train, valid = data
        if global_data:
            x = [i for i in range(start, end+1)]
        else:
            x = [start + i for i in range(len(train))]

        if isinstance(train, torch.Tensor):
            train = train.detach().cpu().numpy()
        if isinstance(valid, torch.Tensor):
            valid = valid.detach().cpu().numpy()
        plt.plot(x, train, label='Train')
        plt.plot(x, valid, label='Valid')
        plt.legend()
    # only one type of data (e.g. dt)
    else:
        if global_data:
            x = [i for i in range(start, end+1)]
        else:
            x = [start + i for i in range(len(train))]
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        plt.plot(x, data)

    plt.xlabel('Epoch')
    plt.ylabel(data_name)
    plt.title(data_name)
    save_name = "_".join(data_name.lower().split(" "))
    plt.savefig(f"{outpath}/{save_name}.pdf")
    plt.close()
