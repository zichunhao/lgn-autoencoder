import os
import os.path as osp
import pickle

def create_model_folder(args, model):
    if not osp.isdir(args.outpath):
        os.makedirs(args.outpath)

    model_fname = get_model_fname(args, model)
    outpath = osp.join(args.outpath, model_fname)

    make_dir(outpath)

    model_kwargs = {'model_name': model_fname, 'learning_rate': args.lr}

    with open(f'{outpath}/model_kwargs.pkl', 'wb') as f:
        pickle.dump(model_kwargs, f, protocol=pickle.HIGHEST_PROTOCOL)

    return outpath

def get_model_fname(args, model):
    model_name = type(model).__name__
    model_fname = f"{model_name}_numLatentScalar_{args.num_latent_scalars}_numLatentVectors_{args.num_latent_vectors}_lr={args.lr}_maxdim={args.maxdim}"
    return model_fname

def make_dir(path):
    if not osp.isdir(path):
        os.makedirs(path)
