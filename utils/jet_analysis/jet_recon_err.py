import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import make_dir
import os.path as osp

FIGSIZE = (16, 4)
LABELS_CARTESIAN = (r'$M$', r'$P_x$', r'$P_y$', r'$P_z$')
LABELS_POLAR = (r'$M$', r'$P_\mathrm{T}$', r'$\eta$', r'$\phi$')
LABELS = (LABELS_CARTESIAN, LABELS_POLAR)
COORDINATES = ('cartesian', 'polar')


def plot_jet_recon_err(args, jet_target_cartesian, jet_gen_cartesian, jet_target_polar, jet_gen_polar,
                       save_dir, epoch=None, eps=1e-16,
                       get_rel_err=(lambda p_target, p_gen, eps: (p_target-p_gen)/(p_target+0.01*np.median(p_target)+eps)),
                       show=False):
    """Plot reconstruction errors for jet."""
    res_cartesian = [get_rel_err(jet_gen_cartesian[i], jet_target_cartesian[i], eps) for i in range(4)]
    res_polar = [get_rel_err(jet_target_polar[i], jet_target_polar[i], eps) for i in range(4)]
    ranges = get_bins(args.num_bins)

    fig, axs = plt.subplots(1, 4, figsize=FIGSIZE, sharey=False)
    for res, labels, coordinate, bin_tuple in zip((res_cartesian, res_polar), LABELS, COORDINATES, ranges):
        for ax, res, bins, label in zip(axs, res, bin_tuple, labels):
            ax.hist(res, bins=bins, label=get_legend(res), histtype='step', stacked=True)
            ax.set_xlabel(fr'$\delta${label}')
            ax.set_ylabel('Number of Jets')
            ax.legend()
        plt.tight_layout()
        if save_dir:
            path = make_dir(osp.join(save_dir, f'jet_reconstruction_errors/{coordinate}'))
            if epoch is not None:
                plt.savefig(osp.join(path, f'jet_reconstruction_errors_epoch_{epoch+1}.pdf'))
            else:
                plt.savefig(osp.join(path, 'jet_reconstruction_errors.pdf'))
        if show:
            plt.show()


def default_get_rel_err(p_target, p_gen, eps, alpha=0.01):
    if type(p_target) is torch.Tensor:
        p_target = p_target.cpu().detach().numpy()
    if type(p_gen) is torch.Tensor:
        p_gen = p_gen.cpu().detach().numpy()
    return (p_target - p_gen) / (p_target + alpha*np.median(p_target) + eps)


def get_bins(
    num_bins,
    cartesian_min_max=((-1, 10), (-10, 10), (-10, 10), (-10, 10)),
    polar_min_max=((-1, 10), (-1, 1.5), (-15, 15), (-15, 15)),
):
    """Get bins for jet reconstruction error plots."""
    ranges_cartesian = tuple([
        np.linspace(*cartesian_min_max[i], num_bins)
        for i in range(len(cartesian_min_max))
    ])

    ranges_polar = tuple([
        np.linspace(*polar_min_max[i], num_bins)
        for i in range(len(polar_min_max))
    ])

    ranges = (ranges_cartesian, ranges_polar)
    return ranges


def get_legend(res):
    """Get legend for plots of jet reconstruction."""
    legend = r'$\mu$: ' + f'{np.mean(res) :.4f},\n'
    legend += r'$\sigma$: ' + f'{np.std(res) :.4f},\n'
    legend += r'$\mathrm{Med}$: ' + f'{np.median(res) :.4f}'
    return legend
