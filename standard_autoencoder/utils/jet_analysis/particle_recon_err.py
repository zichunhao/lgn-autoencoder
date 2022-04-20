import torch
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
from utils.utils import make_dir
from .utils import get_p_polar_tensor, get_stats, NUM_BINS, PLOT_FONT_SIZE, DEVICE
import scipy.optimize as optimize
import logging
import json

FIGSIZE = (12, 8)
LABELS_ABS_COORD = ((r'$p_x$', r'$p_y$', r'$p_z$'), (r'$p_\mathrm{T}$', r'$\eta$', r'$\phi$'))
LABELS_REL_COORD = ((r'$p_x^\mathrm{rel}$', r'$p_y^\mathrm{rel}$', r'$p_z^\mathrm{rel}$'),
                    (r'$p_\mathrm{T}^\mathrm{rel}$', r'$\eta^\mathrm{rel}$', r'$\phi^\mathrm{rel}$'))


def plot_particle_recon_err(args, p_target, p_gen, find_match=True, ranges=None,
                            eps=1e-16, save_dir=None, epoch=None, show=False):
    """Plot the error for reconstruction of particle features.
        - For real/nonpadded particles, relative error will be plotted.
        - For padded particles, distribution will be plotted.

    Parameters
    ----------
    p_target : `torch.Tensor` or `numpy.ndarray`
        Target particle momenta, in Cartesian coordinates, of target real/nonpadded particles.
    p_gen : `torch.Tensor` or `numpy.ndarray`
        Reconstructed particle momenta, in Cartesian coordinates, of real/nonpadded particles.
    find_match : bool, optional
        Whether matching needs to be done. Used for permutation invariant loss.
        Default: True
    ranges : iterable of iterable of iterable of iterable of np.ndarray
        Ranges of plots: ((ranges_rel_err_cartesian, ranges_padded_recons_cartesian), (ranges_rel_err_polar, ranges_padded_recons_polar)),
        where each of ranges_rel_err_cartesian, ranges_padded_recons_cartesian,
        ranges_rel_err_polar, and ranges_padded_recons_polar is a tuple of numpy.ndarray.
    eps : float, optional
        Default: 1e-16
    save_dir : str, optional
        Default: None
    epoch : None or int, optional
        Default: None
    show : bool, optional
        Whether to show plot.
        Default: False
    """

    # Get inputs
    p_target_cartesian = p_target if (p_target.shape[-1] == 3) else p_target[..., 1:]
    p_gen_cartesian = p_gen if (p_gen.shape[-1] == 3) else p_gen[..., 1:]
    p_target_polar = get_p_polar_tensor(p_target)
    p_gen_polar = get_p_polar_tensor(p_gen)

    if not find_match:
        rel_err_cartesian = get_rel_err(p_target_cartesian, p_gen_cartesian).view(-1, 3)
        rel_err_polar = get_rel_err(p_target_polar, p_gen_polar).view(-1, 3)
    else:
        rel_err_cartesian, rel_err_polar = get_rel_err_find_match(
            p_target_cartesian, p_gen_cartesian, p_target_polar, p_gen_polar
        )

    is_padded = torch.any(rel_err_cartesian.isinf(), dim=-1)
    # Relative error for real/nonpadded particles
    rel_err_cartesian = rel_err_cartesian[~is_padded]
    rel_err_polar = rel_err_polar[~is_padded]

    # Padded particle features
    p_padded_recons_cartesian = p_gen_cartesian.view(-1, 3)[is_padded]
    p_padded_recons_polar = p_gen_polar.view(-1, 3)[is_padded]

    LABELS = LABELS_ABS_COORD if args.abs_coord else LABELS_REL_COORD
    if ranges is None:
        ranges = get_bins(NUM_BINS,
                          rel_err_cartesian=rel_err_cartesian.numpy(),
                          rel_err_polar=rel_err_polar.numpy(),
                          p_padded_recons_cartesian=p_padded_recons_cartesian.numpy(),
                          p_padded_recons_polar=p_padded_recons_polar.numpy())
        custom_range = False
    else:
        custom_range = True

    # Plot both Cartesian and polar coordinates
    err_dict = dict()
    for rel_err, p_padded_recons, coordinate, bin_tuple, labels in zip(
        (rel_err_cartesian, rel_err_polar),
        (p_padded_recons_cartesian, p_padded_recons_polar),
        ('cartesian', 'polar'),
        ranges,
        LABELS
    ):
        err_dict_coordinate = {
            'rel_err': [],
            'pad_recons': []
        }
        ranges_real, ranges_padded = bin_tuple
        fig, axs = plt.subplots(2, 3, figsize=FIGSIZE, sharey=False)

        for i, (ax, bins, label) in enumerate(zip(axs[0], ranges_real, labels)):
            res = rel_err[..., i].numpy()
            stats = get_stats(res, bins)
            err_dict_coordinate['rel_err'].append(stats)

            if not custom_range:
                # Find the range based on the FWHM
                FWHM = stats['FWHM']
                bins_suitable = np.linspace(-1.5*FWHM, 1.5*FWHM, NUM_BINS)
                ax.hist(res, bins=bins_suitable, histtype='step', stacked=True)
            else:
                ax.hist(res, bins=bins, histtype='step', stacked=True)

            ax.set_xlabel(fr'$\delta${label}')
            ax.set_ylabel('Number of real particles')
            ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 0), useMathText=True)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
            ax.tick_params(bottom=True, top=True, left=True, right=True, direction='in')
            ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
            for axis in ('x', 'y'):
                ax.tick_params(axis=axis, labelsize=PLOT_FONT_SIZE)

        for i, (ax, bins, label) in enumerate(zip(axs[1], ranges_padded, labels)):
            p = p_padded_recons[..., i].numpy()

            stats = get_stats(p, bins)
            err_dict_coordinate['pad_recons'].append(stats)

            if not custom_range:
            # Find the range based on the FWHM
                FWHM = stats['FWHM']
                bins_suitable = np.linspace(-1.5*FWHM, 1.5*FWHM, NUM_BINS)
                ax.hist(p, histtype='step', stacked=True, bins=bins_suitable)
            else:
                ax.hist(p, histtype='step', stacked=True, bins=bins)

            if args.abs_coord:
                if ('eta' in label.lower()) or ('phi' in label.lower()):
                    # eta and phi are dimensionless
                    ax.set_xlabel(f'Reconstructed padded {label}')
                else:
                    ax.set_xlabel(f'Reconstructed padded {label} (GeV)')
            else:  # relative coordinates are normalized and dimensionless
                ax.set_xlabel(f'Reconstructed padded {label}')
            ax.set_ylabel('Number of padded particles')
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
            for axis in ('x', 'y'):
                ax.tick_params(axis=axis, labelsize=PLOT_FONT_SIZE)
            ax.tick_params(bottom=True, top=True, left=True, right=True, direction='in')
            ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)

        err_dict[coordinate] = err_dict_coordinate

        plt.rcParams.update({'font.size': PLOT_FONT_SIZE})
        plt.tight_layout()

        if save_dir:
            if epoch is not None:
                path = make_dir(osp.join(save_dir, f'particle_reconstruction_errors/{coordinate}'))
                plt.savefig(osp.join(path, f'particle_reconstruction_errors_epoch_{epoch+1}.pdf'))
                dict_path = make_dir(osp.join(path, 'err_dict'))
                file_name = f'particle_reconstruction_errors_epoch_{epoch+1}.json'
                with open(osp.join(dict_path, file_name), 'w') as f:
                    json.dump(str(err_dict), f)
            else:  # Save without creating a subdirectory
                plt.savefig(osp.join(save_dir, f'particle_reconstruction_errors_{coordinate}.pdf'))
                dict_path = osp.join(save_dir, 'particle_reconstruction_errors.json')
                with open(dict_path, 'w') as f:
                    json.dump(str(err_dict), f)
        if show:
            plt.show()
        plt.close()

        logging.debug('Particle feature reconstruction error:')
        logging.debug(err_dict)


def get_rel_err_find_match(p_target_cartesian, p_recons_cartesian, p_target_polar, p_recons_polar, gpu=True):
    """Get relative errors after finding the match beween target and reconstructed/generated jet."""
    if gpu:
        p_target_cartesian = p_target_cartesian.to(DEVICE)
        p_recons_cartesian = p_recons_cartesian.to(DEVICE)
        p_target_polar = p_target_polar.to(DEVICE)
        p_recons_polar = p_recons_polar.to(DEVICE)
    cost = torch.cdist(p_target_cartesian, p_recons_cartesian).cpu().numpy()

    rel_err_cartesian_list = []
    rel_err_polar_list = []
    for i in range(len(p_target_cartesian)):
        matching = optimize.linear_sum_assignment(cost[i])
        rel_err_cartesian = (p_recons_cartesian[i][matching[1]] - p_target_cartesian[i]) / p_target_cartesian[i]
        rel_err_cartesian_list.append(rel_err_cartesian)

        rel_err_polar = (p_recons_polar[i][matching[1]] - p_target_polar[i]) / p_target_polar[i]
        rel_err_polar_list.append(rel_err_polar)

    rel_err_cartesian = torch.stack(rel_err_cartesian_list).view(-1, 3).cpu()
    rel_err_polar = torch.stack(rel_err_polar_list).view(-1, 3).cpu()

    return rel_err_cartesian, rel_err_polar


def get_min_max(err, alpha=1.5):
    num_components = err.shape[-1]
    means = [np.mean(err[..., i]) for i in range(num_components)]
    std_devs = [np.std(err[..., i]) for i in range(num_components)]
    return tuple([
        (means[i] - alpha * std_devs[i], means[i] + alpha * std_devs[i])
        for i in range(num_components)
    ])


def get_bins(num_bins, rel_err_cartesian=None, rel_err_polar=None,
             p_padded_recons_cartesian=None, p_padded_recons_polar=None):
    """Get bins for reconstruction error plots."""
    if rel_err_cartesian is None:
        cartesian_real_min_max = ((-20, 20),)*3,
    else:
        cartesian_real_min_max = get_min_max(rel_err_cartesian)

    if p_padded_recons_cartesian is None:
        cartesian_padded_min_max = ((-200, 200),)*3,
    else:
        cartesian_padded_min_max = get_min_max(rel_err_cartesian)

    if rel_err_polar is None:
        polar_real_min_max = ((-1.5, 10), (-20, 20), (-20, 20)),
    else:
        polar_real_min_max = get_min_max(rel_err_polar)

    if p_padded_recons_polar is None:
        polar_padded_min_max = ((0, 150), (-2, 2), (-np.pi, np.pi))
    else:
        polar_padded_min_max = get_min_max(p_padded_recons_polar)

    ranges_cartesian_real = tuple([
        np.linspace(*cartesian_real_min_max[i])
        for i in range(len(cartesian_real_min_max))
    ])
    ranges_cartesian_padded = tuple([
        np.linspace(*cartesian_padded_min_max[i])
        for i in range(len(cartesian_padded_min_max))
    ])
    ranges_cartesian = (ranges_cartesian_real, ranges_cartesian_padded)

    ranges_polar_real = tuple([
        np.linspace(*polar_real_min_max[i])
        for i in range(len(polar_real_min_max))
    ])

    ranges_polar_padded = tuple([
        np.linspace(*polar_padded_min_max[i])
        for i in range(len(polar_padded_min_max))
    ])
    ranges_polar = (ranges_polar_real, ranges_polar_padded)

    ranges = (ranges_cartesian, ranges_polar)
    return ranges


def get_rel_err(target, recons):
    return ((recons - target) / target).view(-1, target.shape[-1])


def get_legend_rel_err(res):
    """Get legend for plots of real/nonpadded particle reconstruction."""
    legend = r'$\mu$: ' + f'{np.mean(res) :.4f},\n'
    legend += r'$\sigma$: ' + f'{np.std(res) :.4f},\n'
    legend += r'$\mathrm{Med}$: ' + f'{np.median(res) :.4f}'
    return legend


def get_legend_padded(p):
    """Get legend for plots of padded particle reconstruction."""
    legend = r'$\mu$: ' + f'{np.mean(p) :.5f} GeV,\n'
    legend += r'$\sigma$: ' + f'{np.std(p) :.5f} GeV,\n'
    legend += r'$\mathrm{Med}$: ' + f'{np.median(p) :.5f} GeV'
    return legend
