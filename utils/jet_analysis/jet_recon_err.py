import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import make_dir
from utils.jet_analysis.utils import NUM_BINS
import os.path as osp

FIGSIZE = (16, 4)
LABELS_CARTESIAN_ABS_COORD = (r'$M$', r'$P_x$', r'$P_y$', r'$P_z$')
LABELS_POLAR_ABS_COORD = (r'$M$', r'$P_\mathrm{T}$', r'$\eta$', r'$\phi$')
LABELS_ABS_COORD = (LABELS_CARTESIAN_ABS_COORD, LABELS_POLAR_ABS_COORD)
LABELS_CARTESIAN_REL_COORD = (r'$M^\mathrm{rel}$', r'$P_x^\mathrm{rel}$', r'$P_y^\mathrm{rel}$', r'$P_z^\mathrm{rel}$')
LABELS_POLAR_REL_COORD = (r'$M^\mathrm{rel}$', r'$P_\mathrm{T}^\mathrm{rel}$', r'$\eta^\mathrm{rel}$', r'$\phi^\mathrm{rel}$')
LABELS_REL_COORD = (LABELS_CARTESIAN_REL_COORD, LABELS_POLAR_REL_COORD)
COORDINATES = ('cartesian', 'polar')
DEFAULT_BIN_RANGE = 2
MAX_BIN_RANGE = 5


def plot_jet_recon_err(args, jet_target_cartesian, jet_gen_cartesian, jet_target_polar, jet_gen_polar,
                       save_dir, epoch=None, eps=1e-16, drop_zeros=True,
                       get_rel_err=(lambda p_target, p_gen, eps: (p_target-p_gen)/(p_target+eps)),
                       show=False):
    """Plot reconstruction errors for jet."""
    if drop_zeros:
        jet_target_cartesian, jet_gen_cartesian = filter_out_zeros(jet_target_cartesian, jet_gen_cartesian)
        jet_target_polar, jet_gen_polar = filter_out_zeros(jet_target_polar, jet_gen_polar)

    rel_err_cartesian = [get_rel_err(jet_gen_cartesian[i], jet_target_cartesian[i], eps) for i in range(4)]
    rel_err_polar = [get_rel_err(jet_gen_polar[i], jet_target_polar[i], eps) for i in range(4)]
    ranges = get_bins(NUM_BINS, rel_err_cartesian=rel_err_cartesian, rel_err_polar=rel_err_polar)

    LABELS = LABELS_ABS_COORD if args.abs_coord else LABELS_REL_COORD
    for rel_err_coordinate, labels, coordinate, bin_tuple in zip((rel_err_cartesian, rel_err_polar), LABELS, COORDINATES, ranges):
        fig, axs = plt.subplots(1, 4, figsize=FIGSIZE, sharey=False)
        for ax, rel_err, bins, label in zip(axs, rel_err_coordinate, bin_tuple, labels):
            ax.hist(rel_err, bins=bins, label=get_legend(rel_err, bins=bins), histtype='step', stacked=True)
            ax.set_xlabel(fr'$\delta${label}')
            ax.set_ylabel('Number of Jets')
            ax.legend()
        plt.tight_layout()
        if save_dir:
            if epoch is not None:
                path = make_dir(osp.join(save_dir, f'jet_reconstruction_errors/{coordinate}'))
                plt.savefig(osp.join(path, f'jet_reconstruction_errors_epoch_{epoch+1}.pdf'))
            else:  # Save without creating a subdirectory
                plt.savefig(osp.join(save_dir, f'jet_reconstruction_errors_{coordinate}.pdf'))
        if show:
            plt.show()
        plt.close()


def default_get_rel_err(p_target, p_gen, eps, alpha=0.01):
    if type(p_target) is torch.Tensor:
        p_target = p_target.cpu().detach().numpy()
    if type(p_gen) is torch.Tensor:
        p_gen = p_gen.cpu().detach().numpy()
    return (p_target - p_gen) / (p_target + alpha*np.median(p_target) + eps)


def get_bins(num_bins, rel_err_cartesian=None, rel_err_polar=None):
    """Get bins for jet reconstruction error plots."""
    if rel_err_cartesian is None:
        cartesian_min_max = ((-1, 10), (-DEFAULT_BIN_RANGE, DEFAULT_BIN_RANGE), (-DEFAULT_BIN_RANGE, DEFAULT_BIN_RANGE), (-DEFAULT_BIN_RANGE, DEFAULT_BIN_RANGE))
    else:
        mass_min_max = (-min(10 * np.std(rel_err_cartesian[0]), 1), min(2 * np.std(rel_err_cartesian[0]), 2 * MAX_BIN_RANGE))
        px_min_max = (-min(np.std(rel_err_cartesian[1]), MAX_BIN_RANGE), min(np.std(rel_err_cartesian[1]), MAX_BIN_RANGE))
        py_min_max = (-min(np.std(rel_err_cartesian[2]), MAX_BIN_RANGE), min(np.std(rel_err_cartesian[2]), MAX_BIN_RANGE))
        pz_min_max = (-min(np.std(rel_err_cartesian[3]), MAX_BIN_RANGE), min(np.std(rel_err_cartesian[3]), MAX_BIN_RANGE))
        cartesian_min_max = (mass_min_max, px_min_max, py_min_max, pz_min_max)

    if rel_err_polar is None:
        polar_min_max = ((-1, DEFAULT_BIN_RANGE), (-1, DEFAULT_BIN_RANGE), (-DEFAULT_BIN_RANGE, DEFAULT_BIN_RANGE), (-DEFAULT_BIN_RANGE, DEFAULT_BIN_RANGE))
    else:
        mass_min_max = (-min(10 * np.std(rel_err_polar[0]), 1), min(2 * np.std(rel_err_polar[0]), 2 * MAX_BIN_RANGE))
        pt_min_max = (-min(10 * np.std(rel_err_polar[1]), 1), min(2 * np.std(rel_err_polar[1]), 2 * MAX_BIN_RANGE))
        eta_min_max = (-min(np.std(rel_err_polar[2]), MAX_BIN_RANGE), min(np.std(rel_err_polar[2]), MAX_BIN_RANGE))
        phi_min_max = (-min(np.std(rel_err_polar[3]), MAX_BIN_RANGE), min(np.std(rel_err_polar[3]), MAX_BIN_RANGE))
        polar_min_max = (mass_min_max, pt_min_max, eta_min_max, phi_min_max)

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


def get_legend(res, bins):
    """Get legend for plots of jet reconstruction."""
    legend = r'$\mu$: ' + f'{np.mean(res) :.4f},\n'
    legend += r'$\mathrm{FWHM}$: ' + f'{find_fwhm(res, bins) :.4f}'
    # legend += r'$\mathrm{Med}$: ' + f'{np.median(res) :.4f}'
    return legend


def filter_out_zeros(target, gen):
    """Filter out jets with any zero component.

    Parameters
    ----------
    target : iterable of `numpy.ndarray`.
        Target jet components.
    gen : iterable of `numpy.ndarray`.
        Generated/reconstructed jet components.

    Returns
    -------
    target_filtered, gen_filtered
    """
    mask = (target[0] != 0) & (target[1] != 0) & (target[2] != 0) & (target[3] != 0)
    target_filtered = tuple([target[i][mask] for i in range(4)])
    gen_filtered = tuple([gen[i][mask] for i in range(4)])
    return target_filtered, gen_filtered


def find_fwhm(err, bins):
    """Full width at half maximum of a distribution."""
    hist, _ = np.histogram(err, bins=bins)
    max_idx = np.argmax(hist)
    peak = bins[max_idx]

    half_max = hist[max_idx] / 2
    half_max_idx = (np.abs(hist - half_max)).argmin()
    half_peak = bins[half_max_idx]

    return 2 * abs(peak - half_peak)
