import torch
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
from utils.utils import make_dir

FIGSIZE = (12, 8)
LABELS = ((r'$p_x$', r'$p_y$', r'$p_z$'), (r'$p_\mathrm{T}$', r'$\eta$', r'$\phi$'))


def plot_particle_recon_err(args, p_target, p_gen,
                            get_rel_err=(lambda p_target, p_gen, eps: (p_target-p_gen)/(p_target+0.01*np.median(p_target)+eps)),
                            ranges=None, eps=1e-16, save_dir=None, epoch=None, show=False):
    """Plot the error for reconstruction of particle features.
        - For real/nonpadded particles, relative error will be plotted.
        - For padded particles, distribution will be plotted.

    Parameters
    ----------
    p_target : `torch.Tensor` or `numpy.ndarray`
        Target particle momenta, in Cartesian coordinates, of target real/nonpadded particles.
    p_gen : `torch.Tensor` or `numpy.ndarray`
        Reconstructed particle momenta, in Cartesian coordinates, of real/nonpadded particles.
    get_rel_err : function
        The function for computing relative errors.
        The first argument should be a component of the target,
        and the second should be the same component of the reconstructed feature.
    ranges : iterable of iterable of iterable of iterable of np.ndarray
        Ranges of plots: ((ranges_real_cartesian, ranges_padded_cartesian), (ranges_real_target, ranges_padded_target)),
        where each of ranges_real_cartesian, ranges_padded_cartesian, ranges_real_target, ranges_padded_target is a tuple of numpy.ndarray.
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
    p_real_target_array, _, mask = split(p_target, return_mask=True)
    p_gen = p_gen.view(-1, p_gen.shape[-1])
    if type(p_gen) is torch.Tensor:
        p_gen = p_gen.cpu().detach().numpy()
    p_real_gen_array = p_gen[mask]
    p_padded_gen_array = p_gen[~mask]

    p_real_target_cartesian = convert_to_tuple(p_real_target_array)
    p_real_gen_cartesian = convert_to_tuple(p_real_gen_array)
    p_padded_gen_cartesian = convert_to_tuple(p_padded_gen_array)

    p_real_target_polar = convert_to_polar(p_real_target_cartesian, eps)
    p_real_gen_polar = convert_to_polar(p_real_gen_cartesian, eps)
    p_padded_gen_target = convert_to_polar(p_padded_gen_cartesian, eps)

    if ranges is None:
        ranges = get_bins(args.num_bins)

    # Plot both Cartesian and polar coordinates
    for p_real_gen, p_real_target, p_padded_gen, coordinate, bin_tuple, labels in zip(
        (p_real_target_cartesian, p_real_target_polar),
        (p_real_gen_cartesian, p_real_gen_polar),
        (p_padded_gen_cartesian, p_padded_gen_target),
        ('cartesian', 'polar'),
        ranges,
        LABELS
    ):
        ranges_real, ranges_padded = bin_tuple
        fig, axs = plt.subplots(2, 3, figsize=FIGSIZE, sharey=False)

        for ax, out, target, bins, label in zip(axs[0], p_real_gen, p_real_target, ranges_real, labels):
            res = get_rel_err(target.reshape(-1), out.reshape(-1), eps=eps)
            ax.hist(res, bins=bins, label=get_legend_rel_err(res), histtype='step', stacked=True)
            ax.set_xlabel(fr'$\delta${label}')
            ax.set_ylabel('Number of real particles')
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
            ax.tick_params(bottom=True, top=True, left=True, right=True, direction='in')
            ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
            ax.legend(framealpha=0.6, loc='upper right')

        for ax, p, bins, label in zip(axs[1], p_padded_gen, ranges_padded, labels):
            ax.hist(p, histtype='step', stacked=True, bins=bins, label=get_legend_padded(p))
            ax.set_xlabel(f'Reconstructed padded {label} (GeV)')
            ax.set_ylabel('Number of padded particles')
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
            ax.tick_params(bottom=True, top=True, left=True, right=True, direction='in')
            ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
            ax.legend(framealpha=0.6, loc='upper right')

        plt.tight_layout()

        if save_dir:
            path = make_dir(osp.join(save_dir, f'particle_reconstruction_errors/{coordinate}'))
            if epoch is not None:
                plt.savefig(osp.join(path, f'particle_reconstruction_errors_epoch_{epoch+1}.pdf'))
            else:
                plt.savefig(osp.join(path, 'particle_reconstruction_errors.pdf'))
        if show:
            plt.show()
        plt.close()


def get_bins(
    num_bins,
    cartesian_real_min_max=((-20, 20),)*3,
    cartesian_padded_min_max=((-200, 200),)*3,
    polar_real_min_max=((-1.5, 10), (-20, 20), (-20, 20)),
    polar_padded_min_max=((0, 150), (-2, 2), (-np.pi, np.pi))
):
    """Get bins for reconstruction error plots."""
    ranges_cartesian_real = tuple([
        np.linspace(*cartesian_real_min_max[i], num_bins)
        for i in range(len(cartesian_real_min_max))
    ])
    ranges_cartesian_padded = tuple([
        np.linspace(*cartesian_padded_min_max[i], num_bins)
        for i in range(len(cartesian_padded_min_max))
    ])
    ranges_cartesian = (ranges_cartesian_real, ranges_cartesian_padded)

    ranges_polar_real = tuple([
        np.linspace(*polar_real_min_max[i], num_bins)
        for i in range(len(polar_real_min_max))
    ])

    ranges_polar_padded = tuple([
        np.linspace(*polar_padded_min_max[i], num_bins)
        for i in range(len(polar_padded_min_max))
    ])
    ranges_polar = (ranges_polar_real, ranges_polar_padded)

    ranges = (ranges_cartesian, ranges_polar)
    return ranges


def split(p, return_mask=False):
    """Split particles into real (nonpadded) and padded particles.

    Parameters
    ----------
    p : `torch.Tensor` or `numpy.ndarray`
        The collection of particle momenta in Cartesian coordinates.
        Shape: (num_particles, 3) or (num_particles, 4) or (num_particles, 3) or
               (num_jets, num_particles, 3) or (num_jets, num_particles, 4)
    return_mask : bool
        Whether to return mask.
        Default: True.

    Returns
    -------
    (p_real, p_padded) if return_mask else (p_real, p_padded, mask)

    Raises
    ------
    TypeError
        If p is not `torch.tensor` or `numpy.ndarray`.
    ValueError
        If the last dimension of p is not 3 or 4.
    """
    p = p.view(-1, p.shape[-1])  # Remove batching

    # Convert to numpy.ndarray
    if type(p) is torch.Tensor:
        p = p.detach().cpu().numpy()
    elif type(p) is np.ndarray:
        pass
    else:
        raise TypeError(f'type(p) needs to be torch.Tensor or numpy.ndarray. Found: {type(p)}')

    # Get mask
    if p.shape[-1] == 4:
        mask = (p[:, 1] != 0) & (p[:, 2] != 0) & (p[:, 3] != 0)
    elif p.shape[-1] == 3:
        mask = (p[:, 0] != 0) & (p[:, 1] != 0) & (p[:, 2] != 0)
    else:
        raise ValueError(f'Invalid dimension of particle momenta. Choice: (3, 4). Found: {p.shape[-1]}.')

    p_real = p[mask]
    p_padded = p[~mask]

    if not return_mask:
        return p_real, p_padded
    else:
        return p_real, p_padded, mask


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


def convert_to_tuple(p):
    """Convert momenta in numpy.ndarray to component-wise tuple form."""
    return tuple(p[:, i, ...] for i in range(p.shape[-1]))


def convert_to_polar(p_cartesian, eps):
    if len(p_cartesian) == 4:
        _, px, py, pz = p_cartesian
    elif len(p_cartesian) == 3:
        px, py, pz = p_cartesian
    else:
        raise ValueError(f'len(p_cartesian) must be 3 or 4. Found: {len(p_cartesian)}')

    pt = np.sqrt(px ** 2 + py ** 2 + eps)
    eta = np.arcsinh(pz / pt)
    phi = np.arctan2(py, px + eps)

    return pt, eta, phi
