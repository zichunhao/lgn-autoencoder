from utils.jet_analysis.utils import get_jet_name
from utils.utils import make_dir
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

PARTICLE_FEATURE_FIGSIZE = (12, 4)


def plot_p_cartesian(args, p_target, p_gen, save_dir, max_val=[100, 100, 100],
                     num_bins=201, epoch=None, density=False, fill=False, show=False):
    """Plot p distribution in Cartesian coordinates.

    Args
    ----
    p_target : `numpy.ndarray`
        The target jet data, with shape (num_particles, 4).
    p_gen : `numpy.ndarray`
        The reconstructed jet data, with shape (num_particles, 4).
    save_dir : `str`
        The directory to save the figure.
    max_val :  `list`, `tuple`, or `int`
        The maximum values of (px, py, pz) in the plot.
        Optional, default: `201`
    num_bins : `int`
        The number of bins in the histogram.
        Optional, default: `(0.02, 0.02, 0.02)`
    num_bins : `int`
        The number of bins in the histogram.
    epoch : `int`
        The epoch number of the evaluated model.
        Optional, default: `None`
    density : `bool`
        Whether to plot distribution density instead of absolute number.
        Optional, default: `False`
    fill : `bool`
        Whether bins are filled.
        Optional, default: `False`
    show : `bool`
        Whether to show plot.
        Optional, default: `False`
    """

    px_target, py_target, pz_target = p_target
    px_gen, py_gen, pz_gen = p_gen

    fig, axs = plt.subplots(1, 3, figsize=PARTICLE_FEATURE_FIGSIZE, sharey=False)
    if type(max_val) in [tuple, list]:
        if len(max_val) == 3:
            px_max, py_max, pz_max = max_val
        else:
            px_max, py_max, pz_max = max_val[0]
    elif type(max_val) in [float, int]:
        px_max = py_max = pz_max = float(max_val)
    else:
        px_max = py_max = pz_max = max_val = 0.02

    p_targets = [px_target, py_target, pz_target]
    p_gens = [px_gen, py_gen, pz_gen]
    ranges = [np.linspace(-px_max, px_max, num_bins),
              np.linspace(-py_max, py_max, num_bins),
              np.linspace(-pz_max, pz_max, num_bins)]
    names = [r'$p_x$', r'$p_y$', r'$p_z$']
    for ax, p_target, p_gen, bins, name in zip(axs, p_targets, p_gens, ranges, names):
        if not fill:
            ax.hist(p_gen.flatten(), bins=bins, histtype='step', stacked=True,
                    fill=False, label='reconstructed', density=density)
            ax.hist(p_target.flatten(), bins=bins, histtype='step',
                    stacked=True, fill=False, label='target', density=density)
        else:
            ax.hist(p_gen.flatten(), bins=bins, alpha=0.6, label='reconstructed', density=density)
            ax.hist(p_target.flatten(), bins=bins, alpha=0.6, label='target', density=density)
        ax.set_xlabel(f'Particle {name} (GeV)')
        ax.set_ylabel('Number of particles')
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        ax.tick_params(bottom=True, top=True, left=True, right=True, direction='in')
        ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2)

    fig.tight_layout()

    jet_name = get_jet_name(args)
    fig.suptitle(fr'Distribution of target and reconstructed particle $p_x$, $p_y$, and $p_z$ of {jet_name} jets', y=1.03)

    if fill:
        save_dir = osp.join(save_dir, 'filled')
    save_dir = make_dir(osp.join(save_dir, 'particle_cartesian'))

    filename = f'p_cartesian_{args.jet_type}_jet'
    if epoch is not None:
        filename = f'{filename}_epoch_{epoch+1}'
    if density:
        filename = f'{filename}_density'
    plt.savefig(osp.join(save_dir, f'{filename}.pdf'), bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_p_polar(args, p_polar_target, p_polar_gen, save_dir, max_val=(200, 2, np.pi),
                 num_bins=201, epoch=None, density=False, fill=True, show=False):
    """Plot p distribution in polar coordinates (pt, eta, phi)

    Args
    ----
    p_polar_target : tuple of 3 `numpy.ndarray`
        The target jet data in polar coordinates (pt, eta, phi).
    p_polar_gen : tuple of 3 `numpy.ndarray`
        The reconstructed jet data in polar coordinates (pt, eta, phi).
    save_dir : `str`
        The directory to save the figure.
    max_val : `list`, `tuple`, or `float`
        The maximum values of (pt, eta, phi) in the plot.
        Optional, default: `(0.15, np.pi, np.pi)`
    num_bins : `int`
        The number of bins in the histogram.
        Optional, default: `201`
    epoch : `int`
        The epoch number of the evaluated model.
        Optional, default: `None`
    density : `bool`
        Whether to plot distribution density instead of absolute number.
        Optional, default: `False`
    fill : `bool`
        Whether bins are filled.
        Optional, default: `False`
    show : `bool`
        Whether to show plot.
        Optional, default: `False`
    """

    pt_target, eta_target, phi_target = p_polar_target
    pt_gen, eta_gen, phi_gen = p_polar_gen

    fig, axs = plt.subplots(1, 3, figsize=PARTICLE_FEATURE_FIGSIZE, sharey=False)

    if type(max_val) in [tuple, list]:
        if len(max_val) == 3:
            pt_max, eta_max, phi_max = max_val
        elif len(max_val) == 2:
            pt_max = max_val[0]
            eta_max = phi_max = max_val[1]
    elif type(max_val) in [float, int]:
        pt_max = eta_max = phi_max = float(max_val)
    else:
        pt_max = 200
        eta_max = 2
        phi_max = np.pi

    ranges = [np.linspace(0, pt_max, num_bins),
              np.linspace(-eta_max, eta_max, num_bins),
              np.linspace(-phi_max, phi_max, num_bins)]
    names = [r'$p_\mathrm{T}$', r'$\eta$', r'$\phi$']
    for ax, p_target, p_gen, bins, name in zip(axs, p_polar_target, p_polar_gen, ranges, names):
        if not fill:
            ax.hist(p_gen.flatten(), bins=bins, histtype='step', stacked=True,
                    fill=False, label='reconstructed', density=density)
            ax.hist(p_target.flatten(), bins=bins, histtype='step',
                    stacked=True, fill=False, label='target', density=density)
        else:
            ax.hist(p_gen.flatten(), bins=bins, alpha=0.6, label='reconstructed', density=density)
            ax.hist(p_target.flatten(), bins=bins, alpha=0.6, label='target', density=density)
        ax.set_xlabel(f'Particle {name}')
        if name == r'$p_\mathrm{T}$':
            ax.set_xlabel(f'Particle {name} (GeV)')
        ax.set_ylabel('Number of particles')
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        ax.tick_params(bottom=True, top=True, left=True, right=True, direction='in')
        ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4)

    fig.tight_layout()

    jet_name = get_jet_name(args)
    fig.suptitle(r'Distribution of target and reconstructed particle $p_\mathrm{T}$, $\eta$, and $\phi$ ' +
                 f'of {jet_name} jets', y=1.03)

    if fill:
        save_dir = osp.join(save_dir, 'filled')
    save_dir = make_dir(osp.join(save_dir, 'particle_polar'))

    filename = f'p_polar_{args.jet_type}_jet'
    if epoch is not None:
        filename = f'{filename}_epoch_{epoch+1}'
    if density:
        filename = f'{filename}_density'
    plt.savefig(osp.join(save_dir, f'{filename}.pdf'), bbox_inches="tight", transparent=True)
    if show:
        plt.show()
    plt.close()
