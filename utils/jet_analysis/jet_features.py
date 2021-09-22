from utils.jet_analysis.utils import get_jet_name, NUM_BINS
from utils.utils import make_dir
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

FIGSIZE = (16, 4)
LABELS_CARTESIAN_ABS_COORD = (r'$M$ (GeV)', r'$P_x$ (GeV)', r'$P_y$ (GeV)', r'$P_z$ (GeV)')
LABELS_CARTESIAN_REL_COORD = (r'$M^\mathrm{rel}$', r'$P_x^\mathrm{rel}$', r'$P_y^\mathrm{rel}$', r'$P_z^\mathrm{rel}$')
LABELS_POLAR_ABS_COORD = (r'$M$ (GeV)', r'$P_\mathrm{T}$ (GeV)', r'$\eta$', r'$\phi$')
LABELS_POLAR_REL_COORD = (r'$M^\mathrm{rel}$', r'$P_\mathrm{T}^\mathrm{rel}$', r'$\eta^\mathrm{rel}$', r'$\phi^\mathrm{rel}$')

RANGES_CARTESIAN_ABS_COORD = (
    np.linspace(0, 250, NUM_BINS),       # Jet M
    np.linspace(-2000, 2000, NUM_BINS),  # Jet Px
    np.linspace(-2000, 2000, NUM_BINS),  # Jet Py
    np.linspace(-4000, 4000, NUM_BINS)   # Jet Pz
)
RANGES_CARTESIAN_REL_COORD = (
    np.linspace(0, 0.2, NUM_BINS),    # Jet M_rel
    np.linspace(0.5, 1.02, NUM_BINS),    # Jet Px_rel
    np.linspace(-0.02, 0.02, NUM_BINS),  # Jet Px_rel
    np.linspace(-0.02, 0.02, NUM_BINS)   # Jet Pz_rel
)
RANGES_POLAR_ABS_COORD = (
    np.linspace(0, 250, NUM_BINS),        # Jet M
    np.linspace(0, 4000, NUM_BINS),       # Jet Pt
    np.linspace(-2, 2, NUM_BINS),         # Jet eta
    np.linspace(-np.pi, np.pi, NUM_BINS)  # Jet phi
)
RANGES_POLAR_REL_COORD = (
    np.linspace(0, 0.2, NUM_BINS),    # Jet M_rel
    np.linspace(0.5, 1.02, NUM_BINS),    # Jet Pt_rel
    np.linspace(-0.02, 0.02, NUM_BINS),  # Jet eta_rel
    np.linspace(-0.02, 0.02, NUM_BINS)   # Jet phi_rel
)


def plot_jet_p_cartesian(args, jet_features_target, jet_features_gen, save_dir,
                         epoch=None, density=False, fill=True, show=False):
    """Plot jet features (m, px, py, pz) distribution.

    Parameters
    ----------
    jet_features_target : `numpy.ndarray`
        The target jet momenta, with shape (num_jets, 4).
    jet_features_gen : `numpy.ndarray`
        The generated/reconstructed jet momenta, with shape (num_jets, 4).
    save_dir : str
        The directory to save the figure.
    epoch : int
        The epoch number of the evaluated model.
        Optional, default: `None`
    density : bool
        Whether to plot distribution density instead of absolute number.
        Optional, default: `False`
    fill : bool
        Whether bins are filled.
        Optional, default: `False`
    show : bool
        Whether to show plot.
        Optional, default: `False`
    """
    if args.abs_coord:
        ranges = RANGES_CARTESIAN_ABS_COORD
        names = LABELS_CARTESIAN_ABS_COORD
    else:
        ranges = RANGES_CARTESIAN_REL_COORD
        names = LABELS_CARTESIAN_REL_COORD

    fig, axs = plt.subplots(1, 4, figsize=FIGSIZE, sharey=False)
    for ax, p_target, p_gen, bins, name in zip(axs, jet_features_target, jet_features_gen, ranges, names):
        if not fill:
            ax.hist(p_gen.flatten(), bins=bins, histtype='step', stacked=True,
                    fill=False, label='reconstructed', density=density)
            ax.hist(p_target.flatten(), bins=bins, histtype='step',
                    stacked=True, fill=False, label='target', density=density)
        else:
            ax.hist(p_gen.flatten(), bins=bins, alpha=0.6, label='reconstructed', density=density)
            ax.hist(p_target.flatten(), bins=bins, alpha=0.6, label='target', density=density)

        ax.set_xlabel(f'Jet {name}')
        ax.set_ylabel('Number of jets')
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        ax.tick_params(bottom=True, top=True, left=True, right=True, direction='in')
        ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2)

    fig.tight_layout()

    jet_name = get_jet_name(args)
    if args.abs_coord:
        fig.suptitle(fr'Distribution of target and reconstructed jet $M$, $P_x$, $P_y$, and $P_z$ of {jet_name} jet', y=1.03)
    else:
        fig.suptitle('Distribution of target and reconstructed jet ' +
                     r'$M^\mathrm{rel}$, $P_x^\mathrm{rel}$, $P_y^\mathrm{rel}$, and $P_z^\mathrm{rel}$ of ' +
                     f'{jet_name} jet', y=1.03)

    save_dir = make_dir(osp.join(save_dir, 'jet_cartesian'))
    if fill:
        save_dir = osp.join(save_dir, 'filled')

    filename = f'jet_features_cartesian_{args.jet_type}_jet'
    if epoch is not None:
        filename = f'{filename}_epoch_{epoch+1}'
    if density:
        filename = f'{filename}_density'
    plt.savefig(osp.join(save_dir, f'{filename}.pdf'), bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_jet_p_polar(args, jet_features_target, jet_features_gen, save_dir,
                     epoch=None, density=False, fill=True, show=False):
    """Plot jet features (m, pt, eta, phi) distribution.

    Parameters
    ----------
    jet_features_target : `numpy.ndarray`
        The target jet data, with shape (num_particles, 4).
    jet_features_gen : `numpy.ndarray`
        The reconstructed jet data, with shape (num_particles, 4).
    save_dir : str
        The directory to save the figure.
    epoch : int
        The epoch number of the evaluated model.
        Optional, default: `None`
    density : bool
        Whether to plot distribution density instead of absolute number.
        Optional, default: `False`
    fill : bool
        Whether bins are filled.
        Optional, default: `False`
    show : bool
        Whether to show plot.
        Optional, default: `False`
    """
    if args.abs_coord:
        ranges = RANGES_CARTESIAN_ABS_COORD
        names = LABELS_CARTESIAN_ABS_COORD
    else:
        ranges = RANGES_CARTESIAN_REL_COORD
        names = LABELS_CARTESIAN_REL_COORD

    fig, axs = plt.subplots(1, 4, figsize=FIGSIZE, sharey=False)
    for ax, p_target, p_gen, bins, name in zip(axs, jet_features_target, jet_features_gen, ranges, names):
        if not fill:
            ax.hist(p_gen.flatten(), bins=bins, histtype='step', stacked=True,
                    fill=False, label='reconstructed', density=density)
            ax.hist(p_target.flatten(), bins=bins, histtype='step',
                    stacked=True, fill=False, label='target', density=density)
        else:
            ax.hist(p_gen.flatten(), bins=range, alpha=0.6, label='reconstructed', density=density)
            ax.hist(p_target.flatten(), bins=range, alpha=0.6, label='target', density=density)

        ax.set_xlabel(f'Jet {name}')
        ax.set_ylabel('Number of jets')
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        ax.tick_params(bottom=True, top=True, left=True, right=True, direction='in')
        ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2)

    fig.tight_layout()

    jet_name = get_jet_name(args)
    if args.abs_coord:
        fig.suptitle(r'Distribution of target and reconstructed jet $m$, $p_\mathrm{T}$, $\eta$, and $\phi$ ' +
                     f'of {jet_name} jets', y=1.03)
    else:
        fig.suptitle(r'Distribution of target and reconstructed jet $m^\mathrm{rel}$, $p_\mathrm{T}^\mathrm{rel}$, $\eta^\mathrm{rel}$, and $\phi^\mathrm{rel}$ ' +
                     f'of {jet_name} jets', y=1.03)

    if fill:
        save_dir = osp.join(save_dir, 'filled')
    save_dir = make_dir(osp.join(save_dir, 'jet_polar'))

    filename = f'jet_features_polar_{args.jet_type}_jet'
    if epoch is not None:
        filename = f'{filename}_epoch_{epoch+1}'
    if density:
        filename = f'{filename}_density'
    plt.savefig(osp.join(save_dir, f'{filename}.pdf'), bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
