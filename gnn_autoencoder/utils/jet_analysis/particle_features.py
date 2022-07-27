from typing import Optional, Tuple, Union
from .utils import get_jet_name
from utils.utils import make_dir
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from .utils import NUM_BINS, PLOT_FONT_SIZE

FIGSIZE = (12, 4)
LABELS_CARTESIAN_ABS_COORD = (r'$p_x$  (GeV)', r'$p_y$  (GeV)', r'$p_z$  (GeV)')
LABELS_CARTESIAN_REL_COORD = (r'$p_x^\mathrm{rel}$', r'$p_y^\mathrm{rel}$', r'$p_z^\mathrm{rel}$')
LABELS_POLAR_ABS_COORD = (r'$p_\mathrm{T}$ (GeV)', r'$\eta$', r'$\phi$')
LABELS_POLAR_REL_COORD = (r'$p_\mathrm{T}^\mathrm{rel}$', r'$\eta^\mathrm{rel}$', r'$\phi^\mathrm{rel}$')

RANGES_CARTESIAN_ABS_COORD = (
    np.linspace(-100, 100, NUM_BINS),  # px
    np.linspace(-100, 100, NUM_BINS),  # py
    np.linspace(-100, 100, NUM_BINS)   # pz
)
RANGES_CARTESIAN_REL_COORD = (
    np.linspace(0, 0.3, NUM_BINS),       # px_rel
    np.linspace(-0.01, 0.01, NUM_BINS),  # py_rel
    np.linspace(-0.01, 0.01, NUM_BINS)   # pz_rel
)
RANGES_POLAR_ABS_COORD = (
    np.linspace(0, 200, NUM_BINS),        # pt
    np.linspace(-2, 2, NUM_BINS),         # eta
    np.linspace(-np.pi, np.pi, NUM_BINS)  # phi
)
RANGES_POLAR_REL_COORD = (
    np.linspace(0, 0.3, NUM_BINS),     # pt_rel
    np.linspace(-0.5, 0.5, NUM_BINS),  # eta_rel
    np.linspace(-0.5, 0.5, NUM_BINS)   # phi_rel
)


def plot_p_cartesian(
    p_targets: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    p_gens: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    save_dir: str,
    abs_coord: bool,
    jet_type: str = "",
    epoch: Optional[int] = None,
    density: bool = False, 
    fill: bool = False, 
    show: bool = False
) -> None:
    """Plot p distribution in Cartesian coordinates.

    Parameters
    ----------
    p_targets : iterbale of `numpy.ndarray`
        The target jet momentum compoenents, each with shape (num_particles, 4).
    p_gens : iterbale of `numpy.ndarray`
        The reconstructed jet momentum compoenents, each with shape (num_particles, 4).
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

    if abs_coord:
        ranges = RANGES_CARTESIAN_ABS_COORD
        labels = LABELS_CARTESIAN_ABS_COORD
    else:
        ranges = RANGES_CARTESIAN_REL_COORD
        labels = LABELS_CARTESIAN_REL_COORD

    fig, axs = plt.subplots(1, 3, figsize=FIGSIZE, sharey=False)
    for ax, p_target, p_gen, bins, name in zip(axs, p_targets, p_gens, ranges, labels):
        if not fill:
            ax.hist(p_gen.flatten(), bins=bins, histtype='step', stacked=True,
                    fill=False, label='reconstructed', density=density)
            ax.hist(p_target.flatten(), bins=bins, histtype='step',
                    stacked=True, fill=False, label='target', density=density)
        else:
            ax.hist(p_gen.flatten(), bins=bins, alpha=0.6, label='reconstructed', density=density)
            ax.hist(p_target.flatten(), bins=bins, alpha=0.6, label='target', density=density)

        ax.set_xlabel(f'Particle {name}')
        ax.set_ylabel('Number of particles')
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        for axis in ('x', 'y'):
            ax.tick_params(axis=axis, labelsize=PLOT_FONT_SIZE)
        ax.tick_params(bottom=True, top=True, left=True, right=True, direction='in')
        ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    plt.rcParams.update({'font.size': PLOT_FONT_SIZE})

    fig.tight_layout()

    jet_name = get_jet_name(jet_type)
    fig.suptitle(fr'Distribution of target and reconstructed particle $p_x$, $p_y$, and $p_z$ of {jet_name} jets', y=1.03)

    if epoch is not None:
        if fill:
            save_dir = osp.join(save_dir, 'filled')
        save_dir = make_dir(osp.join(save_dir, 'particle_features/cartesian'))
    else:
        pass  # Save without creating a subdirectory

    filename = f'p_cartesian_{jet_type}_jet'
    if epoch is not None:
        filename = f'{filename}_epoch_{epoch+1}'
    if density:
        filename = f'{filename}_density'
    if fill:
        filename = f'{filename}_filled'
    plt.savefig(osp.join(save_dir, f'{filename}.pdf'), bbox_inches="tight")

    if show:
        plt.show()
    plt.close()


def plot_p_polar(
    p_polar_target: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    p_polar_gen: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    save_dir: str,
    abs_coord: bool,
    jet_type: str = "",
    epoch: Optional[int] = None, 
    density: bool = False, 
    fill: bool = True, 
    show: bool = False
) -> None:
    """Plot p distribution in polar coordinates (pt, eta, phi)

    Parameters
    ----------
    p_polar_target : tuple of 3 `numpy.ndarray`
        The target jet data in polar coordinates (pt, eta, phi).
    p_polar_gen : tuple of 3 `numpy.ndarray`
        The reconstructed jet data in polar coordinates (pt, eta, phi).
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

    # pt_target, eta_target, phi_target = p_polar_target
    # pt_gen, eta_gen, phi_gen = p_polar_gen

    fig, axs = plt.subplots(1, 3, figsize=FIGSIZE, sharey=False)

    if abs_coord:
        ranges = RANGES_POLAR_ABS_COORD
        labels = LABELS_POLAR_ABS_COORD
    else:
        ranges = RANGES_POLAR_REL_COORD
        labels = LABELS_POLAR_REL_COORD

    for ax, p_target, p_gen, bins, name in zip(axs, p_polar_target, p_polar_gen, ranges, labels):
        if not fill:
            ax.hist(p_gen.flatten(), bins=bins, histtype='step', stacked=True,
                    fill=False, label='reconstructed', density=density)
            ax.hist(p_target.flatten(), bins=bins, histtype='step',
                    stacked=True, fill=False, label='target', density=density)
        else:
            ax.hist(p_gen.flatten(), bins=bins, alpha=0.6, label='reconstructed', density=density)
            ax.hist(p_target.flatten(), bins=bins, alpha=0.6, label='target', density=density)
        ax.set_xlabel(f'Particle {name}')
        ax.set_ylabel('Number of particles')

        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        for axis in ('x', 'y'):
            ax.tick_params(axis=axis, labelsize=PLOT_FONT_SIZE)
        ax.tick_params(bottom=True, top=True, left=True, right=True, direction='in')
        ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4)
    plt.rcParams.update({'font.size': PLOT_FONT_SIZE})

    fig.tight_layout()

    jet_name = get_jet_name(jet_type)
    if abs_coord:
        fig.suptitle(r'Distribution of target and reconstructed particle $p_\mathrm{T}$, $\eta$, and $\phi$ ' +
                     f'of {jet_name} jets', y=1.03)
    else:
        fig.suptitle(r'Distribution of target and reconstructed particle $p_\mathrm{T}^\mathrm{rel}$, $\eta^\mathrm{rel}$, and $\phi^\mathrm{rel}$ ' +
                     f'of {jet_name} jets', y=1.03)

    if epoch is not None:
        if fill:
            save_dir = osp.join(save_dir, 'filled')
        save_dir = make_dir(osp.join(save_dir, 'particle_features/polar'))
    else:
        pass  # Save without creating a subdirectory

    filename = f'p_polar_{jet_type}_jet'
    if epoch is not None:
        filename = f'{filename}_epoch_{epoch+1}'
    if density:
        filename = f'{filename}_density'
    if fill:
        filename = f'{filename}_filled'
    plt.savefig(osp.join(save_dir, f'{filename}.pdf'), bbox_inches="tight")

    if show:
        plt.show()
    plt.close()
