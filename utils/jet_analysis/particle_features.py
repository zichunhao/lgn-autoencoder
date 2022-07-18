from argparse import Namespace
from typing import Iterable, Optional, Tuple, Union
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
    p3_targets: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]], 
    p3_recons: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]], 
    abs_coord: bool,
    save_dir: str,
    jet_type : str = "",
    epoch: Optional[int] = None, 
    density: bool = False, 
    fill: bool = False, 
    show: bool = False
) -> None:
    """Plot p distribution in Cartesian coordinates.

    :param p3_targets: target jet momentum components, each with shape (num_particles, 3).
    :type p3_targets: np.ndarray or a tuple of 3 np.ndarray
    :param p3_recons: _description_
    :type p3_recons: np.ndarray or a tuple of 3 np.ndarray
    :param abs_coord: whether to use absolute or relative coordinates.
        If False, use relative coordinates.
    :type abs_coord: bool
    :param save_dir: directory to save the plots.
    :type save_dir: str
    :param jet_type: string description of the jet, defaults to ""
        Example: 'g' for gluon jets and 't' for top quark jets.
    :type jet_type: str, optional
    :param epoch: current epoch, defaults to None
    :type epoch: Optional[int], optional
    :param density: whether to plot distribution density, defaults to False
    :type density: bool, optional
    :param fill: whether to fill the bins, defaults to True
    :type fill: bool, optional
    :param show: whether to show the plot, defaults to False
    :type show: bool, optional
    :raises ValueError: if p3_targets and p3_recons do not consist of 3-vectors.
    :raises TypeError: if p3_targets or p3_recons is not an np.ndarray or a tuple of 3 np.ndarray.
    """    

    if abs_coord:
        ranges = RANGES_CARTESIAN_ABS_COORD
        labels = LABELS_CARTESIAN_ABS_COORD
    else:
        ranges = RANGES_CARTESIAN_REL_COORD
        labels = LABELS_CARTESIAN_REL_COORD
    
    if isinstance(p3_targets, np.ndarray):
        if p3_targets.shape[-1] != 3:
            raise ValueError(
                f'p3_targets must be a 3-vector (px, py, pz). Found: {p3_targets.shape[-1]=}'
            )
        p3_targets = tuple([p3_targets[..., i] for i in range(3)])
    elif isinstance(p3_targets, Iterable):
        if len(p3_targets) != 3:
            raise ValueError(
                f'p3_targets must be a 3-vector (px, py, pz). Found: {len(p3_targets)=}'
            )
    else:
        raise TypeError(
            f'p3_targets must be a tuple of 3 numpy.ndarray or numpy.ndarray. Found: {type(p3_targets)}'
        )
        
    if isinstance(p3_recons, np.ndarray):
        if p3_recons.shape[-1] != 3:
            raise ValueError(
                f'p3_recons must be a 3-vector (px, py, pz). Found: {p3_recons.shape[-1]=}'
            )
        p3_recons = tuple([p3_recons[..., i] for i in range(3)])
    elif isinstance(p3_recons, Iterable):
        if len(p3_recons) != 3:
            raise ValueError(
                f'p3_recons must be a 3-vector (px, py, pz). Found: {len(p3_recons)=}'
            )
    else:
        raise TypeError(
            f'p3_recons must be a tuple of 3 numpy.ndarray or numpy.ndarray. Found: {type(p3_recons)}'
        )

    fig, axs = plt.subplots(1, 3, figsize=FIGSIZE, sharey=False)
    for ax, p_target, p3_recons, bins, name in zip(axs, p3_targets, p3_recons, ranges, labels):
        if not fill:
            ax.hist(p3_recons.flatten(), bins=bins, histtype='step', stacked=True,
                    fill=False, label='reconstructed', density=density)
            ax.hist(p_target.flatten(), bins=bins, histtype='step',
                    stacked=True, fill=False, label='target', density=density)
        else:
            ax.hist(p3_recons.flatten(), bins=bins, alpha=0.6, label='reconstructed', density=density)
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
    p3_polar_target: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]], 
    p3_polar_recons: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]], 
    abs_coord: bool,
    save_dir: str,
    jet_type: str = "",
    epoch: Optional[int] = None, 
    density: bool = False, 
    fill: bool = False, 
    show: bool = False
) -> None:
    """Plot p distribution in polar coordinates (pt, eta, phi)

    :param p3_polar_target: target jet data in polar coordinates (pt, eta, phi).
    :type p3_polar_target: np.ndarray or a tuple of 3 np.ndarray
    :param p3_polar_recons: reconstructed jet data in polar coordinates (pt, eta, phi).
    :type p3_polar_recons: np.ndarray or a tuple of 3 np.ndarray
    :param abs_coord: whether to use absolute or relative coordinates.
        If False, use relative coordinates.
    :type abs_coord: bool
    :param save_dir: directory to save the plots.
    :type save_dir: str
    :param jet_type: string description of the jet, defaults to ""
        Example: 'g' for gluon jets and 't' for top quark jets.
    :type jet_type: str, optional
    :param epoch: current epoch, defaults to None
    :type epoch: Optional[int], optional
    :param density: whether to plot distribution density, defaults to False
    :type density: bool, optional
    :param fill: whether to fill the bins, defaults to True
    :type fill: bool, optional
    :param show: whether to show the plot, defaults to False
    :type show: bool, optional
    :raises ValueError: if p3_targets and p3_recons do not consist of 3-vectors.
    :raises TypeError: if p3_targets or p3_recons is not an np.ndarray or a tuple of 3 np.ndarray.
    """    
    
    if isinstance(p3_polar_target, np.ndarray):
        if p3_polar_target.shape[-1] != 3:
            raise ValueError(
                f'p3_polar_target must be a 3-vector (pt, eta, phi). Found: {p3_polar_target.shape[-1]=}'
            )
        p3_polar_target = tuple([p3_polar_target[..., i] for i in range(3)])
    elif isinstance(p3_polar_target, Iterable):
        if len(p3_polar_target) != 3:
            raise ValueError(
                f'p3_polar_target must be a 3-vector (pt, eta, phi). Found: {len(p3_polar_target)=}'
            )
    else:
        raise TypeError(
            f'p3_polar_target must be a tuple of 3 numpy.ndarray or numpy.ndarray. Found: {type(p3_polar_target)}'
        )
    
    if isinstance(p3_polar_recons, np.ndarray):
        if p3_polar_recons.shape[-1] != 3:
            raise ValueError(
                f'p3_polar_recons must be a 3-vector (pt, eta, phi). Found: {p3_polar_recons.shape[-1]=}'
            )
        p3_polar_recons = tuple([p3_polar_recons[..., i] for i in range(3)])
    elif isinstance(p3_polar_recons, Iterable):
        if len(p3_polar_recons) != 3:
            raise ValueError(
                f'p3_polar_recons must be a 3-vector (pt, eta, phi). Found: {len(p3_polar_recons)=}'
            )
    else:
        raise TypeError(
            f'p3_polar_recons must be a tuple of 3 numpy.ndarray or numpy.ndarray. Found: {type(p3_polar_recons)}'
        )

    # pt_target, eta_target, phi_target = p3_polar_target
    # pt_recons, eta_recons, phi_recons = p3_polar_recons

    fig, axs = plt.subplots(1, 3, figsize=FIGSIZE, sharey=False)

    if abs_coord:
        ranges = RANGES_POLAR_ABS_COORD
        labels = LABELS_POLAR_ABS_COORD
    else:
        ranges = RANGES_POLAR_REL_COORD
        labels = LABELS_POLAR_REL_COORD

    for ax, p_target, p3_recons, bins, name in zip(axs, p3_polar_target, p3_polar_recons, ranges, labels):
        if not fill:
            ax.hist(p3_recons.flatten(), bins=bins, histtype='step', stacked=True,
                    fill=False, label='reconstructed', density=density)
            ax.hist(p_target.flatten(), bins=bins, histtype='step',
                    stacked=True, fill=False, label='target', density=density)
        else:
            ax.hist(p3_recons.flatten(), bins=bins, alpha=0.6, label='reconstructed', density=density)
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
