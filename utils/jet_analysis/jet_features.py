from utils.jet_analysis.utils import get_jet_name
from utils.utils import make_dir
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

JET_FEATURE_FIGSIZE = (16, 4)


def plot_jet_p_cartesian(args, jet_features_target, jet_features_gen, save_dir, max_val=[200, 2000, 2000, 4000],
                         num_bins=81, epoch=None, density=False, fill=True, show=False):
    """Plot jet features (m, px, py, pz) distribution.

    Args
    ----
    jet_features_target : `numpy.ndarray`
        The target jet momenta, with shape (num_jets, 4).
    jet_features_gen : `numpy.ndarray`
        The generated/reconstructed jet momenta, with shape (num_jets, 4).
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

    fig, axs = plt.subplots(1, 4, figsize=JET_FEATURE_FIGSIZE, sharey=False)
    if type(max_val) in [tuple, list]:
        if len(max_val) == 4:
            m_max, px_max, py_max, pz_max = max_val
        else:
            m_max, px_max, py_max, pz_max = max_val[0]
    elif type(max_val) in [float, int]:
        m_max, px_max = py_max = pz_max = float(max_val)
    else:
        m_max, px_max, py_max, pz_max = [30000, 2000, 2000, 4000]

    ranges = [np.linspace(0, m_max, num_bins),
              np.linspace(-px_max, px_max, num_bins),
              np.linspace(-py_max, py_max, num_bins),
              np.linspace(-pz_max, pz_max, num_bins)]
    names = ['mass', r'$p_x$', r'$p_y$', r'$p_z$']
    for ax, p_target, p_gen, bins, name in zip(axs, jet_features_target, jet_features_gen, ranges, names):
        if not fill:
            ax.hist(p_gen.flatten(), bins=bins, histtype='step', stacked=True,
                    fill=False, label='reconstructed', density=density)
            ax.hist(p_target.flatten(), bins=bins, histtype='step',
                    stacked=True, fill=False, label='target', density=density)
        else:
            ax.hist(p_gen.flatten(), bins=bins, alpha=0.6, label='reconstructed', density=density)
            ax.hist(p_target.flatten(), bins=bins, alpha=0.6, label='target', density=density)
        ax.set_xlabel(f'Jet {name} (GeV)')
        ax.set_ylabel('Number of jets')
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        ax.tick_params(bottom=True, top=True, left=True, right=True, direction='in')
        ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2)

    fig.tight_layout()

    jet_name = get_jet_name(args)
    fig.suptitle(fr'Distribution of target and reconstructed jet $m$, $p_x$, $p_y$, and $p_z$ of {jet_name} jet', y=1.03)

    save_dir = make_dir(osp.join(save_dir, 'jet_cartesian'))
    if fill:
        save_dir = osp.join(save_dir, 'filled')

    filename = f'jet_features_cartesian_{args.jet_type}_jet'
    if epoch is not None:
        filename = f'{filename}_epoch_{epoch+1}'
    if density:
        filename = f'{filename}_density'
    plt.savefig(osp.join(save_dir, f'{filename}.pdf'), bbox_inches="tight", transparent=True)
    if show:
        plt.show()
    plt.close()


def plot_jet_p_polar(args, jet_features_target, jet_features_gen, save_dir, max_val=(30000, 2000, 2, np.pi),
                     num_bins=201, epoch=None, density=False, fill=True, show=False):
    """Plot jet features (m, pt, eta, phi) distribution.

    Args
    ----
    jet_features_target : `numpy.ndarray`
        The target jet data, with shape (num_particles, 4).
    jet_features_gen : `numpy.ndarray`
        The reconstructed jet data, with shape (num_particles, 4).
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
    if fill:
        save_dir = osp.join(save_dir, 'filled')
    save_dir = make_dir(osp.join(save_dir, 'jet_polar'))

    fig, axs = plt.subplots(1, 4, figsize=JET_FEATURE_FIGSIZE, sharey=False)

    if type(max_val) in [tuple, list]:
        if len(max_val) == 4:
            m_max, pt_max, eta_max, phi_max = max_val
        elif len(max_val) == 3:
            m_max = max_val[0]
            pt_max = max_val[0]
            eta_max = phi_max = max_val[2]
    elif type(max_val) in [float, int]:
        m_max = pt_max = eta_max = phi_max = float(max_val)
    else:
        m_max = 30000
        pt_max = 0.15
        eta_max = phi_max = np.pi

    ranges = [np.linspace(0, m_max, num_bins),
              np.linspace(0, pt_max, num_bins),
              np.linspace(-eta_max, eta_max, num_bins),
              np.linspace(-phi_max, phi_max, num_bins)]
    names = [r'$m$', r'$p_\mathrm{T}$', r'$\eta$', r'$\phi$']

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
        if name in [r'$m$', r'$p_\mathrm{T}$']:
            ax.set_xlabel(f'Jet {name} (GeV)')
        ax.set_ylabel('Number of jets')
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        ax.tick_params(bottom=True, top=True, left=True, right=True, direction='in')
        ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2)

    fig.tight_layout()

    jet_name = get_jet_name(args)
    fig.suptitle(r'Distribution of target and reconstructed jet $m$, $p_\mathrm{T}$, $\eta$, and $\phi$ ' +
                 f'of {jet_name} jets', y=1.03)

    filename = f'jet_features_polar_{args.jet_type}_jet'
    if epoch is not None:
        filename = f'{filename}_epoch_{epoch+1}'
    if density:
        filename = f'{filename}_density'
    plt.savefig(osp.join(save_dir, f'{filename}.pdf'), bbox_inches="tight", transparent=True)
    if show:
        plt.show()
    plt.close()
