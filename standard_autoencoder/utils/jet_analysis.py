import numpy as np
import matplotlib.pyplot as plt
import os.path as osp

from utils.utils import make_dir


def plot_p(args, target_data, gen_data, save_dir, polar_max=(200, 2, np.pi), cartesian_max=(100, 100, 100),
           jet_cartesian_max=(200, 2000, 2000, 4000), jet_polar_max=(200, 4000, 2, np.pi),
           num_bins=201, cutoff=1e-6, epoch=None, show=False):

    plot_p_polar(args, target_data, gen_data, save_dir, max_val=polar_max, num_bins=num_bins,
                 cutoff=cutoff, epoch=epoch, density=False, fill=False, show=show)
    plot_p_cartesian(args, target_data, gen_data, save_dir, max_val=cartesian_max,
                     num_bins=num_bins, cutoff=cutoff, epoch=epoch, density=False, fill=False, show=show)
    plot_jet_p_polar(args, target_data, gen_data, save_dir, max_val=jet_polar_max,
                     num_bins=81, epoch=epoch, density=False, fill=False, show=False)
    plot_jet_p_cartesian(args, target_data, gen_data, save_dir, max_val=jet_cartesian_max,
                         num_bins=81, epoch=epoch, density=False, fill=False, show=False)

    if args.fill:  # Plot filled histograms in addition to unfilled histograms
        plot_p_polar(args, target_data, gen_data, save_dir, max_val=polar_max, num_bins=num_bins,
                     cutoff=cutoff, epoch=epoch, density=False, fill=True, show=show)
        plot_p_cartesian(args, target_data, gen_data, save_dir, max_val=cartesian_max,
                         num_bins=num_bins, cutoff=cutoff, epoch=epoch, density=False, fill=True, show=show)
        plot_jet_p_polar(args, target_data, gen_data, save_dir, max_val=jet_polar_max,
                         num_bins=81, epoch=epoch, density=False, fill=True, show=False)
        plot_jet_p_cartesian(args, target_data, gen_data, save_dir, max_val=jet_cartesian_max,
                             num_bins=81, epoch=epoch, density=False, fill=True, show=False)


def plot_p_cartesian(args, target_data, gen_data, save_dir, max_val=[100, 100, 100],
                     num_bins=201, cutoff=1e-6, epoch=None, density=False, fill=False, show=False):
    """
    Plot p distribution in Cartesian coordinates.

    Parameters
    ----------
    target_data : `numpy.Array`
        The target jet data, with shape (num_particles, 4).
    gen_data : `numpy.Array`
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
    cutoff : `float`
        The cutoff value of 3-momenta.
        Optional, default: `1e-6`.
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
    save_dir = make_dir(osp.join(save_dir, 'cartesian'))
    px_target, py_target, pz_target = get_p_cartesian(target_data, cutoff=cutoff)
    px_gen, py_gen, pz_gen = get_p_cartesian(gen_data, cutoff=cutoff)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
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

    filename = f'p_cartesian_{args.jet_type}_jet'
    if epoch is not None:
        filename = f'{filename}_epoch_{epoch+1}'
    if density:
        filename = f'{filename}_density'
    plt.savefig(osp.join(save_dir, f'{filename}.pdf'), bbox_inches="tight", transparent=True)
    if show:
        plt.show()
    plt.close()


def plot_jet_p_cartesian(args, target_data, gen_data, save_dir, max_val=[200, 2000, 2000, 4000],
                         num_bins=81, epoch=None, density=False, fill=True, show=False):
    """
    Plot jet features (m, px, py, pz) distribution.

    Parameters
    ----------
    target_data : `numpy.Array`
        The target jet data, with shape (num_particles, 4).
    gen_data : `numpy.Array`
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
    save_dir = make_dir(osp.join(save_dir, 'jet_cartesian'))
    if fill:
        save_dir = osp.join(save_dir, 'filled')

    jet_features_target = get_jet_feature_cartesian(target_data)
    jet_features_gen = get_jet_feature_cartesian(gen_data)

    fig, axs = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
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

    filename = f'jet_features_cartesian_{args.jet_type}_jet'
    if epoch is not None:
        filename = f'{filename}_epoch_{epoch+1}'
    if density:
        filename = f'{filename}_density'
    plt.savefig(osp.join(save_dir, f'{filename}.pdf'), bbox_inches="tight", transparent=True)
    if show:
        plt.show()
    plt.close()


def plot_p_polar(args, target_data, gen_data, save_dir, max_val=(200, 2, np.pi),
                 num_bins=201, cutoff=1e-6, epoch=None, density=False, fill=True, show=False):
    """
    Plot p distribution in polar coordinates (pt, eta, phi)

    Parameters
    ----------
    target_data : `numpy.Array`
        The target jet data, with shape (num_particles, 4).
    gen_data : `numpy.Array`
        The reconstructed jet data, with shape (num_particles, 4).
    save_dir : `str`
        The directory to save the figure.
    max_val : `list`, `tuple`, or `float`
        The maximum values of (pt, eta, phi) in the plot.
        Optional, default: `(0.15, np.pi, np.pi)`
    num_bins : `int`
        The number of bins in the histogram.
        Optional, default: `201`
    cutoff : `float`
        The cutoff value of 3-momenta.
        Optional, default: `1e-6`.
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
    save_dir = make_dir(osp.join(save_dir, 'polar'))
    pt_target, eta_target, phi_target = get_p_polar(target_data, cutoff=cutoff)
    pt_gen, eta_gen, phi_gen = get_p_polar(gen_data, cutoff=cutoff)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=False)

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

    p_targets = [pt_target, eta_target, phi_target]
    p_gens = [pt_gen, eta_gen, phi_gen]
    ranges = [np.linspace(0, pt_max, num_bins),
              np.linspace(-eta_max, eta_max, num_bins),
              np.linspace(-phi_max, phi_max, num_bins)]
    names = [r'$p_\mathrm{T}$', r'$\eta$', r'$\phi$']
    for ax, p_target, p_gen, bins, name in zip(axs, p_targets, p_gens, ranges, names):
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

    filename = f'p_polar_{args.jet_type}_jet'
    if epoch is not None:
        filename = f'{filename}_epoch_{epoch+1}'
    if density:
        filename = f'{filename}_density'
    plt.savefig(osp.join(save_dir, f'{filename}.pdf'), bbox_inches="tight", transparent=True)
    if show:
        plt.show()
    plt.close()


def plot_jet_p_polar(args, target_data, gen_data, save_dir, max_val=(30000, 2000, 2, np.pi),
                     num_bins=201, epoch=None, density=False, fill=True, show=False):
    """
    Plot jet features (m, pt, eta, phi) distribution.

    Parameters
    ----------
    target_data : `numpy.Array`
        The target jet data, with shape (num_particles, 4).
    gen_data : `numpy.Array`
        The reconstructed jet data, with shape (num_particles, 4).
    save_dir : `str`
        The directory to save the figure.
    max_val : `list`, `tuple`, or `float`
        The maximum values of (pt, eta, phi) in the plot.
        Optional, default: `(0.15, np.pi, np.pi)`
    num_bins : `int`
        The number of bins in the histogram.
        Optional, default: `201`
    cutoff : `float`
        The cutoff value of 3-momenta.
        Optional, default: `1e-6`.
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

    jet_features_target = get_jet_feature_polar(target_data)
    jet_features_gen = get_jet_feature_polar(gen_data)

    fig, axs = plt.subplots(1, 4, figsize=(16, 4), sharey=False)

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


def get_p_cartesian(jet_data, cutoff=1e-6):
    """
    Get (px, py, pz) from the jet data and filter out values that are too small.

    Input
    -----
    jet_data : `numpy.Array`
        The jet data, with shape (num_particles, 4), which means all jets are merged together.
    cutoff : `float`
        The cutoff value of 3-momenta.
    """
    if jet_data.shape[-1] == 4:
        jet_data = np.copy(jet_data).reshape(-1, 4)
        px = jet_data[:, 1].copy()
        py = jet_data[:, 2].copy()
        pz = jet_data[:, 3].copy()
    elif jet_data.shape[-1] == 3:  # 3 vectors
        jet_data = np.copy(jet_data).reshape(-1, 3)
        px = jet_data[:, 0].copy()
        py = jet_data[:, 1].copy()
        pz = jet_data[:, 2].copy()
    else:
        raise ValueError(f"Particle momenta must be 3- or 4-vectors. Found: {jet_data.shape[-1]=}.")

    p = get_magnitude(jet_data)  # |p| of 3-momenta
    px[p < cutoff] = np.nan
    py[p < cutoff] = np.nan
    pz[p < cutoff] = np.nan

    return px, py, pz


def get_magnitude(p):
    if p.shape[-1] == 4:
        return np.sqrt(np.sum(np.power(p, 2)[..., 1:], axis=-1))
    elif p.shape[-1] == 3:
        return np.sqrt(np.sum(np.power(p, 2), axis=-1))  # E^2 = p^2 for each particle
    else:
        raise ValueError(f"Particle momenta must be 3- or 4-vectors. Found: {p.shape[-1]=}.")

def get_p_polar(jet_data, cutoff=1e-6):
    """
    Get (pt, eta, phi) from the jet data.

    Input
    -----
    jet_data : `numpy.Array`
        The jet data, with shape (num_particles, 4), which means all jets are merged together.
    """
    px, py, pz = get_p_cartesian(jet_data, cutoff=cutoff)

    pt = np.sqrt(px ** 2 + py ** 2)
    eta = np.arcsinh(pz / (pt + 1e-16))
    phi = np.arctan2(py, px)

    return pt, eta, phi


def get_jet_feature_cartesian(p):
    """
    Get jet (m, pt, eta, phi) from the jet data.

    Input
    -----
    jet_data : `numpy.Array`
        The jet data, with shape (num_particles, 4), which means all jets are merged together.
    """
    if p.shape[-1] == 3:
        p0 = np.expand_dims(get_magnitude(p), axis=-1)
        p4 = np.concatenate((p0, p), axis=-1)
    elif p.shape[-1] == 4:
        p4 = p
    else:
        raise ValueError(f"Particle momenta must be 3- or 4-vectors. Found: {p.shape[-1]=}.")

    jet_p4 = np.sum(p4, axis=-2)
    msq = jet_p4[:, 0] ** 2 - np.sum(np.power(jet_p4, 2)[:, 1:], axis=-1)
    jet_mass = np.sqrt(np.abs(msq)) * np.sign(msq)
    jet_px = jet_p4[:, 1]
    jet_py = jet_p4[:, 2]
    jet_pz = jet_p4[:, 3]

    return jet_mass, jet_px, jet_py, jet_pz


def get_jet_feature_polar(p4):
    """
    Get jet (m, pt, eta, phi) from the jet data.

    Input
    -----
    jet_data : `numpy.Array`
        The jet data, with shape (num_particles, 4), which means all jets are merged together.
    """
    m, px, py, pz = get_jet_feature_cartesian(p4)

    pt = np.sqrt(px ** 2 + py ** 2)
    eta = np.arcsinh(pz / (pt + 1e-16))
    phi = np.arctan2(py, px)

    return m, pt, eta, phi


def get_jet_name(args):
    if args.jet_type == 'g':
        jet_name = 'gluon'
    elif args.jet_type == 'q':
        jet_name = 'light quark'
    elif args.jet_type == 't':
        jet_name = 'top quark'
    elif args.jet_type == 'w':
        jet_name = 'W boson'
    elif args.jet_type == 'z':
        jet_name = 'Z boson'
    else:
        jet_name = args.jet_type
    return jet_name
