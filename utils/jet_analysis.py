import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os.path as osp

from utils.utils import make_dir


def plot_p(args, real_data, gen_data, save_dir, polar_max=[0.15, np.pi/4, np.pi/4], cartesian_max=(0.02, 0.02, 0.02),
           num_bins=201, cutoff=1e-6, epoch=None, fill=False, show=False):

    plot_p_polar(args, real_data, gen_data, save_dir, max_val=polar_max, num_bins=num_bins, cutoff=cutoff, epoch=epoch, density=False, fill=fill, show=show)
    plot_p_cartesian(args, real_data, gen_data, save_dir, max_val=cartesian_max, num_bins=num_bins, cutoff=cutoff, epoch=epoch, density=False, fill=fill, show=show)


def plot_p_cartesian(args, real_data, gen_data, save_dir, max_val=[0.02, 0.02, 0.02], num_bins=201, cutoff=1e-6, epoch=None, density=False, fill=False, show=False):
    """
    Plot p distribution in Cartesian coordinates.

    Parameters
    ----------
    real_data : `numpy.Array`
        The target jet data, with shape (num_particles, 4).
    gen_data : `numpy.Array`
        The generated jet data, with shape (num_particles, 4).
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

    save_dir = make_dir(osp.join(save_dir, 'cartesian'))
    px_real, py_real, pz_real = get_p_cartesian(real_data, cutoff=cutoff)
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

    p_reals = [px_real, py_real, pz_real]
    p_gens = [px_gen, py_gen, pz_gen]
    ranges = [np.linspace(-px_max,px_max,num_bins), np.linspace(-py_max,py_max,num_bins), np.linspace(-pz_max,pz_max,num_bins)]
    names = [r'$p_x$', r'$p_y$', r'$p_z$']
    for ax, p_real, p_gen, range, name in zip(axs, p_reals, p_gens, ranges, names):
        if not fill:
            ax.hist(p_gen.flatten(), bins=range, histtype='step', stacked=True, fill=False, label='generated', density=density)
            ax.hist(p_real.flatten(), bins=range, histtype='step', stacked=True, fill=False, label='target', density=density)
        else:
            ax.hist(p_gen.flatten(), bins=range, alpha=0.6, label='generated', density=density)
            ax.hist(p_real.flatten(), bins=range, alpha=0.6, label='target', density=density)
        ax.set_xlabel(f'Particle {name}')
        ax.set_ylabel('Number of particles')
        ax.legend()
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.yaxis.major.formatter._useMathText = True
        ax.tick_params(bottom=True, top=True, left=True, right=True, direction='in')
        ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)

    fig.tight_layout()

    jet_name = get_jet_name(args)
    fig.suptitle(fr'Distribution of target and generated particle $p_x$, $p_y$, and $p_z$ of {jet_name} jets', y=1.03)

    filename = f'p_cartesian_{args.jet_type}_jet'
    if epoch is not None:
        filename = f'{filename}_epoch_{epoch+1}'
    if density:
        filename = f'{filename}_density'
    plt.savefig(osp.join(save_dir, f'{filename}.pdf'), bbox_inches = "tight")
    if show:
        plt.show()
    plt.close()

def plot_p_polar(args, real_data, gen_data, save_dir, max_val=[0.15, np.pi, np.pi], num_bins=201, cutoff=1e-6, epoch=None, density=False, fill=True, show=False):
    """
    Plot p distribution in Cartesian coordinates

    Parameters
    ----------
    real_data : `numpy.Array`
        The target jet data, with shape (num_particles, 4).
    gen_data : `numpy.Array`
        The generated jet data, with shape (num_particles, 4).
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

    save_dir = make_dir(osp.join(save_dir, 'polar'))
    pt_real, eta_real, phi_real = get_p_polar(real_data, cutoff=cutoff)
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
        pt_max = 0.15
        eta_max = phi_max = np.pi

    p_reals = [pt_real, eta_real, phi_real]
    p_gens = [pt_gen, eta_gen, phi_gen]
    ranges = [np.linspace(0,pt_max,num_bins), np.linspace(-eta_max,eta_max,num_bins), np.linspace(-phi_max,phi_max,num_bins)]
    names = [r'$p_\mathrm{T}$', r'$\eta$', r'$\phi$']
    for ax, p_real, p_gen, bins, name in zip(axs, p_reals, p_gens, ranges, names):
        if not fill:
            ax.hist(p_gen.flatten(), bins=bins, histtype='step', stacked=True, fill=False, label='generated', density=density)
            ax.hist(p_real.flatten(), bins=bins, histtype='step', stacked=True, fill=False, label='target', density=density)
        else:
            ax.hist(p_gen.flatten(), bins=bins, alpha=0.6, label='generated', density=density)
            ax.hist(p_real.flatten(), bins=bins, alpha=0.6, label='target', density=density)
        ax.set_xlabel(f'Particle {name}')
        ax.set_ylabel('Number of particles')
        ax.legend()
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.yaxis.major.formatter._useMathText = True
        ax.tick_params(bottom=True, top=True, left=True, right=True, direction='in')
        ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)

    fig.tight_layout()

    jet_name = get_jet_name(args)
    fig.suptitle(r'Distribution of target and generated particle $p_\mathrm{T}$, $\eta$, and $\phi$ ' + f'of {jet_name} jets', y=1.03)

    filename = f'p_polar_{args.jet_type}_jet'
    if epoch is not None:
        filename = f'{filename}_epoch_{epoch+1}'
    if density:
        filename = f'{filename}_density'
    plt.savefig(osp.join(save_dir, f'{filename}.pdf'), bbox_inches = "tight")
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
    px = jet_data[:, 1].copy()
    py = jet_data[:, 2].copy()
    pz= jet_data[:, 3].copy()

    p = get_magnitude(jet_data)  # |p| of 3-momenta
    px[p < cutoff] = np.nan
    py[p < cutoff] = np.nan
    pz[p < cutoff] = np.nan

    return px, py, pz

def get_magnitude(data):
    return np.sqrt(data[:, 1] ** 2 + data[:, 2] ** 2 + data[:, 3] ** 3)

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
