import numpy as np
import matplotlib.pyplot as plt
import torch
import os.path as osp

from utils.utils import make_dir, eps

# Constants
PARTICLE_FEATURE_FIGSIZE = (12, 4)
JET_FEATURE_FIGSIZE = (16, 4)
JET_IMAGE_FIGSIZE = (7.5, 3)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GPU_STR = ['cuda', 'gpu']


def plot_p(args, p4_target, p4_gen, save_dir,
           cartesian_max=(100, 100, 100), polar_max=(200, 2, np.pi),
           jet_cartesian_max=(200, 2000, 2000, 4000), jet_polar_max=(200, 4000, 2, np.pi),
           num_bins=201, cutoff=1e-6, epoch=None, show=False):
    """Plot particle features, jet features, and jet images.

    Args
    ----
    p4_target :  `torch.Tensor`
        The target jets.
        Shape: (num_jets, num_particles, 4)
    p4_gen :  `torch.Tensor`
        The generated/reconstructed jets by model.
        Shape: (num_jets, num_particles, 4)
    save_dir : `str`
        The saving directories for figures.
    cartesian_max : `tuple`
        The maximum values, (px_max, py_max, pz_max), for particle feature plots in cartesian coordinates.
            - px will range from -px_max to px_max.
            - py will range from -py_max to py_max.
            - pz will range from -pz_max to pz_max.
        Default: (100, 100, 100)
    polar_max : `tuple`
        The maximum values, (pt_max, eta_max, phi_max), for particle feature plots in polar coordinates.
            - pt will range from 0 to pt_max.
            - eta will range from -eta_max to eta_max.
            - phi will range from -phi_max to phi_max.
        Default: (200, 2, np.pi)
    jet_cartesian_max : `tuple`
        The maximum values, (m_max, pt_max, eta_max, phi_max), for jet feature plots in polar coordinates.
            - jet mass (m) will range from 0 to m_max.
            - pt will range from 0 to pt_max.
            - eta will range from -eta_max to eta_max.
            - phi will range from -phi_max to phi_max.
        Default: (200, 2000, 2000, 4000)
    jet_polar_max : `tuple`
        The maximum values, (pt_max, eta_max, phi_max), for jet feature plots in polar coordinates.
            - jet mass (m) will range from 0 to m_max.
            - pt will range from 0 to pt_max.
            - eta will range from -eta_max to eta_max.
            - phi will range from -phi_max to phi_max.
        Default: (200, 4000, 2, np.pi)
    num_bins : `int`
        Number of bins for histograms of particle and jet features.
        Default: 201
    cutoff : `float`
        The cutoff value for |p| = sqrt(px^2 + py^2 + pz^2).
        Particle momentum lower than `cutoff` will be considered padded particles and thus dropped.
        Default: 1e-6
    epoch : `int`
        The epoch number.
        Default: None
    show : `bool`
        Whether to show plots.
        Default: False
    """
    EPS = eps(args)

    # tuples
    p_target_polar = get_p_polar(p4_target, cutoff=cutoff, eps=EPS)
    jet_target_polar = get_jet_feature_polar(p4_target)
    p_gen_polar = get_p_polar(p4_gen, cutoff=cutoff, eps=EPS)
    jet_gen_polar = get_jet_feature_polar(p4_gen)

    p_target_cartesian = get_p_cartesian(p4_target.detach().cpu().numpy(), cutoff=cutoff)
    jet_target_cartesian = get_jet_feature_cartesian(p4_target.detach().cpu().numpy())
    p_gen_cartesian = get_p_cartesian(p4_gen.detach().cpu().numpy(), cutoff=cutoff)
    jet_gen_cartesian = get_jet_feature_cartesian(p4_gen.detach().cpu().numpy())

    # arrays
    jets_target = get_p_polar_tensor(p4_target, eps=EPS)
    jets_gen = get_p_polar_tensor(p4_gen, eps=EPS)

    plot_p_polar(args, p_target_polar, p_gen_polar, save_dir, max_val=polar_max, num_bins=num_bins,
                 epoch=epoch, density=False, fill=False, show=show)
    plot_jet_p_polar(args, jet_target_polar, jet_gen_polar, save_dir, max_val=jet_polar_max,
                     num_bins=81, epoch=epoch, density=False, fill=False, show=False)
    plot_p_cartesian(args, p_target_cartesian, p_gen_cartesian, save_dir, max_val=cartesian_max,
                     num_bins=num_bins, epoch=epoch, density=False, fill=False, show=show)
    plot_jet_p_cartesian(args, jet_target_cartesian, jet_gen_cartesian, save_dir, max_val=jet_cartesian_max,
                         num_bins=81, epoch=epoch, density=False, fill=False, show=show)
    plot_jet_image(args, jets_target, jets_gen, save_dir, epoch, vmin=args.jet_image_vmin, show=show)

    if args.fill:  # Plot filled histograms in addition to unfilled histograms
        plot_p_polar(args, p_target_polar, p_gen_polar, save_dir, max_val=polar_max, num_bins=num_bins,
                     epoch=epoch, density=False, fill=True, show=show)
        plot_p_cartesian(args, p_target_cartesian, p_target_polar, save_dir, max_val=cartesian_max,
                         num_bins=num_bins, epoch=epoch, density=False, fill=True, show=show)
        plot_jet_p_polar(args, jet_target_polar, jet_gen_polar, save_dir, max_val=jet_polar_max,
                         num_bins=81, epoch=epoch, density=False, fill=True, show=show)
        plot_jet_p_cartesian(args, jet_gen_cartesian, jet_gen_cartesian, save_dir, max_val=jet_cartesian_max,
                             num_bins=81, epoch=epoch, density=False, fill=True, show=show)


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
    save_dir = make_dir(osp.join(save_dir, 'cartesian'))

    filename = f'p_cartesian_{args.jet_type}_jet'
    if epoch is not None:
        filename = f'{filename}_epoch_{epoch+1}'
    if density:
        filename = f'{filename}_density'
    plt.savefig(osp.join(save_dir, f'{filename}.pdf'), bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


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
    save_dir = make_dir(osp.join(save_dir, 'polar'))

    filename = f'p_polar_{args.jet_type}_jet'
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


def pixelate(jet, mask=None, npix=64, maxR=1.0):
    """Pixelate the jet with Raghav Kansal's method.
    Reference: https://github.com/rkansal47/mnist_graph_gan/blob/neurips21/jets/final_plots.py#L191-L204

    Args
    ----
    jet : np.ndarray
        Momenta in polar coordinates with shape (num_jets, 3, num_jet_particles)
    mask : np.ndarray
        Mask of data.
        Default: None
    npix : int
        Number of pixels of the jet image.
        Default: 64
    maxR : int
        Maximum radius of the jet image.
        Default: 1.0

    Return
    ------
    Pixelated jet with shape (npix, npix).
    """
    bins = np.linspace(-maxR, maxR, npix + 1)
    pt = jet[:, 0]
    binned_eta = np.digitize(jet[:, 1], bins) - 1
    binned_phi = np.digitize(jet[:, 2], bins) - 1
    if mask is not None:
        pt *= mask

    jet_image = np.zeros((npix, npix))

    for eta, phi, pt in zip(binned_eta, binned_phi, pt):
        if eta >= 0 and eta < npix and phi >= 0 and phi < npix:
            jet_image[phi, eta] += pt

    return jet_image


def get_jet_rel(jets):
    """Get jet momenta in relative coordinates (ptrel, etarel, phirel).

    Args
    ----
    jets : `numpy.ndarray` or `torch.Tensor`
        The jets in absolute polar coordinates.

    Return
    ------
    jets : `numpy.ndarray`
        The jet momenta in relative coordinates.
    """
    import awkward as ak
    from coffea.nanoevents.methods import vector
    ak.behavior.update(vector.behavior)

    if isinstance(jets, torch.Tensor):
        jets = jets.detach().cpu().numpy()

    part_vecs = ak.zip(
        {
            "pt": jets[:, :, 0:1],
            "eta": jets[:, :, 1:2],
            "phi": jets[:, :, 2:3],
            "mass": np.zeros_like(jets[:, :, 1:2])
        }, with_name="PtEtaPhiMLorentzVector")

    # sum over all the particles in each jet to get the jet 4-vector
    jet_vecs = part_vecs.sum(axis=1)[:, :2]

    # subtract the jet eta, phi from each particle to convert to normalized coordinates
    jets[:, :, 1] -= ak.to_numpy(jet_vecs.eta)
    jets[:, :, 2] -= ak.to_numpy(jet_vecs.phi)

    # divide each particle pT by jet pT if we want relative jet pT
    jets[:, :, 0] /= ak.to_numpy(jet_vecs.pt)

    return jets


def get_average_jet_image(jets, maxR=0.5, npix=64, abs_coord=True):
    """Get the average jet image from a collection of jets.

    Args
    ----
    jets : `numpy.ndarray` or `torch.Tensor`
        A collection of jets in polar coordinates.
    maxR : `float`
        Maximum radius.
        Default: 0.5
    npix : `int`
        Number of pixels.
        Default: 64
    abs_coord : `bool`
        Whether jets are in absolute coordinates.
        Default: True

    Return
    ------
    jet_image : `numpy.ndarray`
        The average jet image over the collection.
    """

    if abs_coord:
        jets = get_jet_rel(jets)
    jet_image = [pixelate(jets[i], mask=None, npix=npix, maxR=maxR)
                 for i in range(len(jets))]
    jet_image = np.stack(jet_image, axis=0)
    jet_image = np.mean(jet_image, axis=0)
    return jet_image


def plot_jet_image(args, p4_target, p4_gen, save_dir, epoch, vmin=1e-10, show=False):
    """Plot jet image, one for target jets and one for generated/reconstructed jets.

    Args
    ---
    p4_target : np.ndarray
        Target jets in polar coordinates (pt, eta, phi).
        Shape : (num_jets, num_particles, 3)
    p4_gen : np.ndarray
        Generated/reconstructed jets by the model in polar coordinates (pt, eta, phi).
        Shape : (num_jets, num_particles, 3)
    save_dir : str
        Parent directory for plots.
    epoch : int
        The current epoch.
    """

    target_pix = get_average_jet_image(p4_target, npix=args.jet_image_npix)
    gen_pix = get_average_jet_image(p4_gen, npix=args.jet_image_npix)

    from matplotlib.colors import LogNorm

    # # Auto log scale
    # fig, axs = plt.subplots(1, 2, figsize=JET_IMAGE_FIGSIZE)
    # target = axs[0].imshow(target_pix, origin='lower', norm=LogNorm(), vmin=vmin, vmax=1)
    # fig.colorbar(target, ax=axs[0])
    # gen = axs[1].imshow(gen_pix, origin='lower', norm=LogNorm(), vmin=vmin, vmax=1)
    # fig.colorbar(gen, ax=axs[1])
    # # Set labels
    # for i in range(2):
    #     axs[i].set_xlabel(r"$\Delta\phi^\mathrm{rel}$")
    #     axs[i].set_ylabel(r"$\Delta\eta^\mathrm{rel}$")
    #
    # save_dir_auto = make_dir(osp.join(save_dir, 'jet_image'))
    # filename = f'{args.jet_type}_jet_image_epoch_{epoch+1}'
    # plt.tight_layout()
    # plt.savefig(osp.join(save_dir_auto, f'{filename}.pdf'), bbox_inches="tight", dpi=args.jet_image_dpi)
    # plt.close()

    # Same log scale
    fig, axs = plt.subplots(1, 2, figsize=JET_IMAGE_FIGSIZE)
    target = axs[0].pcolor(target_pix, norm=LogNorm(vmin=vmin, vmax=1))
    fig.colorbar(target, ax=axs[0])
    gen = axs[1].pcolor(gen_pix, norm=LogNorm(vmin=vmin, vmax=1))
    fig.colorbar(gen, ax=axs[1])
    # Set labels
    for i in range(2):
        axs[i].set_xlabel(r"$\Delta\phi^\mathrm{rel}$")
        axs[i].set_ylabel(r"$\Delta\eta^\mathrm{rel}$")

    save_dir_samelog = make_dir(osp.join(save_dir, 'jet_image'))
    filename = f'{args.jet_type}_jet_image_epoch_{epoch+1}'
    plt.tight_layout()
    plt.savefig(osp.join(save_dir_samelog, f'{filename}.pdf'), bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def get_magnitude(p, device='gpu'):
    """Get the momentum magnitude |p| of the 4-vector.
    Args
    ----
    p : `numpy.ndarray` or `torch.Tensor`
        The 4-momentum.

    Return
    ------
    |p| = sq
    """
    if isinstance(p, np.ndarray):
        return np.sqrt(np.sum(np.power(p, 2)[..., 1:], axis=-1))
    elif isinstance(p, torch.Tensor):
        if device in GPU_STR:
            p = p.to(device=DEVICE)
        return torch.sqrt(torch.sum(torch.pow(p, 2)[..., 1:], dim=-1)).detach().cpu()
    else:
        raise ValueError(f"The input must be numpy.ndarray or torch.Tensor. Found: {type(p)}.")


def get_p_cartesian(jets, cutoff=1e-6):
    """Get (px, py, pz) from the jet data and filter out values that are too small.

    Args
    ----
    jets : `numpy.ndarray`
        The jet data, with shape (num_particles, 4), which means all jets are merged together.
    cutoff : `float`
        The cutoff value of 3-momenta.

    Return
    ------
    A tuple (px, py, pz). Each is a numpy.ndarray.
    """
    if isinstance(jets, np.ndarray):
        jets = np.copy(jets).reshape(-1, 4)
        px = jets[:, 1].copy()
        py = jets[:, 2].copy()
        pz = jets[:, 3].copy()
        p = get_magnitude(jets)  # |p| of 3-momenta
        if cutoff > 0:
            px[p < cutoff] = np.nan
            py[p < cutoff] = np.nan
            pz[p < cutoff] = np.nan
    elif isinstance(jets, torch.Tensor):
        jets = torch.clone(jets).reshape(-1, 4)
        px = torch.clone(jets[:, 1]).detach().cpu().numpy()
        py = torch.clone(jets[:, 2]).detach().cpu().numpy()
        pz = torch.clone(jets[:, 3]).detach().cpu().numpy()
        p = get_magnitude(jets)  # |p| of 3-momenta
        if cutoff > 0:
            px[p < cutoff] = np.nan
            py[p < cutoff] = np.nan
            pz[p < cutoff] = np.nan
    else:
        raise ValueError(f"The input must be numpy.ndarray or torch.Tensor. Found: {type(jets)}.")

    return px, py, pz


def get_p_polar(p4, cutoff=1e-6, eps=1e-12, device='gpu'):
    """
    Get (pt, eta, phi) from the jet data.

    Args
    ----
    p4 : `torch.Tensor`
        The jet data, with shape (num_particles, 4), which means all jets are merged together.

    Return
    ------
    Particle momenta in polar coordinates as a numpy.ndarray.
    """
    if isinstance(p4, np.ndarray):
        px, py, pz = get_p_cartesian(p4, cutoff=cutoff)
        pt = np.sqrt(px ** 2 + py ** 2 + eps)
        eta = np.arcsinh(pz / pt)
        phi = np.arctan2(py, px + eps)
    elif isinstance(p4, torch.Tensor):
        if device in GPU_STR:
            p4 = p4.to(device=DEVICE)

        p_polar = get_p_polar_tensor(p4)
        pt = p_polar[..., 0].detach().cpu().numpy()
        eta = p_polar[..., 1].detach().cpu().numpy()
        phi = p_polar[..., 2].detach().cpu().numpy()

        if cutoff > 0:
            p = get_magnitude(p4).detach().cpu().numpy()
            pt[p < cutoff] = np.nan
            eta[p < cutoff] = np.nan
            phi[p < cutoff] = np.nan

    return pt, eta, phi


def get_p_polar_tensor(p, eps=1e-16):
    """(E, px, py, pz) -> (pt, eta, phi)"""
    px = p[..., 1]
    py = p[..., 2]
    pz = p[..., 3]

    pt = torch.sqrt(px ** 2 + py ** 2)
    try:
        eta = torch.asinh(pz / (pt + eps))
    except AttributeError:
        eta = arcsinh(pz / pt)
    phi = torch.atan2(py + eps, px)

    return torch.stack((pt, eta, phi), dim=-1)


def arcsinh(z):
    return torch.log(z + torch.sqrt(1 + torch.pow(z, 2)))


def get_jet_feature_cartesian(p4, device='gpu'):
    """
    Get jet (m, pt, eta, phi) from the jet data.

    Args
    ----
    jet_data : `numpy.ndarray` or `torch.Tensor`
        The jet data, with shape (num_particles, 4), which means all jets are merged together.
    """

    if isinstance(p4, np.ndarray):
        jet_p4 = np.sum(p4, axis=-2)
        msq = jet_p4[:, 0] ** 2 - np.sum(np.power(jet_p4, 2)[:, 1:], axis=-1)
        jet_mass = np.sqrt(np.abs(msq)) * np.sign(msq)
        jet_px = jet_p4[:, 1]
        jet_py = jet_p4[:, 2]
        jet_pz = jet_p4[:, 3]
    elif isinstance(p4, torch.Tensor):  # torch.Tensor
        if device in GPU_STR:
            p4 = p4.to(device=DEVICE)
        jet_p4 = torch.sum(p4, axis=-2)
        msq = jet_p4[:, 0] ** 2 - torch.sum(torch.pow(jet_p4, 2)[:, 1:], axis=-1)
        jet_mass = (torch.sqrt(torch.abs(msq)) * torch.sign(msq)).detach().cpu()
        jet_px = jet_p4[:, 1].detach().cpu()
        jet_py = jet_p4[:, 2].detach().cpu()
        jet_pz = jet_p4[:, 3].detach().cpu()
    else:
        raise ValueError(f"The input must be numpy.ndarray or torch.Tensor. Found: {type(p4)}.")

    return jet_mass, jet_px, jet_py, jet_pz


def get_jet_feature_polar(p4, device='gpu', eps=1e-16):
    """
    Get jet (m, pt, eta, phi) from the jet data.

    Args
    ----
    jet_data : `numpy.ndarray`
        The jet data, with shape (num_particles, 4), which means all jets are merged together.
    """

    m, px, py, pz = get_jet_feature_cartesian(p4)

    if isinstance(p4, np.ndarray):
        pt = np.sqrt(px ** 2 + py ** 2)
        eta = np.arcsinh(pz / (pt + eps))
        phi = np.arctan2(py, px)
        return m, pt, eta, phi
    elif isinstance(p4, torch.Tensor):
        if device == 'gpu':
            p4 = p4.to(device=DEVICE)
        pt = torch.sqrt(px ** 2 + py ** 2)
        try:
            eta = torch.arcsinh(pz / (pt + eps))
        except AttributeError:
            eta = arcsinh(pz / (pt + eps))
        phi = torch.atan2(py, px)
        return m.detach().cpu().numpy(), pt.detach().cpu().numpy(), eta.detach().cpu().numpy(), phi.detach().cpu().numpy()
    else:
        raise ValueError(f"The input must be numpy.ndarray or torch.Tensor. Found: {type(p4)}.")


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
