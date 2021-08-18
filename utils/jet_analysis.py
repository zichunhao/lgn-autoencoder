import energyflow
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp

from utils.utils import make_dir

# Figure sizes
PARTICLE_FEATURE_FIGSIZE = (12, 4)
JET_FEATURE_FIGSIZE = (16, 4)
JET_IMAGE_FIGSIZE = (7.5, 3)


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
    for same_log_scale in [True, False]:
        plot_jet_image(args, target_data, gen_data, save_dir, epoch, same_log_scale=same_log_scale)

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
    """Plot p distribution in Cartesian coordinates.

    Args
    ----
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
    """Plot jet features (m, px, py, pz) distribution.

    Args
    ----
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

    jet_features_target = get_jet_feature_cartesian(target_data)
    jet_features_gen = get_jet_feature_cartesian(gen_data)

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


def plot_p_polar(args, target_data, gen_data, save_dir, max_val=(200, 2, np.pi),
                 num_bins=201, cutoff=1e-6, epoch=None, density=False, fill=True, show=False):
    """Plot p distribution in polar coordinates (pt, eta, phi)

    Args
    ----
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

    pt_target, eta_target, phi_target = get_p_polar(target_data, cutoff=cutoff)
    pt_gen, eta_gen, phi_gen = get_p_polar(gen_data, cutoff=cutoff)

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


def plot_jet_p_polar(args, target_data, gen_data, save_dir, max_val=(30000, 2000, 2, np.pi),
                     num_bins=201, epoch=None, density=False, fill=True, show=False):
    """Plot jet features (m, pt, eta, phi) distribution.

    Args
    ----
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


def get_p_cartesian(jet_data, cutoff=1e-6):
    """Get (px, py, pz) from the jet data and filter out values that are too small.

    Args
    ----
    jet_data : `numpy.Array`
        The jet data, with shape (num_particles, 4), which means all jets are merged together.
    cutoff : `float`
        The cutoff value of 3-momenta.

    Return
    ------
    A tuple (px, py, pz)
    """
    jet_data = np.copy(jet_data).reshape(-1, 4)
    px = jet_data[:, 1].copy()
    py = jet_data[:, 2].copy()
    pz = jet_data[:, 3].copy()

    p = get_magnitude(jet_data)  # |p| of 3-momenta
    if cutoff > 0:
        px[p < cutoff] = np.nan
        py[p < cutoff] = np.nan
        pz[p < cutoff] = np.nan

    return px, py, pz


def pixelate(p_polar, npix=64):
    """Pixelate the jet using energyflow. More on https://energyflow.network/docs/utils/

    Args
    ---
    p_polar : np.Array
        The momentum of a jet in polar coordinates (pt, eta, phi).
        Note that pt should be below O(100). Otherwise, the pixelate of energyflow will break down.
        Shape: (num_particles, 3)
    npix : int
        The number of pixels generated. This is an argument of the energyflow method.

    Return
    ------
    Pixelated jet with shape (npix, npix, 1).
    """
    p_polar = np.concatenate((p_polar, np.ones((p_polar.shape[0], 1))), axis=-1).transpose()
    try:
        p_polar_pix = energyflow.utils.pixelate(p_polar, npix=npix)
    except FloatingPointError:
        return None
    return p_polar_pix


def get_average_jet_image(p4, num_particles=30, npix=64):
    """Get an average jet image over the whole dataset.
    Args
    ---
    p4 : np.Array
        The 4-momenta of jets in the whole dataset.
    num_particles : int
        Number of particles per jet
    npix : int
        The number of pixels generated. This is an argument of the energyflow method.

    Return
    ------
    An average jet image with shape (npix, npix, 1).
    """
    pt, eta, phi = get_p_polar(p4, cutoff=-1)  # Remove any cutoff

    pt = np.reshape(np.expand_dims(pt, axis=-1), (-1, num_particles, 1)) / 1000  # Convert back to TeV
    eta = np.reshape(np.expand_dims(eta, axis=-1), (-1, num_particles, 1))
    phi = np.reshape(np.expand_dims(phi, axis=-1), (-1, num_particles, 1))
    p_polar = np.concatenate((pt, eta, phi), axis=-1)

    pix = [pixelate(p_polar[i], npix=npix) for i in range(p_polar.shape[0])]
    pix = list(filter(lambda x: x is not None, pix))
    pix = np.stack(pix, axis=0)
    pix = np.mean(pix, axis=0)  # Get average
    return pix


def plot_jet_image(args, p4_target, p4_gen, save_dir, epoch, same_log_scale=True):
    """Plot jet image, one for target jets and one for generated/reconstructed jets.

    Args
    ---
    p4_target : np.Array
        Target jets.
        Shape : (num_jets, num_particles, 4)
    p4_gen : np.Array
        Generated/reconstructed jets by the model.
        Shape : (num_jets, num_particles, 4)
    save_dir : str
        Parent directory for plots.
    epoch : int
        The current epoch.
    same_log_scale : bool
        Whether to use the same log scale for the two images.
        Default: True
    """

    target_pix = get_average_jet_image(p4_target, num_particles=args.num_jet_particles, npix=args.jet_image_npix)  # Removing cutoff
    gen_pix = get_average_jet_image(p4_gen, num_particles=args.num_jet_particles, npix=args.jet_image_npix)

    from matplotlib.colors import LogNorm
    fig, axs = plt.subplots(1, 2, figsize=JET_IMAGE_FIGSIZE)

    # Target jet image
    if same_log_scale:
        target = axs[0].imshow(target_pix, origin='lower', norm=LogNorm(), vmin=1e-11, vmax=1)
    else:
        target = axs[0].imshow(target_pix, origin='lower', norm=LogNorm(vmin=1e-11, vmax=1))
    axs[0].title.set_text('Average Target Jet')
    fig.colorbar(target, ax=axs[0])

    # Generated/reconstructed jet image
    if same_log_scale:
        gen = axs[1].imshow(gen_pix, origin='lower', norm=LogNorm(), vmin=1e-11, vmax=1)
    else:
        gen = axs[1].imshow(gen_pix, origin='lower', norm=LogNorm(vmin=1e-11, vmax=1))
    axs[1].title.set_text('Average Reconstructed Jet')
    fig.colorbar(gen, ax=axs[1])

    # Set labels
    for i in range(2):
        axs[i].set_xlabel(r"$\Delta\eta$ cell")
        axs[i].set_ylabel(r"$\Delta\phi$ cell")

    # Save figure
    if same_log_scale:
        save_dir = make_dir(osp.join(save_dir, 'jet_image_samelog'))
    else:
        save_dir = make_dir(osp.join(save_dir, 'jet_image'))
    filename = f'{args.jet_type}_jet_image_epoch_{epoch+1}'
    if same_log_scale:
        filename += 'samelog'

    plt.tight_layout()
    plt.savefig(osp.join(save_dir, f'{filename}.pdf'), bbox_inches="tight", dpi=args.jet_image_dpi)
    plt.close()


def get_magnitude(p):
    return np.sqrt(np.sum(np.power(p, 2)[..., 1:], axis=-1))


def get_p_polar(jet_data, cutoff=1e-6):
    """
    Get (pt, eta, phi) from the jet data.

    Args
    ----
    jet_data : `numpy.Array`
        The jet data, with shape (num_particles, 4), which means all jets are merged together.
    """

    px, py, pz = get_p_cartesian(jet_data, cutoff=cutoff)

    pt = np.sqrt(px ** 2 + py ** 2)
    eta = np.arcsinh(pz / (pt + 1e-16))
    phi = np.arctan2(py, px)

    return pt, eta, phi


def get_jet_feature_cartesian(p4):
    """
    Get jet (m, pt, eta, phi) from the jet data.

    Args
    ----
    jet_data : `numpy.Array`
        The jet data, with shape (num_particles, 4), which means all jets are merged together.
    """

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

    Args
    ----
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
