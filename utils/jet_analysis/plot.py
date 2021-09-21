import numpy as np
from utils.jet_analysis.jet_features import plot_jet_p_cartesian, plot_jet_p_polar
from utils.jet_analysis.particle_features import plot_p_cartesian, plot_p_polar
from utils.jet_analysis.jet_images import plot_jet_image
from utils.jet_analysis.utils import get_p_polar, get_p_cartesian, get_jet_feature_polar, get_jet_feature_cartesian, get_p_polar_tensor
from utils.jet_analysis.particle_recon_err import plot_particle_recon_err
from utils.jet_analysis.jet_recon_err import plot_jet_recon_err
from utils.utils import get_eps


def plot_p(args, p4_target, p4_gen, save_dir,
           cartesian_max=(100, 100, 100), polar_max=(200, 2, np.pi),
           jet_cartesian_max=(200, 2000, 2000, 4000), jet_polar_max=(200, 4000, 2, np.pi),
           particle_recon_err=False, num_bins=81, cutoff=1e-6, epoch=None, show=False):
    """Plot particle features, jet features, and jet images.

    Parameters
    ----------
    p4_target :  `torch.Tensor`
        The target jets.
        Shape: (num_jets, num_particles, 4)
    p4_gen :  `torch.Tensor`
        The generated/reconstructed jets by model.
        Shape: (num_jets, num_particles, 4)
    save_dir : str
        The saving directories for figures.
    cartesian_max : tuple, optional
        The maximum values, (px_max, py_max, pz_max), for particle feature plots in cartesian coordinates.
            - px will range from -px_max to px_max.
            - py will range from -py_max to py_max.
            - pz will range from -pz_max to pz_max.
        Default: (100, 100, 100)
    polar_max : tuple, optional
        The maximum values, (pt_max, eta_max, phi_max), for particle feature plots in polar coordinates.
            - pt will range from 0 to pt_max.
            - eta will range from -eta_max to eta_max.
            - phi will range from -phi_max to phi_max.
        Default: (200, 2, np.pi)
    jet_cartesian_max : tuple, optional
        The maximum values, (m_max, pt_max, eta_max, phi_max), for jet feature plots in polar coordinates.
            - jet mass (m) will range from 0 to m_max.
            - pt will range from 0 to pt_max.
            - eta will range from -eta_max to eta_max.
            - phi will range from -phi_max to phi_max.
        Default: (200, 2000, 2000, 4000)
    jet_polar_max : tuple, optional
        The maximum values, (pt_max, eta_max, phi_max), for jet feature plots in polar coordinates.
            - jet mass (M) will range from 0 to M_max.
            - pt will range from 0 to pt_max.
            - eta will range from -eta_max to eta_max.
            - phi will range from -phi_max to phi_max.
        Default: (200, 4000, 2, np.pi)
    plot_particle_recon_err : bool, optional
        Whether to plot reconstruction error for particle features. Only used with one-to-one mapping.
        Default: False
    num_bins : int, optional
        Number of bins for histograms of particle and jet features.
        Default: 201
    cutoff : float, optional
        The cutoff value for |p| = sqrt(px^2 + py^2 + pz^2).
        Particle momentum lower than `cutoff` will be considered padded particles and thus dropped.
        Default: 1e-6
    epoch : int, optional
        The epoch number.
        Default: None
    show : bool, optional
        Whether to show plots.
        Default: False
    """

    EPS = get_eps(args)

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
    plot_jet_image(args, jets_target, jets_gen, save_dir, epoch, maxR=0.5, vmin=args.jet_image_vmin, show=show)

    if args.fill:  # Plot filled histograms in addition to unfilled histograms
        plot_p_polar(args, p_target_polar, p_gen_polar, save_dir, max_val=polar_max, num_bins=num_bins,
                     epoch=epoch, density=False, fill=True, show=show)
        plot_p_cartesian(args, p_target_cartesian, p_target_polar, save_dir, max_val=cartesian_max,
                         num_bins=num_bins, epoch=epoch, density=False, fill=True, show=show)
        plot_jet_p_polar(args, jet_target_polar, jet_gen_polar, save_dir, max_val=jet_polar_max,
                         num_bins=81, epoch=epoch, density=False, fill=True, show=show)
        plot_jet_p_cartesian(args, jet_gen_cartesian, jet_gen_cartesian, save_dir, max_val=jet_cartesian_max,
                             num_bins=81, epoch=epoch, density=False, fill=True, show=show)

    if particle_recon_err:
        plot_particle_recon_err(args, p4_target[..., 1:], p4_gen[..., 1:], save_dir=save_dir, epoch=epoch)

    plot_jet_recon_err(args, jet_target_cartesian, jet_gen_cartesian, jet_target_polar, jet_gen_polar,
                       save_dir, epoch=epoch)
