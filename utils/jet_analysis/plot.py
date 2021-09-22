import numpy as np
from utils.jet_analysis.jet_features import plot_jet_p_cartesian, plot_jet_p_polar
from utils.jet_analysis.particle_features import plot_p_cartesian, plot_p_polar
from utils.jet_analysis.jet_images import plot_jet_image
from utils.jet_analysis.utils import get_p_polar, get_p_cartesian, get_jet_feature_polar, get_jet_feature_cartesian, get_p_polar_tensor
from utils.jet_analysis.particle_recon_err import plot_particle_recon_err
from utils.jet_analysis.jet_recon_err import plot_jet_recon_err
from utils.utils import get_eps


def plot_p(args, p4_target, p4_gen, save_dir, particle_recon_err=False, cutoff=1e-6, epoch=None, show=False):
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

    plot_p_polar(args, p_target_polar, p_gen_polar, save_dir, epoch=epoch, density=False, fill=False, show=show)
    plot_jet_p_polar(args, jet_target_polar, jet_gen_polar, save_dir, epoch=epoch, density=False, fill=False, show=show)
    plot_p_cartesian(args, p_target_cartesian, p_gen_cartesian, save_dir, epoch=epoch, density=False, fill=False, show=show)
    plot_jet_p_cartesian(args, jet_target_cartesian, jet_gen_cartesian, save_dir, epoch=epoch, density=False, fill=False, show=show)
    if args.fill:  # Plot filled histograms in addition to unfilled histograms
        plot_p_polar(args, p_target_polar, p_gen_polar, save_dir, epoch=epoch, density=False, fill=True, show=show)
        plot_p_cartesian(args, p_target_cartesian, p_target_polar, save_dir, epoch=epoch, density=False, fill=True, show=show)
        plot_jet_p_polar(args, jet_target_polar, jet_gen_polar, save_dir, epoch=epoch, density=False, fill=True, show=show)
        plot_jet_p_cartesian(args, jet_gen_cartesian, jet_gen_cartesian, save_dir, epoch=epoch, density=False, fill=True, show=show)

    plot_jet_image(args, jets_target, jets_gen, save_dir, epoch, maxR=0.5, vmin=args.jet_image_vmin, show=show)

    if particle_recon_err:
        plot_particle_recon_err(args, p4_target[..., 1:], p4_gen[..., 1:], save_dir=save_dir, epoch=epoch)

    plot_jet_recon_err(args, jet_target_cartesian, jet_gen_cartesian, jet_target_polar, jet_gen_polar,
                       save_dir, epoch=epoch)
