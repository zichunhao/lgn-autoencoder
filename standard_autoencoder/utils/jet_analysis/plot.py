from .jet_features import plot_jet_p_cartesian, plot_jet_p_polar
from .particle_features import plot_p_cartesian, plot_p_polar
from .jet_images import plot_jet_image
from .utils import get_p_polar, get_p_cartesian, get_p4, get_p4_cartesian_from_polar
from .utils import get_jet_feature_polar, get_jet_feature_cartesian
from .utils import get_p_polar_tensor, get_recons_err_ranges
from .particle_recon_err import plot_particle_recon_err
from .jet_recon_err import plot_jet_recon_err
from utils.utils import get_eps


def plot_p(args, p4_target, p4_gen, save_dir, cutoff=1e-6, epoch=None, show=False):
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
    if p4_target.shape[-1] == 3:
        p4_target = get_p4(p4_target)
    if p4_gen.shape[-1] == 3:
        p4_gen = get_p4(p4_gen)

    # tuples
    if args.polar_coord:
        p4_target = get_p4_cartesian_from_polar(p4_target)
        p4_gen = get_p4_cartesian_from_polar(p4_gen)
    
    p_target_polar = get_p_polar(p4_target, cutoff=cutoff, eps=EPS)
    jet_target_polar = get_jet_feature_polar(p4_target)
    p_gen_polar = get_p_polar(p4_gen, cutoff=cutoff, eps=EPS)
    jet_gen_polar = get_jet_feature_polar(p4_gen)

    p_target_cartesian = get_p_cartesian(p4_target.detach().cpu().numpy(), cutoff=cutoff)
    jet_target_cartesian = get_jet_feature_cartesian(p4_target.detach().cpu().numpy())
    p_gen_cartesian = get_p_cartesian(p4_gen.detach().cpu().numpy(), cutoff=cutoff)
    jet_gen_cartesian = get_jet_feature_cartesian(p4_gen.detach().cpu().numpy())

    plot_p_polar(args, p_target_polar, p_gen_polar, save_dir, epoch=epoch, density=False, fill=False, show=show)
    plot_jet_p_polar(args, jet_target_polar, jet_gen_polar, save_dir, epoch=epoch, density=False, fill=False, show=show)
    plot_p_cartesian(args, p_target_cartesian, p_gen_cartesian, save_dir, epoch=epoch, density=False, fill=False, show=show)
    plot_jet_p_cartesian(args, jet_target_cartesian, jet_gen_cartesian, save_dir, epoch=epoch, density=False, fill=False, show=show)
    if args.fill:  # Plot filled histograms in addition to unfilled histograms
        plot_p_polar(args, p_target_polar, p_gen_polar, save_dir, epoch=epoch, density=False, fill=True, show=show)
        plot_p_cartesian(args, p_target_cartesian, p_target_polar, save_dir, epoch=epoch, density=False, fill=True, show=show)
        plot_jet_p_polar(args, jet_target_polar, jet_gen_polar, save_dir, epoch=epoch, density=False, fill=True, show=show)
        plot_jet_p_cartesian(args, jet_gen_cartesian, jet_gen_cartesian, save_dir, epoch=epoch, density=False, fill=True, show=show)

    jet_images_list = []
    for same_norm in (True, False):
        jets_target = get_p_polar_tensor(p4_target, eps=EPS)
        jets_gen = get_p_polar_tensor(p4_gen, eps=EPS)
        target_pix_average, gen_pix_average, target_pix, gen_pix = plot_jet_image(
            args, jets_target, jets_gen, save_dir, epoch, same_norm=same_norm, maxR=0.5, vmin=args.jet_image_vmin, show=show
        )
        jet_images_dict = {
            'target_average': target_pix_average,
            'target': target_pix,
            'reconstructed_average': gen_pix_average,
            'reconstructed': gen_pix
        }
        jet_images_list.append(jet_images_dict)

    particle_recons_ranges, jet_recons_ranges = get_recons_err_ranges(args)

    plot_particle_recon_err(args, p4_target, p4_gen, find_match=('mse' not in args.loss_choice.lower()),
                            ranges=particle_recons_ranges, save_dir=save_dir, epoch=epoch)

    plot_jet_recon_err(args, jet_target_cartesian, jet_gen_cartesian, jet_target_polar, jet_gen_polar,
                       save_dir, ranges=jet_recons_ranges, epoch=epoch)

    return tuple(jet_images_list)
