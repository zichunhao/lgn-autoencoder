from argparse import Namespace
from typing import Optional, Tuple

import torch
import numpy as np
from .jet_features import plot_jet_p_cartesian, plot_jet_p_polar
from .particle_features import plot_p_cartesian, plot_p_polar
from .jet_images import plot_jet_image
from .utils import get_p_polar, get_p_cartesian
from .utils import get_jet_feature_polar, get_jet_feature_cartesian, get_p4_cartesian_from_polar
from .utils import get_p_polar_tensor, get_recons_err_ranges
from .particle_recon_err import plot_particle_recon_err
from .jet_recon_err import plot_jet_recon_err
from utils.utils import get_eps


def plot_p(
    args: Namespace, 
    p4_target: torch.Tensor, 
    p4_recons: torch.Tensor, 
    save_dir: str, 
    cutoff: float = 1e-6, 
    epoch: Optional[int] = None, 
    show: bool = False
) -> Tuple[np.ndarray, ...]:
    """
    Plot particle features, jet features, reconstruction errors, and jet images.

    :param p4_target: target jets with shape (num_jets, num_particles, 4)
    :type p4_target: torch.Tensor
    :param p4_recons: generated/reconstructed jets with shape (num_jets, num_particles, 4)
    :type p4_recons: torch.Tensor
    :param save_dir: directory to save plots to.
    :type save_dir: str
    :param cutoff: cutoff value for |p| = sqrt(px^2 + py^2 + pz^2), defaults to 1e-6.
        Particle momentum lower than `cutoff` will be considered padded particles and thus dropped.
        Used for plotting particle distributions to avoid big spikes due to padding.
        Default: 1e-6
    :type cutoff: float, optional
    :param epoch: current epoch, defaults to None
    :type epoch: Optional[int], optional
    :param show: whether to show plot, defaults to False
    :type show: bool, optional
    :return: the jet images.
    :rtype: _type_
    """    
    """Plot particle features, jet features, and jet images.

    Parameters
    ----------
    p4_target :  `torch.Tensor`
        The target jets.
        Shape: (num_jets, num_particles, 4)
    p4_recons :  `torch.Tensor`
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
    
    if args.polar_coord:
        p4_target = get_p4_cartesian_from_polar(p4_target)
        p4_recons = get_p4_cartesian_from_polar(p4_recons)

    p_target_polar = get_p_polar(p4_target, cutoff=cutoff, eps=EPS, return_arr=False)
    jet_target_polar = get_jet_feature_polar(p4_target, return_arr=False)
    p_recons_polar = get_p_polar(p4_recons, cutoff=cutoff, eps=EPS, return_arr=False)
    jet_recons_polar = get_jet_feature_polar(p4_recons)

    p_target_cartesian = get_p_cartesian(p4_target.detach().cpu().numpy(), cutoff=cutoff)
    jet_target_cartesian = get_jet_feature_cartesian(p4_target.detach().cpu().numpy())
    p_recons_cartesian = get_p_cartesian(p4_recons.detach().cpu().numpy(), cutoff=cutoff)
    jet_recons_cartesian = get_jet_feature_cartesian(p4_recons.detach().cpu().numpy())

    plot_p_polar(
        p3_polar_target=p_target_polar, 
        p3_polar_recons=p_recons_polar, 
        abs_coord=args.abs_coord,
        jet_type=args.jet_type,
        save_dir=save_dir, 
        epoch=epoch, 
        density=False, 
        fill=False, 
        show=show
    )
    plot_jet_p_polar(
        jet_features_target=jet_target_polar, 
        jet_features_recons=jet_recons_polar, 
        save_dir=save_dir, 
        abs_coord=args.abs_coord,
        jet_type=args.jet_type,
        epoch=epoch, 
        density=False, 
        fill=False, 
        show=show
    )
    plot_p_cartesian(
        p3_targets=p_target_cartesian, 
        p3_recons=p_recons_cartesian, 
        save_dir=save_dir, 
        abs_coord=args.abs_coord,
        jet_type=args.jet_type,
        epoch=epoch, 
        density=False, 
        fill=False, 
        show=show
    )
    plot_jet_p_cartesian(
        jet_features_target=jet_target_cartesian, 
        jet_features_recons=jet_recons_cartesian, 
        save_dir=save_dir, 
        abs_coord=args.abs_coord,
        jet_type=args.jet_type,
        epoch=epoch, 
        density=False, 
        fill=True, 
        show=show
    )
    
    if args.fill:  # Plot filled histograms in addition to unfilled histograms
        plot_p_polar(
            p3_polar_target=p_target_polar, 
            p3_polar_recons=p_recons_polar, 
            abs_coord=args.abs_coord,
            jet_type=args.jet_type,
            save_dir=save_dir, 
            epoch=epoch, 
            density=False, 
            fill=True, 
            show=show
        )
        plot_jet_p_polar(
            jet_features_target=jet_target_polar, 
            jet_features_recons=jet_recons_polar, 
            save_dir=save_dir, 
            abs_coord=args.abs_coord,
            jet_type=args.jet_type,
            epoch=epoch, 
            density=False, 
            fill=True, 
            show=show
        )
        plot_p_cartesian(
            p3_targets=p_target_cartesian, 
            p3_recons=p_recons_cartesian, 
            save_dir=save_dir, 
            abs_coord=args.abs_coord,
            jet_type=args.jet_type,
            epoch=epoch, 
            density=False, 
            fill=True, 
            show=show
        )
        plot_jet_p_cartesian(
            jet_features_target=jet_target_cartesian, 
            jet_features_recons=jet_recons_cartesian, 
            save_dir=save_dir, 
            abs_coord=args.abs_coord,
            jet_type=args.jet_type,
            epoch=epoch, 
            density=False, 
            fill=True, 
            show=show
        )

    jet_images_list = []
    for same_norm in (True, False):
        jets_target = get_p_polar_tensor(p4_target, eps=EPS)
        jets_recons = get_p_polar_tensor(p4_recons, eps=EPS)
        target_pix_average, gen_pix_average, target_pix, gen_pix = plot_jet_image(
            p_target=jets_target, 
            p_recons=jets_recons, 
            maxR=args.jet_image_maxR,
            abs_coord=args.abs_coord,
            jet_type=args.jet_type,
            save_dir=save_dir, 
            epoch=epoch, 
            num_jet_images=args.num_jet_images,
            jet_image_npix=args.jet_image_npix,
            same_norm=same_norm, 
            vmin=args.jet_image_vmin, 
            show=show
        )
        jet_images_dict = {
            'target_average': target_pix_average,
            'target': target_pix,
            'reconstructed_average': gen_pix_average,
            'reconstructed': gen_pix
        }
        jet_images_list.append(jet_images_dict)

    particle_recons_ranges, jet_recons_ranges = get_recons_err_ranges(args)

    plot_particle_recon_err(
        p_target=p4_target, 
        p_recons=p4_recons,
        abs_coord=args.abs_coord,
        custom_particle_recons_ranges=args.custom_particle_recons_ranges,
        find_match=('mse' not in args.loss_choice.lower()),
        ranges=particle_recons_ranges, 
        save_dir=save_dir, 
        epoch=epoch
    )

    plot_jet_recon_err(
        jet_target_cartesian=jet_target_cartesian, 
        jet_recons_cartesian=jet_recons_cartesian, 
        abs_coord=args.abs_coord,
        jet_target_polar=jet_target_polar, 
        jet_recons_polar=jet_recons_polar,
        custom_jet_recons_ranges=args.custom_jet_recons_ranges,
        save_dir=save_dir, 
        ranges=jet_recons_ranges, 
        epoch=epoch
    )

    return tuple(jet_images_list)
