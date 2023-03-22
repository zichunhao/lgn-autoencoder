from typing import Callable, Iterable, Optional, Tuple, Union
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import make_dir
from .utils import NUM_BINS, PLOT_FONT_SIZE, find_fwhm, get_stats
from scipy import stats
import os.path as osp
import logging
import json

FIGSIZE = (16, 4)
LABELS_CARTESIAN_ABS_COORD = (r"$M$", r"$P_x$", r"$P_y$", r"$P_z$")
LABELS_POLAR_ABS_COORD = (r"$M$", r"$P_\mathrm{T}$", r"$\eta$", r"$\phi$")
LABELS_ABS_COORD = (LABELS_CARTESIAN_ABS_COORD, LABELS_POLAR_ABS_COORD)
LABELS_CARTESIAN_REL_COORD = (
    r"$M^\mathrm{rel}$",
    r"$P_x^\mathrm{rel}$",
    r"$P_y^\mathrm{rel}$",
    r"$P_z^\mathrm{rel}$",
)
LABELS_POLAR_REL_COORD = (
    r"$M^\mathrm{rel}$",
    r"$P_\mathrm{T}^\mathrm{rel}$",
    r"$\eta^\mathrm{rel}$",
    r"$\phi^\mathrm{rel}$",
)
LABELS_REL_COORD = (LABELS_CARTESIAN_REL_COORD, LABELS_POLAR_REL_COORD)
COORDINATES = ("cartesian", "polar")
DEFAULT_BIN_RANGE = 2
MAX_BIN_RANGE = 5


def plot_jet_recon_err(
    jet_target_cartesian: np.ndarray,
    jet_recons_cartesian: np.ndarray,
    jet_target_polar: np.ndarray,
    jet_recons_polar: np.ndarray,
    save_dir: str,
    abs_coord: bool,
    custom_jet_recons_ranges: bool,
    epoch: Optional[int] = None,
    eps: float = 1e-16,
    drop_zeros: bool = True,
    ranges: Optional[np.ndarray] = None,
    get_rel_err: Callable = (
        lambda p_target, p_recons, eps: (p_target - p_recons) / (p_target + eps)
    ),
    show: bool = False,
) -> None:
    """Plot reconstruction errors for jet."""
    if drop_zeros:
        jet_target_cartesian, jet_recons_cartesian = filter_out_zeros(
            jet_target_cartesian, jet_recons_cartesian
        )
        jet_target_polar, jet_recons_polar = filter_out_zeros(
            jet_target_polar, jet_recons_polar
        )

    rel_err_cartesian = [
        get_rel_err(jet_recons_cartesian[i], jet_target_cartesian[i], eps)
        for i in range(4)
    ]
    rel_err_polar = [
        get_rel_err(jet_recons_polar[i], jet_target_polar[i], eps) for i in range(4)
    ]
    if (not custom_jet_recons_ranges) or (ranges is None):
        ranges = get_bins(
            NUM_BINS, rel_err_cartesian=rel_err_cartesian, rel_err_polar=rel_err_polar
        )

    LABELS = LABELS_ABS_COORD if abs_coord else LABELS_REL_COORD
    err_dict = dict()
    for rel_err_coordinate, labels, coordinate, bin_tuple in zip(
        (rel_err_cartesian, rel_err_polar), LABELS, COORDINATES, ranges
    ):
        stats_coordinate_list = []
        fig, axs = plt.subplots(1, 4, figsize=FIGSIZE, sharey=False)
        for ax, rel_err, bins, label in zip(axs, rel_err_coordinate, bin_tuple, labels):
            stats = get_stats(rel_err, bins)
            stats_coordinate_list.append(stats)

            if not custom_jet_recons_ranges:
                # Find the range based on the FWHM
                median = stats["median"]
                iqr = stats["IQR"]
                bins_suitable = np.linspace(
                    median - 4 * iqr, median + 4 * iqr, NUM_BINS
                )
                ax.hist(rel_err, bins=bins_suitable, histtype="step", stacked=True)
            else:
                ax.hist(rel_err, bins=bins, histtype="step", stacked=True)
            ax.set_xlabel(rf"$\delta${label}")
            ax.set_ylabel("Number of Jets")
            ax.ticklabel_format(
                axis="x", style="sci", scilimits=(-2, 0), useMathText=True
            )
            ax.ticklabel_format(
                axis="y", style="sci", scilimits=(0, 0), useMathText=True
            )

            for axis in ("x", "y"):
                ax.tick_params(axis=axis, labelsize=PLOT_FONT_SIZE)

        err_dict[coordinate] = stats_coordinate_list

        plt.rcParams.update({"font.size": PLOT_FONT_SIZE})
        plt.tight_layout()

        if save_dir:
            if epoch is not None:
                path = make_dir(
                    osp.join(save_dir, f"jet_reconstruction_errors/{coordinate}")
                )
                plt.savefig(
                    osp.join(path, f"jet_reconstruction_errors_epoch_{epoch+1}.pdf")
                )
                dict_path = make_dir(osp.join(path, "err_dict"))
                file_name = f"jet_reconstruction_errors_epoch_{epoch+1}.json"
                with open(osp.join(dict_path, file_name), "w") as f:
                    json.dump(err_dict, f)
            else:  # Save without creating a subdirectory
                plt.savefig(
                    osp.join(save_dir, f"jet_reconstruction_errors_{coordinate}.pdf")
                )
                dict_path = osp.join(save_dir, "jet_reconstruction_errors.json")
                with open(dict_path, "w") as f:
                    json.dump(err_dict, f)
        if show:
            plt.show()
        plt.close()

    logging.debug("Jet reconstruction errors:")
    logging.debug(err_dict)


def default_get_rel_err(
    p_target: Union[np.ndarray, torch.Tensor],
    p_recons: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-16,
    alpha: float = 0.01,
):
    if type(p_target) is torch.Tensor:
        p_target = p_target.cpu().detach().numpy()
    if type(p_recons) is torch.Tensor:
        p_recons = p_recons.cpu().detach().numpy()
    return (p_target - p_recons) / (p_target + alpha * np.median(p_target) + eps)


def get_bins(
    num_bins: int,
    rel_err_cartesian: Optional[Tuple[float, float, float, float]] = None,
    rel_err_polar: Optional[Tuple[float, float, float, float]] = None,
    alpha: float = 1.5,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """Get bins for jet reconstruction error plots."""
    if rel_err_cartesian is None:
        cartesian_min_max = (
            (-1, 10),
            (-DEFAULT_BIN_RANGE, DEFAULT_BIN_RANGE),
            (-DEFAULT_BIN_RANGE, DEFAULT_BIN_RANGE),
            (-DEFAULT_BIN_RANGE, DEFAULT_BIN_RANGE),
        )
    else:
        err = rel_err_cartesian[0]
        num_components = len(err)
        medians = [np.median(err[..., i]) for i in range(num_components)]
        iqrs = [stats.iqr(err[..., i]) for i in range(num_components)]
        cartesian_min_max = tuple(
            [
                (medians[i] - alpha * iqrs[i], medians[i] + alpha * iqrs[i])
                for i in range(num_components)
            ]
        )

    if rel_err_polar is None:
        polar_min_max = (
            (-1, DEFAULT_BIN_RANGE),
            (-1, DEFAULT_BIN_RANGE),
            (-DEFAULT_BIN_RANGE, DEFAULT_BIN_RANGE),
            (-DEFAULT_BIN_RANGE, DEFAULT_BIN_RANGE),
        )
    else:
        err = rel_err_polar[0]
        num_components = len(err)
        medians = [np.median(err[..., i]) for i in range(num_components)]
        iqrs = [stats.iqr(err[..., i]) for i in range(num_components)]
        polar_min_max = tuple(
            [
                (medians[i] - alpha * iqrs[i], medians[i] + alpha * iqrs[i])
                for i in range(num_components)
            ]
        )

    ranges_cartesian = tuple(
        [
            np.linspace(*cartesian_min_max[i], num_bins)
            for i in range(len(cartesian_min_max))
        ]
    )

    ranges_polar = tuple(
        [np.linspace(*polar_min_max[i], num_bins) for i in range(len(polar_min_max))]
    )

    ranges = (ranges_cartesian, ranges_polar)
    return ranges


def get_legend(res: np.ndarray, bins: np.ndarray) -> str:
    """Get legend for plots of jet reconstruction."""
    legend = r"$\mu$: " + f"{np.mean(res) :.2E},\n"
    # legend += r'$\sigma$: ' + f'{np.std(res) :.3E} \n'
    legend += r"$\mathrm{FWHM}$: " + f"{find_fwhm(res, bins) :.2E}"
    # legend += r'$\mathrm{Med}$: ' + f'{np.median(res) :.3E}'
    return legend


def filter_out_zeros(
    target: Iterable[np.ndarray], recons: Iterable[np.ndarray]
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """Filter out jets with any zero component.

    Parameters
    ----------
    target : iterable of `numpy.ndarray`.
        Target jet components.
    recons : iterable of `numpy.ndarray`.
        Generated/reconstructed jet components.

    Returns
    -------
    target_filtered, recons_filtered
    """
    mask = (target[0] != 0) & (target[1] != 0) & (target[2] != 0) & (target[3] != 0)
    target_filtered = tuple([target[i][mask] for i in range(4)])
    recons_filtered = tuple([recons[i][mask] for i in range(4)])
    return target_filtered, recons_filtered
