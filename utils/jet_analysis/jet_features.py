from typing import Optional
from utils.jet_analysis.utils import get_jet_name, NUM_BINS, PLOT_FONT_SIZE
from utils.utils import make_dir
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

FIGSIZE = (16, 4)
LABELS_CARTESIAN_ABS_COORD = (
    r"$M$ (GeV)",
    r"$P_x$ (GeV)",
    r"$P_y$ (GeV)",
    r"$P_z$ (GeV)",
)
LABELS_CARTESIAN_REL_COORD = (
    r"$M^\mathrm{rel}$",
    r"$P_x^\mathrm{rel}$",
    r"$P_y^\mathrm{rel}$",
    r"$P_z^\mathrm{rel}$",
)
LABELS_POLAR_ABS_COORD = (r"$M$ (GeV)", r"$P_\mathrm{T}$ (GeV)", r"$\eta$", r"$\phi$")
LABELS_POLAR_REL_COORD = (
    r"$M^\mathrm{rel}$",
    r"$P_\mathrm{T}^\mathrm{rel}$",
    r"$\eta^\mathrm{rel}$",
    r"$\phi^\mathrm{rel}$",
)

RANGES_CARTESIAN_ABS_COORD = (
    np.linspace(0, 250, NUM_BINS),  # Jet M
    np.linspace(-2000, 2000, NUM_BINS),  # Jet Px
    np.linspace(-2000, 2000, NUM_BINS),  # Jet Py
    np.linspace(-4000, 4000, NUM_BINS),  # Jet Pz
)
RANGES_CARTESIAN_REL_COORD = (
    np.linspace(0, 0.2, NUM_BINS),  # Jet M_rel
    np.linspace(0.5, 1.02, NUM_BINS),  # Jet Px_rel
    np.linspace(-0.02, 0.02, NUM_BINS),  # Jet Px_rel
    np.linspace(-0.02, 0.02, NUM_BINS),  # Jet Pz_rel
)
RANGES_POLAR_ABS_COORD = (
    np.linspace(0, 250, NUM_BINS),  # Jet M
    np.linspace(0, 2000, NUM_BINS),  # Jet Pt
    np.linspace(-2, 2, NUM_BINS),  # Jet eta
    np.linspace(-np.pi, np.pi, NUM_BINS),  # Jet phi
)
RANGES_POLAR_REL_COORD = (
    np.linspace(0, 5, NUM_BINS),  # Jet M_rel
    np.linspace(0.5, 1.02, NUM_BINS),  # Jet Pt_rel
    np.linspace(-0.02, 0.02, NUM_BINS),  # Jet eta_rel
    np.linspace(-0.02, 0.02, NUM_BINS),  # Jet phi_rel
)


def plot_jet_p_cartesian(
    jet_features_target: np.ndarray,
    jet_features_recons: np.ndarray,
    abs_coord: bool,
    save_dir: str,
    jet_type: str = "",
    epoch: Optional[int] = None,
    density: bool = False,
    fill: bool = False,
    show: bool = False,
):
    """Plot jet features (m, px, py, pz) distribution.

    :param jet_features_target: target jet momenta, with shape (num_jets, 4).
    :type jet_features_target: np.ndarray
    :param jet_features_recons: enerated/reconstructed jet momenta, with shape (num_jets, 4).
    :type jet_features_recons: np.ndarray
    :param abs_coord: whether to use absolute coordinates.
        If False, use relative coordinates.
    :type abs_coord: bool
    :param save_dir: directory to save the figure.
    :type save_dir: str
    :param jet_type: the string description of jet, defaults to "".
        Example: 'g' for gluon jets and 't' for top quark jets.
    :type jet_type: str, optional
    :param epoch: current epoch, defaults to None
    :type epoch: Optional[int], optional
    :param density: whether to plot distribution density, defaults to False.
    :type density: bool, optional
    :param fill: whether bins are filled., defaults to True
    :type fill: bool, optional
    :param show: whether to show plot, defaults to False
    :type show: bool, optional
    """
    if abs_coord:
        ranges = RANGES_CARTESIAN_ABS_COORD
        names = LABELS_CARTESIAN_ABS_COORD
    else:
        ranges = RANGES_CARTESIAN_REL_COORD
        names = LABELS_CARTESIAN_REL_COORD

    fig, axs = plt.subplots(1, 4, figsize=FIGSIZE, sharey=False)
    for ax, p_target, p_recons, bins, name in zip(
        axs, jet_features_target, jet_features_recons, ranges, names
    ):
        if not fill:
            ax.hist(
                p_recons.flatten(),
                bins=bins,
                histtype="step",
                stacked=True,
                fill=False,
                label="reconstructed",
                density=density,
            )
            ax.hist(
                p_target.flatten(),
                bins=bins,
                histtype="step",
                stacked=True,
                fill=False,
                label="target",
                density=density,
            )
        else:
            ax.hist(
                p_recons.flatten(),
                bins=bins,
                alpha=0.6,
                label="reconstructed",
                density=density,
            )
            ax.hist(
                p_target.flatten(),
                bins=bins,
                alpha=0.6,
                label="target",
                density=density,
            )

        ax.set_xlabel(f"Jet {name}")
        ax.set_ylabel("Number of jets")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        for axis in ("x", "y"):
            ax.tick_params(axis=axis, labelsize=PLOT_FONT_SIZE)
        ax.tick_params(bottom=True, top=True, left=True, right=True, direction="in")
        ax.tick_params(
            labelbottom=True, labeltop=False, labelleft=True, labelright=False
        )

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=2)
    plt.rcParams.update({"font.size": PLOT_FONT_SIZE})

    fig.tight_layout()

    jet_name = get_jet_name(jet_type)
    if abs_coord:
        fig.suptitle(
            rf"Distribution of target and reconstructed jet $M$, $P_x$, $P_y$, and $P_z$ of {jet_name} jet",
            y=1.03,
        )
    else:
        fig.suptitle(
            "Distribution of target and reconstructed jet "
            + r"$M^\mathrm{rel}$, $P_x^\mathrm{rel}$, $P_y^\mathrm{rel}$, and $P_z^\mathrm{rel}$ of "
            + f"{jet_name} jet",
            y=1.03,
        )

    if epoch is not None:
        save_dir = make_dir(osp.join(save_dir, "jet_features/cartesian"))
        if fill:
            save_dir = make_dir(osp.join(save_dir, "filled"))
    else:
        pass  # Save without creating a subdirectory

    filename = f"jet_features_cartesian_{jet_type}_jet"
    if epoch is not None:
        filename = f"{filename}_epoch_{epoch+1}"
    if density:
        filename = f"{filename}_density"
    if fill:
        filename = f"{filename}_fill"
    plt.savefig(osp.join(save_dir, f"{filename}.pdf"), bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_jet_p_polar(
    jet_features_target: np.ndarray,
    jet_features_recons: np.ndarray,
    abs_coord: bool,
    save_dir: str,
    jet_type: str = "",
    epoch: Optional[int] = None,
    density: bool = False,
    fill: bool = True,
    show: bool = False,
) -> None:
    """Plot jet features (m, pt, eta, phi) distribution.

    :param jet_features_target: target jet momenta, with shape (num_jets, 4).
    :type jet_features_target: np.ndarray
    :param jet_features_recons: enerated/reconstructed jet momenta, with shape (num_jets, 4).
    :type jet_features_recons: np.ndarray
    :param abs_coord: whether to use absolute coordinates.
        If False, use relative coordinates.
    :type abs_coord: bool
    :param save_dir: directory to save the figure.
    :type save_dir: str
    :param jet_type: the string description of jet, defaults to "".
        Example: 'g' for gluon jets and 't' for top quark jets.
    :type jet_type: str, optional
    :param epoch: current epoch, defaults to None
    :type epoch: Optional[int], optional
    :param density: whether to plot distribution density, defaults to False.
    :type density: bool, optional
    :param fill: whether bins are filled., defaults to True
    :type fill: bool, optional
    :param show: whether to show plot, defaults to False
    :type show: bool, optional
    """

    if abs_coord:
        ranges = RANGES_POLAR_ABS_COORD
        names = LABELS_POLAR_ABS_COORD
    else:
        ranges = RANGES_POLAR_REL_COORD
        names = LABELS_POLAR_REL_COORD

    fig, axs = plt.subplots(1, 4, figsize=FIGSIZE, sharey=False)
    for ax, p_target, p_recons, bins, name in zip(
        axs, jet_features_target, jet_features_recons, ranges, names
    ):
        if not fill:
            ax.hist(
                p_recons.flatten(),
                bins=bins,
                histtype="step",
                stacked=True,
                fill=False,
                label="reconstructed",
                density=density,
            )
            ax.hist(
                p_target.flatten(),
                bins=bins,
                histtype="step",
                stacked=True,
                fill=False,
                label="target",
                density=density,
            )
        else:
            ax.hist(
                p_recons.flatten(),
                bins=range,
                alpha=0.6,
                label="reconstructed",
                density=density,
            )
            ax.hist(
                p_target.flatten(),
                bins=range,
                alpha=0.6,
                label="target",
                density=density,
            )

        ax.set_xlabel(f"Jet {name}")
        ax.set_ylabel("Number of jets")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        for axis in ("x", "y"):
            ax.tick_params(axis=axis, labelsize=PLOT_FONT_SIZE)
        ax.tick_params(bottom=True, top=True, left=True, right=True, direction="in")
        ax.tick_params(
            labelbottom=True, labeltop=False, labelleft=True, labelright=False
        )

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=2)
    plt.rcParams.update({"font.size": PLOT_FONT_SIZE})

    fig.tight_layout()

    jet_name = get_jet_name(jet_type)
    if abs_coord:
        fig.suptitle(
            r"Distribution of target and reconstructed jet $M$, $P_\mathrm{T}$, $\eta$, and $\phi$ "
            + f"of {jet_name} jets",
            y=1.03,
        )
    else:
        fig.suptitle(
            r"Distribution of target and reconstructed jet $M^\mathrm{rel}$, $P_\mathrm{T}^\mathrm{rel}$, $\eta^\mathrm{rel}$, and $\phi^\mathrm{rel}$ "
            + f"of {jet_name} jets",
            y=1.03,
        )

    if epoch is not None:
        save_dir = make_dir(osp.join(save_dir, "jet_features/polar"))
        if fill:
            save_dir = osp.join(save_dir, "filled")
    else:
        pass  # Save without creating a subdirectory

    filename = f"jet_features_polar_{jet_type}_jet"
    if epoch is not None:
        filename = f"{filename}_epoch_{epoch+1}"
    if density:
        filename = f"{filename}_density"
    if fill:
        filename = f"{filename}_fill"
    plt.savefig(osp.join(save_dir, f"{filename}.pdf"), bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
