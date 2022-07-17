from typing import Optional, Tuple, Union
from utils.utils import make_dir
import os.path as osp
import numpy as np
import torch
import matplotlib.pyplot as plt
import awkward as ak
from coffea.nanoevents.methods import vector
ak.behavior.update(vector.behavior)

IMG_VMAX = 0.05


def plot_jet_image(
    p_target: np.ndarray, 
    p_recons: np.ndarray, 
    save_dir: str, 
    epoch: int,
    num_jet_images: int,
    jet_image_npix: int,
    abs_coord: bool,
    jet_type: bool = "",
    same_norm: bool = True,
    maxR: bool = 0.5, 
    vmin: bool = 1e-8, 
    show: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Plot average jet image and one-to-one jet image

    Parameters
    ----------
    p_target : np.ndarray
        Target jets in polar coordinates (pt, eta, phi).
        Shape : (num_jets, num_particles, 3)
    p_recons : np.ndarray
        Generated/reconstructed jets by the model in polar coordinates (pt, eta, phi).
        Shape : (num_jets, num_particles, 3)
    save_dir : str
        Parent directory for plots.
    epoch : int
        The current epoch.
    same_norm : bool, optional
        Whether to use the same normalization.
        If true, reconstructed jets will be normalized based on the target.
        Default: True
    show : bool, optional
        Whether to show jet images.

    """
    if same_norm and abs_coord:
        target_pix_average, recons_pix_average = get_average_jet_image_same_norm(
            jet_target=p_target, 
            jet_recons=p_recons, 
            npix=jet_image_npix, 
            maxR=maxR, 
        )
        target_pix, recons_pix = get_n_jet_images_same_norm(
            jet_target=p_target, 
            jet_recons=p_recons, 
            num_jets=num_jet_images, 
            maxR=maxR,
            npix=jet_image_npix
        )
    else:
        target_pix_average = get_average_jet_image(
            jets=p_target, 
            maxR=maxR,
            npix=jet_image_npix,
            abs_coord=abs_coord
        )
        recons_pix_average = get_average_jet_image(
            jets=p_recons,
            maxR=maxR, 
            npix=jet_image_npix,
            abs_coord=abs_coord
        )

        target_pix = get_n_jet_images(
            jets=p_target, 
            maxR=maxR,
            num_jets=num_jet_images,
            npix=jet_image_npix,
            abs_coord=abs_coord
        )
        recons_pix = get_n_jet_images(
            jets=p_recons, 
            maxR=maxR,
            num_jets=num_jet_images, 
            npix=jet_image_npix,
            abs_coord=abs_coord
        )

    fig, axs = plt.subplots(
        len(target_pix)+1, 2, 
        figsize=get_one_to_one_jet_image_figsize(num_jets=len(target_pix))
    )

    from copy import copy
    cm = copy(plt.cm.jet)
    cm.set_under(color='white')

    for i, axs_row in enumerate(axs):
        if i == 0:
            from matplotlib.colors import LogNorm
            target = axs_row[0].imshow(
                target_pix_average, 
                norm=LogNorm(vmin=vmin, vmax=1),
                origin='lower', 
                cmap=cm,
                interpolation='nearest', 
                extent=[-maxR, maxR, -maxR, maxR]
            )
            axs_row[0].title.set_text('Average Target Jet')

            recons = axs_row[1].imshow(
                recons_pix_average, 
                norm=LogNorm(vmin=vmin, vmax=1), 
                origin='lower', 
                cmap=cm,
                interpolation='nearest', 
                extent=[-maxR, maxR, -maxR, maxR]
            )
            axs_row[1].title.set_text('Average Reconstructed Jet')

            cbar_target = fig.colorbar(target, ax=axs_row[0])
            cbar_recons = fig.colorbar(recons, ax=axs_row[1])
        else:
            target = axs_row[0].imshow(
                target_pix[i-1], 
                origin='lower', 
                cmap=cm, 
                interpolation='nearest',
                vmin=vmin, 
                extent=[-maxR, maxR, -maxR, maxR], 
                vmax=IMG_VMAX
            )
            axs_row[0].title.set_text('Target Jet')

            recons = axs_row[1].imshow(
                recons_pix[i-1], 
                origin='lower', 
                cmap=cm, 
                interpolation='nearest',
                vmin=vmin, 
                extent=[-maxR, maxR, -maxR, maxR], 
                vmax=IMG_VMAX
            )
            axs_row[1].title.set_text('Reconstructed Jet')
            cbar_target = fig.colorbar(target, ax=axs_row[0])
            cbar_recons = fig.colorbar(recons, ax=axs_row[1])

        for cbar in [cbar_target, cbar_recons]:
            cbar.set_label(r'$p_\mathrm{T}$')

        for j in range(len(axs_row)):
            axs_row[j].set_xlabel(r"$\phi^\mathrm{rel}$")
            axs_row[j].set_ylabel(r"$\eta^\mathrm{rel}$")

    plt.tight_layout()

    if epoch is not None:
        filename = f'{jet_type}_jet_images_epoch_{epoch+1}.pdf'
        if same_norm:
            save_dir = make_dir(osp.join(save_dir, 'jet_images_same_norm'))
        else:
            save_dir = make_dir(osp.join(save_dir, 'jet_images'))
    else:  # Save without creating a subdirectory
        if same_norm:
            filename = f'{jet_type}_jet_images_same_norm.pdf'
        else:
            filename = f'{jet_type}_jet_images.pdf'
    plt.savefig(osp.join(save_dir, filename), bbox_inches="tight")

    if show:
        plt.show()
    plt.close()

    return target_pix_average, recons_pix_average, target_pix, recons_pix


def pixelate(
    jet: np.ndarray, 
    mask: Optional[np.ndarray] = None, 
    npix: int = 64,
    maxR: float = 1.0
) -> np.ndarray:
    """Pixelate the jet with Raghav Kansal's method.
    Reference: https://github.com/rkansal47/mnist_graph_gan/blob/neurips21/jets/final_plots.py#L191-L204

    Parameters
    ----------
    jet : np.ndarray
        Momenta in polar coordinates with shape (num_jets, 3, num_jet_particles)
    mask : np.ndarray, optional
        Mask of data.
        Default: None
    npix : int, optional
        Number of pixels of the jet image.
        Default: 64
    maxR : int, optional
        Maximum radius of the jet image.
        Default: 1.0

    Returns
    -------
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


def get_jet_rel(
    jets: np.ndarray
) -> np.ndarray:
    """Get jet momenta in relative coordinates (ptrel, etarel, phirel).

    Parameters
    ----------
    jets : `numpy.ndarray` or `torch.Tensor`

    Returns
    -------
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
    try:
        jet_vecs = part_vecs.sum(axis=1)[:, :2]
    except AttributeError:
        jet_vecs = ak.sum(part_vecs, axis=1)[:, :2]

    jets = normalize(jets, jet_vecs)
    return jets


def get_average_jet_image(
    jets: Union[np.ndarray, torch.Tensor], 
    maxR: float = 0.5, 
    npix: int = 64, 
    abs_coord: bool = True
):
    """Get the average jet image from a collection of jets.

    Parameters
    ----------
    jets : `numpy.ndarray` or `torch.Tensor`
        A collection of jets in polar coordinates.
    maxR : float
        Maximum radius.
        Default: 0.5
    npix : int
        Number of pixels.
        Default: 64
    abs_coord : bool
        Whether jets are in absolute coordinates.
        Default: True

    Returns
    -------
    jet_image : `numpy.ndarray`
        The average jet image over the collection.
    """

    if abs_coord:
        jets = get_jet_rel(jets)
    jet_image = [
        pixelate(jets[i], mask=None, npix=npix, maxR=maxR)
        for i in range(len(jets))
    ]
    jet_image = np.stack(jet_image, axis=0)
    jet_image = np.mean(jet_image, axis=0)
    return jet_image


def get_n_jet_images(
    jets: Union[np.ndarray, torch.Tensor],
    num_jets: int = 15, 
    maxR: float = 0.5, 
    npix: int = 24, 
    abs_coord: int = True
):
    """Get the first num_jets jet images from a collection of jets.

    Parameters
    ----------
    jets : `numpy.ndarray` or `torch.Tensor`
        A collection of jets in polar coordinates.
    num_jets : int
        The number of jet images to produce.
    maxR : float
        Maximum radius.
        Default: 0.5
    npix : int
        Number of pixels.
        Default: 64
    abs_coord : bool
        Whether jets are in absolute coordinates.
        Default: True

    Returns
    -------
    jet_image : `numpy.ndarray`
        The first num_jets jet images.
    """

    if abs_coord:
        jets = get_jet_rel(jets)
    jet_image = [
        pixelate(jets[i], mask=None, npix=npix, maxR=maxR)
        for i in range(min(num_jets, len(jets)))
    ]
    jet_image = np.stack(jet_image, axis=0)
    return jet_image


def get_jet_rel_same_norm(
    jet_target: Union[np.ndarray, torch.Tensor], 
    jet_recons: Union[np.ndarray, torch.Tensor]
) -> Tuple[np.ndarray, np.ndarray]:
    """Get jet momenta in relative coordinates 
    (ptrel, etarel, phirel) 
    using the coordinates of target jet.

    Parameters
    ----------
    jet_target : `numpy.ndarray` or `torch.Tensor`
        Target jet.
    jet_recons : `numpy.ndarray` or `torch.Tensor`
        Reconstructed jet.

    Returns
    -------
    (jet_target, jet_recons) in relative coordinates.
    """
    if isinstance(jet_target, torch.Tensor):
        jet_target = jet_target.detach().cpu().numpy()
    if isinstance(jet_recons, torch.Tensor):
        jet_recons = jet_recons.detach().cpu().numpy()

    part_vecs = ak.zip(
        {
            "pt": jet_target[:, :, 0:1],
            "eta": jet_target[:, :, 1:2],
            "phi": jet_target[:, :, 2:3],
            "mass": np.zeros_like(jet_target[:, :, 1:2])
        }, with_name="PtEtaPhiMLorentzVector")

    # sum over all the particles in each jet to get the jet 4-vector
    jet_vecs = part_vecs.sum(axis=1)[:, :2]

    jet_target = normalize(jet_target, jet_vecs)
    jet_recons = normalize(jet_recons, jet_vecs)

    return jet_target, jet_recons


def get_average_jet_image_same_norm(
    jet_target: Union[np.ndarray, torch.Tensor], 
    jet_recons: Union[np.ndarray, torch.Tensor], 
    maxR: float = 0.5, 
    npix: int = 64
):
    """Get the average jet image, in relative coordinate with respect to the target jet, from a collection of jets.

    Parameters
    ----------
    jet_target : `numpy.ndarray` or `torch.Tensor`
        A collection of target jets in polar coordinates.
    jet_recons : `numpy.ndarray` or `torch.Tensor`
        A collection of reconstructed/generated jets in polar coordinates.
    maxR : float
        Maximum radius.
        Default: 0.5
    npix : int
        Number of pixels.
        Default: 64

    Returns
    -------
    The average jet images (target, recons), 
    in relative coordinates with respect to the target jets, 
    over the collection.
    """
    jet_target, jet_recons = get_jet_rel_same_norm(jet_target, jet_recons)
    target_image = [
        pixelate(jet_target[i], mask=None, npix=npix, maxR=maxR)
        for i in range(len(jet_target))
    ]
    target_image = np.stack(target_image, axis=0)
    target_image = np.mean(target_image, axis=0)

    recons_image = [
        pixelate(jet_recons[i], mask=None, npix=npix, maxR=maxR)
        for i in range(len(jet_recons))
    ]
    recons_image = np.stack(recons_image, axis=0)
    recons_image = np.mean(recons_image, axis=0)

    return target_image, recons_image


def get_n_jet_images_same_norm(
    jet_target: Union[np.ndarray, torch.Tensor], 
    jet_recons: Union[np.ndarray, torch.Tensor], 
    num_jets: int = 15, 
    maxR: float = 0.5, 
    npix: int = 24
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the first num_jets jet images from a collection of jets.

    Parameters
    ----------
    jet_target : `numpy.ndarray` or `torch.Tensor`
        A collection of target jets in polar coordinates.
    jet_recons : `numpy.ndarray` or `torch.Tensor`
        A collection of reconstructed/generated jets in polar coordinates.
    num_jets : int
        The number of jet images to produce.
    maxR : float
        Maximum radius.
        Default: 0.5
    npix : int
        Number of pixels.
        Default: 64

    Returns
    -------
    (target_image, recons_image) : The first num_jets jet images.
    """
    jet_target, jet_recons = get_jet_rel_same_norm(jet_target, jet_recons)

    target_image = [
        pixelate(jet_target[i], mask=None, npix=npix, maxR=maxR)
        for i in range(min(num_jets, len(jet_target)))
    ]
    target_image = np.stack(target_image, axis=0)

    recons_image = [
        pixelate(jet_recons[i], mask=None, npix=npix, maxR=maxR)
        for i in range(min(num_jets, len(jet_recons)))
    ]
    recons_image = np.stack(recons_image, axis=0)

    return target_image, recons_image


def normalize(
    jet: np.ndarray, 
    jet_vecs: ak.Array
) -> np.ndarray:
    """Normalize jet based on jet_vecs.

    Parameters
    ----------
    jet : numpy.ndarray
        The jet to normalize.
    jet_vecs : awkward.array
        Jet vectors
    """
    jet[:, :, 1] -= ak.to_numpy(jet_vecs.eta)
    jet[:, :, 2] -= ak.to_numpy(jet_vecs.phi)
    jet[:, :, 0] /= ak.to_numpy(jet_vecs.pt)
    return jet


def get_one_to_one_jet_image_figsize(
    num_jets: int = 15
) -> Tuple[float, float]:
    """Returns the figure size of one-to-one jet images"""
    return (7.5, 3*num_jets)
