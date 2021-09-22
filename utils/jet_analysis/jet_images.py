from utils.utils import make_dir
import os.path as osp
import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_jet_image(args, p4_target, p4_gen, save_dir, epoch, maxR=0.5, vmin=1e-8, show=False):
    """Plot average jet image and one-to-one jet image

    Parameters
    ----------
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
    vmin : float
        vmin for LogNorm of average jet image.
        A positive number.
        Default: 1e-8
    show : bool
        Whether to show jet images.

    """

    target_pix_average = get_average_jet_image(p4_target, npix=args.jet_image_npix,
                                               abs_coord=args.abs_coord)
    gen_pix_average = get_average_jet_image(p4_gen, npix=args.jet_image_npix,
                                            abs_coord=args.abs_coord)

    target_pix = get_n_jet_images(p4_target, num_jets=args.num_jet_images, npix=args.jet_image_npix,
                                  maxR=maxR, abs_coord=args.abs_coord)
    gen_pix = get_n_jet_images(p4_gen, num_jets=args.num_jet_images, npix=args.jet_image_npix,
                               maxR=maxR, abs_coord=args.abs_coord)

    # Same log scale
    fig, axs = plt.subplots(
        len(target_pix)+1, 2, figsize=get_one_to_one_jet_image_figsize(num_jets=len(target_pix))
    )

    from copy import copy
    cm = copy(plt.cm.jet)
    cm.set_under(color='white')

    for i, axs_row in enumerate(axs):
        if i == 0:
            from matplotlib.colors import LogNorm
            target = axs_row[0].imshow(target_pix_average, norm=LogNorm(vmin=vmin, vmax=1), origin='lower', cmap=cm,
                                       interpolation='nearest', extent=[-maxR, maxR, -maxR, maxR])
            axs_row[0].title.set_text('Average Target Jet')

            gen = axs_row[1].imshow(gen_pix_average, norm=LogNorm(vmin=vmin, vmax=1), origin='lower', cmap=cm,
                                    interpolation='nearest', extent=[-maxR, maxR, -maxR, maxR])
            axs_row[1].title.set_text('Average Reconstructed Jet')

            cbar_target = fig.colorbar(target, ax=axs_row[0])
            cbar_gen = fig.colorbar(gen, ax=axs_row[1])
        else:
            target = axs_row[0].imshow(target_pix[i-1], origin='lower', cmap=cm, interpolation='nearest',
                                       vmin=vmin, extent=[-maxR, maxR, -maxR, maxR], vmax=0.05)
            axs_row[0].title.set_text('Target Jet')

            gen = axs_row[1].imshow(gen_pix[i-1], origin='lower', cmap=cm, interpolation='nearest',
                                    vmin=vmin, extent=[-maxR, maxR, -maxR, maxR], vmax=0.05)
            axs_row[1].title.set_text('Reconstructed Jet')
            cbar_target = fig.colorbar(target, ax=axs_row[0])
            cbar_gen = fig.colorbar(gen, ax=axs_row[1])

        for cbar in [cbar_target, cbar_gen]:
            cbar.set_label(r'$p_\mathrm{T}$')

        for j in range(len(axs_row)):
            axs_row[j].set_xlabel(r"$\phi^\mathrm{rel}$")
            axs_row[j].set_ylabel(r"$\eta^\mathrm{rel}$")

    plt.tight_layout()
    filename = f'{args.jet_type}_jet_images_epoch_{epoch+1}.pdf'
    save_dir = make_dir(osp.join(save_dir, 'jet_images'))
    plt.savefig(osp.join(save_dir, filename), bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def pixelate(jet, mask=None, npix=64, maxR=1.0):
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


def get_jet_rel(jets):
    """Get jet momenta in relative coordinates (ptrel, etarel, phirel).

    Parameters
    ----------
    jets : `numpy.ndarray` or `torch.Tensor`
        The jets in absolute polar coordinates.

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
    jet_vecs = part_vecs.sum(axis=1)[:, :2]

    # subtract the jet eta, phi from each particle to convert to normalized coordinates
    jets[:, :, 1] -= ak.to_numpy(jet_vecs.eta)
    jets[:, :, 2] -= ak.to_numpy(jet_vecs.phi)

    # divide each particle pT by jet pT if we want relative jet pT
    jets[:, :, 0] /= ak.to_numpy(jet_vecs.pt)

    return jets


def get_average_jet_image(jets, maxR=0.5, npix=64, abs_coord=True):
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
    jet_image = [pixelate(jets[i], mask=None, npix=npix, maxR=maxR)
                 for i in range(len(jets))]
    jet_image = np.stack(jet_image, axis=0)
    jet_image = np.mean(jet_image, axis=0)
    return jet_image


def get_n_jet_images(jets, num_jets=15, maxR=0.5, npix=24, abs_coord=True):
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
    jet_image = [pixelate(jets[i], mask=None, npix=npix, maxR=maxR)
                 for i in range(min(num_jets, len(jets)))]
    jet_image = np.stack(jet_image, axis=0)
    return jet_image


def get_one_to_one_jet_image_figsize(num_jets=15):
    """Returns the figure size of one-to-one jet images"""
    return (7.5, 3*num_jets)
