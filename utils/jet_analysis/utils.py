import torch
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GPU_STR = ['cuda', 'gpu']


def get_magnitude(p, device='gpu'):
    """Get the momentum magnitude |p| of the 4-vector.
    Parameters
    ----------
    p : `numpy.ndarray` or `torch.Tensor`
        The 4-momentum.

    Returns
    -------
    |p| = sq

    Raises
    ------
    ValueError
        If p is not of type numpy.ndarray or torch.Tensor.
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

    Parameters
    ----------
    jets : `numpy.ndarray`
        The jet data, with shape (num_particles, 4), which means all jets are merged together.
    cutoff : float
        The cutoff value of 3-momenta.

    Returns
    -------
    A tuple (px, py, pz). Each is a numpy.ndarray.

    Raises
    ------
    ValueError
        If p is not of type numpy.ndarray or torch.Tensor.
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

    Parameters
    ----------
    p4 : `torch.Tensor`
        The jet data, with shape (num_particles, 4), which means all jets are merged together.

    Returns
    -------
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

    Parameters
    ----------
    p4 : `numpy.ndarray` or `torch.Tensor`
        The jet data, with shape (num_particles, 4), which means all jets are merged together.
    device : str, optional
        The device for computation when type(p4) is torch.Tensor.
        Default: 'gpu'.

    Raises
    ------
    ValueError
        If p is not of type numpy.ndarray or torch.Tensor.
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

    Parameters
    ----------
    p4 : `numpy.ndarray` or `torch.Tensor`
        The jet data, with shape (num_particles, 4), which means all jets are merged together.

    Raises
    ------
    ValueError
        If p4 is not of type numpy.ndarray or torch.Tensor.
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
