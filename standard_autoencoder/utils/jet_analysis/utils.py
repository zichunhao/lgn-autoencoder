import torch
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_BINS = 81  # Number of bins for all histograms
PLOT_FONT_SIZE = 12


def get_magnitude(p, gpu=True):
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
        if gpu:
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
        mask = (p > cutoff)
        if cutoff > 0:
            px = px[mask]
            py = py[mask]
            pz = pz[mask]
    elif isinstance(jets, torch.Tensor):
        jets = torch.clone(jets).reshape(-1, 4)
        px = torch.clone(jets[:, 1]).detach().cpu().numpy()
        py = torch.clone(jets[:, 2]).detach().cpu().numpy()
        pz = torch.clone(jets[:, 3]).detach().cpu().numpy()
        p = get_magnitude(jets)  # |p| of 3-momenta
        if cutoff > 0:
            px = px[mask]
            py = py[mask]
            pz = pz[mask]
    else:
        raise ValueError(f"The input must be numpy.ndarray or torch.Tensor. Found: {type(jets)}.")

    return px, py, pz


def get_p_polar(p4, cutoff=1e-6, eps=1e-12, gpu=True):
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
        if gpu:
            p4 = p4.to(device=DEVICE)

        p_polar = get_p_polar_tensor(p4)
        pt = p_polar[..., 0].detach().cpu().numpy()
        eta = p_polar[..., 1].detach().cpu().numpy()
        phi = p_polar[..., 2].detach().cpu().numpy()

        if cutoff > 0:
            p = get_magnitude(p4).detach().cpu().numpy()
            mask = (p > cutoff)
            pt = pt[mask]
            eta = eta[mask]
            phi = phi[mask]

    return pt, eta, phi

def get_p4_cartesian_from_polar(p: torch.Tensor) -> torch.Tensor:
    '''
    (pt, eta, phi) -> (E, px, py, pz) or
    (E, pt, eta, phi) -> (E, px, py, pz).
    '''
    if p.shape[-1] == 4:
        E_idx, pt_idx, eta_idx, phi_idx = 0, 1, 2, 3
        E = (p[..., E_idx]).unsqueeze(-1)
    elif p.shape[-1] == 3:
        pt_idx, eta_idx, phi_idx = 0, 1, 2
        E = (p[..., pt_idx] * torch.cosh(p[..., eta_idx])).unsqueeze(-1)
    else:
        raise ValueError(f'Invalid shape of feature dimension: {p.shape[-1]}.')
    
    px = (p[..., pt_idx] * torch.cos(p[..., phi_idx])).unsqueeze(-1)
    py = (p[..., pt_idx] * torch.sin(p[..., phi_idx])).unsqueeze(-1)
    pz = (p[..., pt_idx] * torch.sinh(p[..., eta_idx])).unsqueeze(-1)
    
    return torch.cat((E, px, py, pz), dim=-1)
    

def get_p4(p3):
    '''Convert particle p3 to p4'''
    p0 = torch.norm(p3, dim=-1, keepdim=True)
    return torch.cat((p0, p3), dim=-1)


def get_p_polar_tensor(p, eps=1e-16):
    """(E, px, py, pz) -> (pt, eta, phi)"""
    if p.shape[-1] == 4:
        px = p[..., 1]
        py = p[..., 2]
        pz = p[..., 3]
    elif p.shape[-1] == 3:
        px = p[..., 0]
        py = p[..., 1]
        pz = p[..., 2]
    else:
        raise ValueError(f'Invalid error. p.shape[-1] should be either 3 or 4. Found: {p.shape[-1]}.')

    pt = torch.sqrt(px ** 2 + py ** 2)
    try:
        eta = torch.asinh(pz / (pt + eps))
    except AttributeError:
        eta = arcsinh(pz / pt)
    phi = torch.atan2(py + eps, px)

    return torch.stack((pt, eta, phi), dim=-1)


def arcsinh(z):
    return torch.log(z + torch.sqrt(1 + torch.pow(z, 2)))


def get_jet_feature_cartesian(p4, gpu=True):
    """
    Get jet (m, pt, eta, phi) from the jet data.

    Parameters
    ----------
    p4 : `numpy.ndarray` or `torch.Tensor`
        The jet data, with shape (num_particles, 4), which means all jets are merged together.
    gpu : bool, optional
        Whether to use gpu whenever possible.
        Default: True

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
        if gpu:
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


def get_jet_feature_polar(p4, gpu=True, eps=1e-16):
    """
    Get jet (m, pt, eta, phi) from the jet data.

    Parameters
    ----------
    p4 : `numpy.ndarray` or `torch.Tensor`
        The jet data, with shape (num_particles, 4), which means all jets are merged together.
    gpu : bool, optional
        Whether to use gpu whenever possible.
        Default: True

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
        if gpu:
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


def find_fwhm(err, bins):
    """Full width at half maximum of a distribution."""
    hist, _ = np.histogram(err, bins=bins)
    max_idx = np.argmax(hist)
    peak = bins[max_idx]

    half_max = hist[max_idx] / 2
    half_max_idx = (np.abs(hist - half_max)).argmin()
    half_peak = bins[half_max_idx]

    return 2 * abs(peak - half_peak)


def get_stats(res, bins):
    return {'mean': np.mean(res), 'FWHM': find_fwhm(res, bins)}


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


# Ranges
def get_recons_err_ranges(args):
    """Get bins for reconstruction error plots"""
    if not args.custom_particle_recons_ranges:  # set to default ranges
        particle_recons_ranges = None
    else:
        particle_recons_ranges = _get_particle_recons_ranges(args)

    if not args.custom_jet_recons_ranges:  # set to default ranges
        jet_recons_ranges = None
    else:
        jet_recons_ranges = _get_jet_recons_ranges(args)

    return particle_recons_ranges, jet_recons_ranges


def _get_particle_recons_ranges(args):

    rel_err_cartesian = tuple([
        np.linspace(
            args.particle_rel_err_min_cartesian[i],
            args.particle_rel_err_max_cartesian[i],
            NUM_BINS
        )
        for i in range(3)
    ])

    rel_err_polar = tuple([
        np.linspace(
            args.particle_rel_err_min_polar[i],
            args.particle_rel_err_max_polar[i],
            NUM_BINS
        )
        for i in range(3)
    ])

    padded_recons_cartesian = tuple([
        np.linspace(
            args.particle_padded_recons_min_cartesian[i],
            args.particle_padded_recons_max_cartesian[i],
            NUM_BINS
        )
        for i in range(3)
    ])

    padded_recons_polar = tuple([
        np.linspace(
            args.particle_padded_recons_min_polar[i],
            args.particle_padded_recons_max_polar[i],
            NUM_BINS
        )
        for i in range(3)
    ])

    return ((rel_err_cartesian, padded_recons_cartesian),
            (rel_err_polar, padded_recons_polar))


def _get_jet_recons_ranges(args):
    rel_err_cartesian = tuple([
        np.linspace(
            args.jet_rel_err_min_cartesian[i],
            args.jet_rel_err_max_cartesian[i],
            NUM_BINS
        )
        for i in range(4)
    ])
    rel_err_polar = tuple([
        np.linspace(
            args.jet_rel_err_min_polar[i],
            args.jet_rel_err_max_polar[i],
            NUM_BINS
        )
        for i in range(4)
    ])

    return (rel_err_cartesian, rel_err_polar)
