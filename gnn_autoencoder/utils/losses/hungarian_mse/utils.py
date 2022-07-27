import torch


def get_eps(p):
    dtype = p.dtype
    if dtype is [torch.float64, torch.double]:
        return 1e-16
    else:
        return 1e-12


def get_p_polar(p):
    """
    (E, px, py, pz) or (px, py, pz)-> (eta, phi, pt)
    """
    if p.shape[-1] == 4:
        idx_px, idx_py, idx_pz = 1, 2, 3
    elif p.shape[-1] == 3:
        idx_px, idx_py, idx_pz = 0, 1, 2
    else:
        raise ValueError(f'Wrong last dimension of p. Should be 3 or 4 but found: {p.shape[-1]}.')

    px = p[..., idx_px]
    py = p[..., idx_py]
    pz = p[..., idx_pz]
    eps = get_eps(p)

    pt = torch.sqrt(px ** 2 + py ** 2 + eps)
    try:
        eta = torch.asinh(pz / (pt + eps))
    except AttributeError:
        eta = arcsinh(pz / (pt + eps))
    phi = torch.atan2(py + eps, px + eps)

    return torch.stack((pt, eta, phi), dim=-1)


def arcsinh(z):
    return torch.log(z + torch.sqrt(1 + torch.pow(z, 2)))


def get_polar_rel(p, jet, input_cartesian=True):
    if input_cartesian:
        p = get_p_polar(p)  # Convert to polar first
        jet = get_p_polar(jet)

    pt = p[..., 0]
    eta = p[..., 1]
    phi = p[..., 2]

    num_particles = p.shape[-2]
    pt /= jet[..., 0].unsqueeze(dim=-1).repeat(1, num_particles)
    eta -= jet[..., 1].unsqueeze(dim=-1).repeat(1, num_particles)
    phi -= jet[..., 2].unsqueeze(dim=-1).repeat(1, num_particles)

    return torch.stack((pt, eta, phi), dim=-1)


def get_p_cartesian(p, return_p0=False):
    if p.shape[-1] == 4:
        idx_pt, idx_eta, idx_phi = 1, 2, 3
    elif p.shape[-1] == 3:
        idx_pt, idx_eta, idx_phi = 0, 1, 2
    else:
        raise ValueError(f'Wrong last dimension of p. Should be 3 or 4 but found: {p.shape[-1]}.')

    pt = p[..., idx_pt]
    eta = p[..., idx_eta]
    phi = p[..., idx_phi]

    px = pt * torch.cos(phi)
    py = pt * torch.cos(phi)
    pz = pt * torch.sinh(eta)

    if not return_p0:
        return torch.stack((px, py, pz), dim=-1)
    else:
        E = pt * torch.cosh(eta)
        return torch.stack((E, px, py, pz), dim=-1)


def check_p_dim(p):
    """Check whether p is a 3- or 4-vector.

    Raise
    -----
    ValueError if p is not a 3- or 4-vector (i.e. p.shape[-1] is not 3 or 4).
    """
    if p.shape[-1] not in [3, 4]:
        raise ValueError(f'Wrong last dimension of p. Should be 3 or 4 but found: {p.shape[-1]}.')
