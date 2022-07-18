import torch


def get_eps(x: torch.Tensor) -> float:
    """Return the epsilon value for the given x based on the device.

    :param p: the tensor to get the epsilon value for
    :type p: torch.Tensor
    :return: 1e-16 if x is in double precision (i.e. torch.float64 or torch.double) and 1e-12 otherwise
    :rtype: float
    """
    return 1e-16 if x.dtype in (torch.float64, torch.double) else 1e-12


def get_p_polar(p: torch.Tensor) -> torch.Tensor:
    """(E, px, py, pz) or (px, py, pz)-> (eta, phi, pt)

    :param p: particle features (E, px, py, pz) or (px, py, pz)
    :type p: torch.Tensor
    :raises ValueError: if p is not a 3- or 4-vector (i.e. p.shape[-1] is not 3 or 4)
    :return: particle features in polar coordinates (eta, phi, pt)
    :rtype: torch.Tensor
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


def arcsinh(z: torch.Tensor) -> torch.Tensor:
    """self implemented arcsinh function 
    ..math::
        arcsinh(z) = \log(z + \sqrt(z^2 + 1))
    if torch is not up to date

    :param z: input of arcsinh
    :type z: torch.Tensor
    :return: arcsinh(z)
    :rtype: torch.Tensor
    """
    return torch.log(z + torch.sqrt(1 + torch.pow(z, 2)))


def get_polar_rel(
    p: torch.Tensor, 
    jet: torch.Tensor, 
    input_cartesian: bool = True
) -> torch.Tensor:
    """
    Get polar coordinates relative to a jet.

    :param p: the particle features (E, px, py, pz) or (px, py, pz) or (pt, eta, phi)
    :type p: torch.Tensor
    :param jet: the jet features (E, Px, Py, Pz) or (E, Pt, Eta, Phi)
    :type jet: torch.Tensor
    :param input_cartesian: _description_, defaults to True
    :type input_cartesian: bool, optional
    :return: the relative polar coordinates (pt, eta, phi)
    :rtype: torch.Tensor
    """
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


def get_p_cartesian(
    p: torch.Tensor, 
    return_p0: bool = False
) -> torch.Tensor:
    """
    (E, pt, eta, phi) or (pt, eta, phi) -> (px, py, pz)

    :param p: the particle features (E, pt, eta, phi) or (pt, eta, phi)
    :type p: torch.Tensor
    :param return_p0: whether to return p0 (i.e. E), defaults to False
    :type return_p0: bool, optional
    :raises ValueError: if p is not a 3- or 4-vector (i.e. p.shape[-1] is not 3 or 4)
    :return: the particle features in cartesian coordinates (px, py, pz)
    :rtype: torch.Tensor
    """
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


def check_p_dim(p: torch.Tensor) -> None:
    """_summary_
    Check whether p is a 3 - or 4-vector.
    :raises ValueError: if p is not a 3 - or 4-vector(i.e. p.shape[-1] is not 3 or 4).
    """
    if p.shape[-1] not in [3, 4]:
        raise ValueError(f'Wrong last dimension of p. Should be 3 or 4 but found: {p.shape[-1]}.')
