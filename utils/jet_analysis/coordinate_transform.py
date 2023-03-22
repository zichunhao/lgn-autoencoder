import numpy as np
import torch
from typing import Union, Iterable


def p4_polar_from_p4_cartesian_massless(
    p4: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Transform 4-momenta from Cartesian coordinates to polar coordinates for massless particle features.

    Args:
        p4 (np.ndarray): array of 4-momenta in Cartesian coordinates,
        of shape ``[..., 4]``. The last axis should be in order
        :math:`(E/c, p_x, p_y, p_z)`.

    Returns:
        np.ndarray: array of 4-momenta in polar coordinates, arranged in order
        :math:`(E/c, p_\mathrm{T}, \eta, \phi)`, where :math:`\eta` is the pseudorapidity.
    """

    eps = __get_default_eps(p4)  # default epsilon for the dtype

    # (E/c, px, py, pz) -> (E/c, pT, eta, phi)
    p0, px, py, pz = __unbind(p4, axis=-1)
    pt = __sqrt(px**2 + py**2)
    eta = __arcsinh(pz / (pt + eps))
    phi = __arctan2(py, px)

    return __stack([p0, pt, eta, phi], axis=-1)


def p4_cartesian_from_p4_polar_massless(
    p4: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Transform 4-momenta from polar coordinates to Cartesian coordinates for massless particle features.

    Args:
        p4 (np.ndarray): array of 4-momenta in polar coordinates,
        of shape ``[..., 4]``. The last axis should be in order
        :math:`(E/c, p_\mathrm{T}, \eta, \phi)`, where :math:`\eta` is the pseudorapidity.

    Returns:
        np.ndarray: array of 4-momenta in polar coordinates, arranged in order
        :math:`(E/c, p_x, p_y, p_z)`.
    """

    # (E/c, pT, eta, phi) -> (E/c, px, py, pz)
    p0, pt, eta, phi = __unbind(p4, axis=-1)
    px = pt * __cos(phi)
    py = pt * __sin(phi)
    pz = pt * __sinh(eta)

    return __stack([p0, px, py, pz], axis=-1)


def p4_cartesian_from_polarrel(p_polarrel, jet_polar):
    pt_rel, eta_rel, phi_rel = p_polarrel.unbind(dim=-1)
    _, Pt, Eta, Phi = jet_polar.unsqueeze(dim=-2).unbind(dim=-1)
    pt = pt_rel * Pt
    eta = eta_rel + Eta
    phi = phi_rel + Phi
    phi = (phi + np.pi) % (2 * np.pi) - np.pi  # [-pi, pi]

    p0 = pt * torch.cosh(eta)
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)

    return torch.stack((p0, px, py, pz), dim=-1)


def polarrel_from_p4_cartesian_massless(
    p4: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Get particle features in relative polar coordinates from 4-momenta in Cartesian coordinates.

    Args:
        p4 (np.ndarray): array of 4-momenta in Cartesian coordinates,
        of shape ``[..., 4]``. The last axis should be in order
        :math:`(E/c, p_x, p_y, p_z)`.

    Returns:
        np.ndarray: array of features in relative polar coordinates, arranged in order
        :math:`(p_\mathrm{T}^\mathrm{rel}, \eta^\mathrm{rel}, \phi^\mathrm{rel})`.
    """

    eps = __get_default_eps(p4)  # default epsilon for the dtype

    # particle (pT, eta, phi)
    p4_polar = p4_polar_from_p4_cartesian_massless(p4)
    _, pt, eta, phi = __unbind(p4_polar, axis=-1)

    # jet (PT, Eta, Phi)
    # expand dimension to (..., 1, 4) to match p4 shape
    if isinstance(p4, torch.Tensor):
        jet_cartesian = torch.sum(p4, dim=-2)
        jet_cartesian = jet_cartesian.unsqueeze(-2)
    else:
        jet_cartesian = np.sum(p4, axis=-2)
        jet_cartesian = jet_cartesian[..., np.newaxis, :]
    jet_polar = p4_polar_from_p4_cartesian_massless(jet_cartesian)
    _, Pt, Eta, Phi = __unbind(jet_polar, axis=-1)

    # get relative features
    pt_rel = pt / (Pt + eps)
    eta_rel = eta - Eta
    phi_rel = phi - Phi
    phi_rel = (phi_rel + np.pi) % (2 * np.pi) - np.pi  # modify to [-pi, pi]

    assert not eta.isnan().any(), "eta is nan"
    assert not Eta.isnan().any(), "Eta is nan"

    return __stack([pt_rel, eta_rel, phi_rel], axis=-1)


def polarrel_from_p4_polar_massless(
    p4: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Get particle features in relative polar coordinates from 4-momenta in polar coordinates.

    Args:
        p4 (np.ndarray): array of 4-momenta in polar coordinates,
        of shape ``[..., 4]``. The last axis should be in order
        :math:`(E/c, p_\mathrm{T}, \eta, \phi)`, where :math:`\eta` is the pseudorapidity.

    Returns:
        np.ndarray: array of features in relative polar coordinates, arranged in order
        :math:`(p_\mathrm{T}^\mathrm{rel}, \eta^\mathrm{rel}, \phi^\mathrm{rel})`.
    """

    eps = __get_default_eps(p4)  # default epsilon for the dtype

    # particle (pT, eta, phi)
    p4_polar = p4
    _, pt, eta, phi = __unbind(p4_polar, axis=-1)

    # jet (PT, Eta, Phi)
    p4 = p4_cartesian_from_p4_polar_massless(p4_polar)
    jet_cartesian = __sum(p4, axis=-2)
    # expand dimension to (..., 1, 4) to match p4 shape
    if isinstance(jet_cartesian, torch.Tensor):
        jet_cartesian = jet_cartesian.unsqueeze(-2)
    else:
        jet_cartesian = jet_cartesian[..., np.newaxis, :]
    jet_polar = p4_polar_from_p4_cartesian_massless(jet_cartesian)
    _, Pt, Eta, Phi = __unbind(jet_polar, axis=-1)

    # get relative features
    pt_rel = pt / (Pt + eps)
    eta_rel = eta - Eta
    phi_rel = phi - Phi
    phi_rel = (phi_rel + np.pi) % (2 * np.pi) - np.pi  # modify to [-pi, pi]

    return __stack([pt_rel, eta_rel, phi_rel], axis=-1)


def p4_polar_from_p4_cartesian(
    p4: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Transform 4-momenta from Cartesian coordinates to polar coordinates.

    Args:
        p4 (np.ndarray): array of 4-momenta in Cartesian coordinates,
        of shape ``[..., 4]``. The last axis should be in order
        :math:`(E/c, p_x, p_y, p_z)`.

    Returns:
        np.ndarray: array of 4-momenta in polar coordinates, arranged in order
        :math:`(E/c, p_\mathrm{T}, y, \phi)`, where :math:`y` is the rapidity.
    """

    eps = __get_default_eps(p4)  # default epsilon for the dtype

    # (E/c, p_x, p_y, p_z) -> (E/c, p_T, y, phi)
    p0, px, py, pz = __unbind(p4, axis=-1)
    pt = __sqrt(px**2 + py**2)
    y = 0.5 * __log((p0 + pz + eps) / (p0 - pz + eps))
    phi = __arctan2(py, px)

    return __stack([p0, pt, y, phi], axis=-1)


def p4_cartesian_from_p4_polar(
    p4: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Transform 4-momenta from polar coordinates to Cartesian coordinates.

    Args:
        p4 (np.ndarray): array of 4-momenta in Cartesian coordinates,
        of shape ``[..., 4]``. The last axis should be in order
        :math:`(E/c, p_\mathrm{T}, y, \phi)`, where :math:`y` is the rapidity.

    Returns:
        np.ndarray: array of 4-momenta in polar coordinates, arranged in order
        :math:`(E/c, p_\mathrm{T}, y, \phi)`, where :math:`y` is the rapidity.
        :math:`(E/c, p_x, p_y, p_z)`.
    """

    eps = __get_default_eps(p4)  # default epsilon for the dtype

    # (E/c, pT, y, phi) -> (E/c, px, py, pz)
    p0, pt, y, phi = __unbind(p4, axis=-1)
    px = pt * __cos(phi)
    py = pt * __sin(phi)
    # get pz
    mt = p0 / (__cosh(y) + eps)  # get transverse mass
    pz = mt * __sinh(y)
    return __stack([p0, px, py, pz], axis=-1)


def __unbind(
    x: Union[np.ndarray, torch.Tensor], axis: int
) -> Union[np.ndarray, torch.Tensor]:
    """Unbind an np.ndarray or torch.Tensor along a given axis."""
    if isinstance(x, torch.Tensor):
        return torch.unbind(x, dim=axis)
    elif isinstance(x, np.ndarray):
        return np.rollaxis(x, axis=axis)
    else:
        raise TypeError(
            f"x must be either a numpy array or a torch tensor, not {type(x)}"
        )


def __stack(
    x: Iterable[Union[np.ndarray, torch.Tensor]], axis: int
) -> Union[np.ndarray, torch.Tensor]:
    """Stack an iterable of np.ndarray or torch.Tensor along a given axis."""
    if not isinstance(x, Iterable):
        raise TypeError("x must be an iterable.")

    if isinstance(x[0], torch.Tensor):
        return torch.stack(x, dim=axis)
    elif isinstance(x[0], np.ndarray):
        return np.stack(x, axis=axis)
    else:
        raise TypeError(
            f"x must be either a numpy array or a torch tensor, not {type(x)}"
        )


def __cos(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Cosine function that works with np.ndarray and torch.Tensor."""
    if isinstance(x, torch.Tensor):
        return torch.cos(x)
    elif isinstance(x, np.ndarray):
        return np.cos(x)
    else:
        raise TypeError(
            f"x must be either a numpy array or a torch tensor, not {type(x)}"
        )


def __sin(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Sine function that works with np.ndarray and torch.Tensor."""
    if isinstance(x, torch.Tensor):
        return torch.sin(x)
    elif isinstance(x, np.ndarray):
        return np.sin(x)
    else:
        raise TypeError(
            f"x must be either a numpy array or a torch tensor, not {type(x)}"
        )


def __sinh(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Hyperbolic sine function that works with np.ndarray and torch.Tensor."""
    if isinstance(x, torch.Tensor):
        return torch.sinh(x)
    elif isinstance(x, np.ndarray):
        return np.sinh(x)
    else:
        raise TypeError(
            f"x must be either a numpy array or a torch tensor, not {type(x)}"
        )


def __arcsinh(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Inverse hyperbolic sine function that works with np.ndarray and torch.Tensor."""
    if isinstance(x, torch.Tensor):
        return torch.asinh(x)
    elif isinstance(x, np.ndarray):
        return np.arcsinh(x)
    else:
        raise TypeError(
            f"x must be either a numpy array or a torch tensor, not {type(x)}"
        )


def __cosh(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Hyperbolic cosine function that works with np.ndarray and torch.Tensor."""
    if isinstance(x, torch.Tensor):
        return torch.cosh(x)
    elif isinstance(x, np.ndarray):
        return np.cosh(x)
    else:
        raise TypeError(
            f"x must be either a numpy array or a torch tensor, not {type(x)}"
        )


def __log(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Logarithm function that works with np.ndarray and torch.Tensor."""
    if isinstance(x, torch.Tensor):
        return torch.log(x)
    elif isinstance(x, np.ndarray):
        return np.log(x)
    else:
        raise TypeError(
            f"x must be either a numpy array or a torch tensor, not {type(x)}"
        )


def __arctan2(
    y: Union[np.ndarray, torch.Tensor], x: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """Arctangent function that works with np.ndarray and torch.Tensor."""
    if isinstance(y, torch.Tensor):
        return torch.atan2(y, x)
    elif isinstance(y, np.ndarray):
        return np.arctan2(y, x)
    else:
        raise TypeError(
            f"y must be either a numpy array or a torch tensor, not {type(y)}"
        )


def __sqrt(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Square root function that works with np.ndarray and torch.Tensor."""
    if isinstance(x, torch.Tensor):
        return torch.sqrt(x)
    elif isinstance(x, np.ndarray):
        return np.sqrt(x)
    else:
        raise TypeError(
            f"x must be either a numpy array or a torch tensor, not {type(x)}"
        )


def __sum(
    x: Union[np.ndarray, torch.Tensor], axis: int, keepdims: bool = False
) -> Union[np.ndarray, torch.Tensor]:
    """Sum function that works with np.ndarray and torch.Tensor."""
    if isinstance(x, torch.Tensor):
        return x.sum(axis, keepdim=keepdims)
    elif isinstance(x, np.ndarray):
        return np.sum(x, axis=axis, keepdims=keepdims)
    else:
        raise TypeError(
            f"x must be either a numpy array or a torch tensor, not {type(x)}"
        )


def __get_default_eps(x: Union[np.ndarray, torch.Tensor]) -> float:
    if isinstance(x, torch.Tensor):
        return torch.finfo(x.dtype).eps
    elif isinstance(x, np.ndarray):
        return np.finfo(x.dtype).eps
    else:
        raise TypeError(
            f"x must be either a numpy array or a torch tensor, not {type(x)}"
        )


__ALL__ = [
    p4_polar_from_p4_cartesian_massless,
    p4_cartesian_from_p4_polar_massless,
    p4_cartesian_from_polarrel,
    polarrel_from_p4_cartesian_massless,
    polarrel_from_p4_polar_massless,
    p4_polar_from_p4_cartesian,
    p4_cartesian_from_p4_polar,
]
