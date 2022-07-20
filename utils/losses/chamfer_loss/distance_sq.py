from typing import Optional
import torch
from lgn.cg_lib.zonal_functions import p_cplx_to_rep, repdot
from utils.utils import get_p_polar


def convert_to_complex(
    real_ps: torch.Tensor, 
    eps: float = 0
) -> torch.Tensor:
    """Extend real jet 4-momenta from the dataset 
    to complex field (with imaginary dimension begin 0 tensors)

    :param real_ps: real 4-momenta of the jets.
        Shape: `(batch_size, num_particles, 4)`.
    :type: torch.Tensor
    :return: 4-momenta extended to the field of complex numbers, 
    with 0s in the imaginary components. 
    The shape is `(2, batch_size, num_particles, 4)`
    :rtype: torch.Tensor
    """
    return torch.stack((real_ps, torch.zeros_like(real_ps) + eps), dim=0)


def norm_sq(p: torch.Tensor) -> torch.Tensor:
    """Calculate the norm square of a real 4-vector, p

    :param p: real 4-momenta of the jets of shape `(batch_size, num_particles, 4)`.
    :type: torch.Tensor
    :return: The norm squared of p with metric diag(+, -, -, -).
    :rtype: torch.Tensor
    """
    psq = torch.pow(p, 2)
    return 2 * psq[..., 0] - psq.sum(dim=-1)


def norm_sq_p3(p3: torch.Tensor) -> torch.Tensor:
    """Calculate the norm square of a real 3-vector, p3

    :param p3: real 3-momenta of the jets
        Shape: `(batch_size, num_particles, 3)`
    :type: torch.Tensor
    :return: The norm squared of p3.
    :rtype: torch.Tensor
    """
    return torch.sum(torch.pow(p3, 2)[..., 1:], dim=-1)


def normsq_cplx(p4: torch.Tensor) -> torch.Tensor:
    """
    Compute the norm squared p4^2 for complex p4 and then the take the norm of the complex number.
    1. Lorentz norm is computed, resulting in a complex number.
    Shape: `(2, OTHER_DIMENSIONS)`
    2. Norm of the complex number is taken.
    Shape: `(OTHER_DIMENSIONS)`

    :param p4: 4-momenta with shape `(2, OTHER_DIMENSIONS, 4)`
        p4 = [p, q], where p is the real component, and q is the imaginary component.
    :type: torch.Tensor
    :return: The norm squared of p4. Recall p^2 = - m.
    :rtype: torch.Tensor
    """
    p4_real_sq = torch.pow(p4[0], 2)
    m_real = 2 * p4_real_sq[..., 0] - p4_real_sq.sum(dim=-1)
    pq = p4[0] * p4[1]
    m_im = 2 * (2 * pq[..., 0] - pq.sum(dim=-1))
    return torch.sqrt(torch.pow(m_real, 2) + torch.pow(m_im, 2))


def normsq_canonical(p4: torch.Tensor) -> torch.Tensor:
    """
    1. p is expressed in the basis of spherical harmonics, which is naturally defined in the field of complex numbers.
    2. The norm squared is computed.
    3. The result is converted to real.
    """
    p4 = p_cplx_to_rep(p4)
    p_sq = repdot(p4, p4)[(1, 1)]
    return torch.pow(p_sq[0], 2) + torch.pow(p_sq[1], 2)


def normsq_real(p4: torch.Tensor) -> torch.Tensor:
    """
    Compute the norm of complex p4 and then find the Lorentz norm squared.
    1. Norm of p4 is taken so that it only contains real components.
    Shape: `(OTHER_DIMENSIONS,4)`
    2. The Lorentz norm is computed using the Minkowskian metric (+, -, -, -).
    Shape: `(OTHER_DIMENSIONS)`

    :param p4: 4-momenta with shape `(2, OTHER_DIMENSIONS, 4)`
        p4 = [p, q], where p is the real component, and q is the imaginary component.
    :type: torch.Tensor
    :return: The norm squared of p4. Recall p^2 = - m.
    :rtype: torch.Tensor
    """
    p4_norm = torch.sqrt(torch.pow(p4[0], 2) + torch.pow(p4[1], 2))
    return 2 * p4_norm[..., 0] - p4_norm.sum(dim=-1)


def normsq_p3(
    p4: torch.Tensor, 
    im: bool = True
) -> torch.Tensor:
    """Calculate the norm square of the 3-momentum. This is useful when particle mass is neglible.
    1. Norm of p4 is taken so that it only contains real components.
    Shape: `(OTHER_DIMENSIONS, 4)`
    2. The norm of the 3-momenta is computed (in 3D Euclidean metric in )
    Shape: `(OTHER_DIMENSIONS)`
    """
    p4_real = p4[0]
    p_real = torch.sum(torch.pow(p4_real, 2)[..., 1:], dim=-1)
    if not im:
        return p_real

    p4_im = p4[1]
    p_im = torch.sum(torch.pow(p4_im, 2)[..., 1:], dim=-1)
    return p_real + p_im


def normsq_polar(
    p: torch.Tensor, 
    q: torch.Tensor, 
    im: bool = True
) -> torch.Tensor:
    """Calculate the norm square of the 3-momentum in polar coordinates.

    :param p: input tensor in Cartesian coordinates.
    :type p: torch.Tensor
    :param q: input tensor in Cartesian coordinates.
    :type q: torch.Tensor
    :param im: whether the imaginary component is taken into consideration, defaults to True
    :type im: bool, optional
    :return: Euclidean distance of polar features.
    :rtype: torch.Tensor
    """
    p_polar = get_p_polar(p[0], keep_p0=True)  # eta, phi, pt
    q_polar = get_p_polar(q[0], keep_p0=True)
    if im:
        p_polar = get_p_polar(p, keep_p0=True)
        q_polar = torch.stack((q_polar, q_polar), dim=0)
        q_polar[1, ..., -1] = 0
    return torch.sum(torch.pow(p_polar - q_polar, 2), dim=-1)  # ΔE^2 + ΔpT^2 + Δphi^2 + Δeta^2ß


def pairwise_distance_sq(
    p: torch.Tensor, 
    q: torch.Tensor, 
    norm_choice: str, 
    eps: float = 1e-16, 
    im: bool = True,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> torch.Tensor:
    """Compute the pairwise distance between jet 4-momenta p and q.

    :param p: Real 4-momenta of shape (batch_size, num_particles, 4)
    :type p: torch.Tensor
    :param q: Real 4-momenta of shape (batch_size, num_particles, 4)
    :type q: torch.Tensor
    :param norm_choice: the choice of norm to use.
        Options:
        - 'real': real, Cartesian coordinate.
        - 'canonical': canonical coordinate of the Lorentz group.
        - 'cplx' or 'complex': distance computed first and then taking the complex norm.
        - 'polar': polar coordinates.
    :type norm_choice: str
    :param eps: _description_, defaults to 1e-16
    :type eps: float, optional
    :param device: _description_, defaults to torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    :type device: torch.device, optional
    :raises ValueError: if the batch sizes of p and q are not equal
    :raises ValueError: if p does not consist of 3- or 4-vectors.
    :raises ValueError: if q does not consist of 3- or 4-vectors.
    :return: batched distance between each pair of the two collections of row vectors.
    :rtype: torch.Tensor
    """
    if (p.shape[1] != q.shape[1]):
        raise ValueError(f"The batch size of p and q are not equal! p.shape[1] is {p.shape[1]}, whereas q.shape[1] is {q.shape[1]}!")
    if (p.shape[-1] != 4):
        raise ValueError(f"p should consist of 4-vectors, but p.shape[-1] is {p.shape[-1]}.")
    if (q.shape[-1] != 4):
        raise ValueError(f"q should consist of 4-vectors, but q.shape[-1] is {q.shape[-1]}.")

    batch_size = p.shape[1]
    num_row = p.shape[-2]
    num_col = q.shape[-2]
    vec_dim = 4  # 4-vectors

    p1 = p.repeat(1, 1, 1, num_col).view(2, batch_size, -1, num_col, vec_dim).to(device)
    q1 = q.repeat(1, 1, num_row, 1).view(2, batch_size, num_row, -1, vec_dim).to(device)

    if norm_choice.lower() == 'real':
        dist = normsq_real(p1 - q1) + eps
    elif norm_choice.lower() == 'canonical':
        dist = normsq_canonical(p1 - q1)
    elif norm_choice.lower() in ['cplx', 'complex']:
        dist = normsq_cplx(p1 - q1)
    elif norm_choice.lower() == 'polar':
        dist = normsq_polar(p1, q1, im=im)
    else:
        dist = normsq_p3(p1 - q1, im=im)
    return dist


def pairwise_distance_sq_real(
    p: torch.Tensor, 
    q: torch.Tensor, 
    eps: float = 1e-16, 
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> torch.Tensor:
    """Compute the pairwise distance between jet 4-momenta p and q.

    :param p: Real 4-momenta of shape (batch_size, num_particles, 4)
    :type p: torch.Tensor
    :param q: Real 4-momenta of shape (batch_size, num_particles, 4)
    :type q: torch.Tensor
    :param eps: value of epsilon value to use, defaults to 1e-16
    :type eps: float, optional
    :param device: device to be used, defaults to gpu if available
    :type device: torch.device, optional
    :raises ValueError: if the batch sizes of p and q are not equal
    :raises ValueError: if p does not consist of 3- or 4-vectors.
    :raises ValueError: if q does not consist of 3- or 4-vectors.
    :return: batched distance between each pair of the two collections of row vectors.
        The distance is defined by
        - The Minkowskian distance with a metric diag(+, -, -, -) if both p anc q consist of 4-vectors.
        - Euclidean distance otherwise.
    :rtype: torch.Tensor
    """
    # check
    if (p.shape[0] != q.shape[0]):
        raise ValueError(f"The batch size of p and q are not equal! p.shape[0] is {p.shape[0]}, whereas q.shape[0] is {q.shape[0]}!")
    if (p.shape[-1] != 4) and (p.shape[-1] != 3):
        raise ValueError(f"p should consist of 3-vectors or 4-vectors, but p.shape[-1] is {p.shape[-1]}.")
    if (q.shape[-1] != 4) and (q.shape[-1] != 4):
        raise ValueError(f"q should consist of 3-vectors or 4-vectors, but q.shape[-1] is {q.shape[-1]}.")
    
    # preprocess to match feature dimension
    if (p.shape[-1] == 4) and (q.shape[-1] == 3):
        p = p[..., 1:]
    elif (p.shape[-1] == 3) and (q.shape[-1] == 4):
        q = q[..., 1:]

    # compute distance matrix
    batch_size = p.shape[0]
    num_row = p.shape[-2]
    num_col = q.shape[-2]
    vec_dim = p.shape[-1]

    p1 = p.repeat(1, 1, num_col).view(batch_size, -1, num_col, vec_dim).to(device)
    q1 = q.repeat(1, num_row, 1).view(batch_size, num_row, -1, vec_dim).to(device)

    if vec_dim == 4:
        dist = norm_sq(p1-q1)
    elif vec_dim == 3:
        dist = norm_sq_p3(p1-q1)

    return torch.sqrt(dist + eps)


def cdist(
    x1: torch.Tensor, 
    x2: torch.Tensor, 
    p: float = 2,
    eps: Optional[float] = 1e-16,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> torch.Tensor:
    """self implemented version of torch.cdist for 3- and 4-vectors
    that does not result in gradient clipping by adding eps.

    :param x1: input tensor of shape :math:`B \times P \times M`
    :type x1: torch.Tensor
    :param x2: input tensor of shape :math:`B \times P \times M`
    :type x2: torch.Tensor
    :param p: p value for the p-norm distance to calculate between each vector pair, defaults to 2
    :type p: float, optional
    :param eps: epsilon value, defaults to 1e-16
        If eps is None or 0, torch.cdist will be called directly.
    :type eps: float, optional
    :param device: the device to use, defaults gpu if available or cpu if not
    :type device: torch.device, optional
    :raises ValueError: if x1 or x2 are not 3- or 4-vectors.
    :return: batched the p-norm distance between each pair of the two collections of row vectors.
    :rtype: torch.Tensor
    """
    if (eps is None) or (eps == 0):
        return torch.cdist(x1, x2, p=2)
    
    x1 = x1.to(device)
    x2 = x2.to(device)
    
    if x1.shape[-1] == 4 and x2.shape[-1] == 4:
        diffs = - (torch.unsqueeze(x1[..., 1:], -2) -
                   torch.unsqueeze(x2[..., 1:], -3))
    elif x1.shape[-1] == 3 and x2.shape[-1] == 3:
        diffs = - (torch.unsqueeze(x1, -2) -
                   torch.unsqueeze(x2, -3))
    else:
        raise ValueError(
            f"x1 and x2 must be both 3- or 4-vectors. Found: {x1.shape[-1]=} and {x2.shape[-1]=}."
        )
    if (p % 2 == 0):
        return torch.sum(diffs ** p, dim=-1)
    else:
        return torch.sum(torch.abs(diffs + eps) ** p, dim=-1)
        
    # return torch.pow(dist_sq + eps, 1/p)
    
