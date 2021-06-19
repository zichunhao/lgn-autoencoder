import torch
from lgn.cg_lib.zonal_functions import p_cplx_to_rep, repdot


def convert_to_complex(real_ps):
    """
    Convert real jet 4-momenta from the dataset to complex number (with imaginary dimension begin 0 tensors)

    Input
    -----
    real_ps : `torch.Tensor`
        Real 4-momenta of the jets.
        Shape: `(batch_size, num_particles, 4)`

    Output
    ------
    The 4-momenta extended to the field of complex numbers, with 0s in the imaginary components.
    The shape is `(2, batch_size, num_particles, 4)`
    """
    return torch.stack((real_ps, torch.zeros_like(real_ps)), dim=0)


def normsq_cplx(p4):
    """
    Compute the norm squared p4^2 for complex p4 and then the take the norm of the complex number.
    1. Loretnz norm is computed, resulting in a complex number.
    Shape: `(2, OTHER_DIMENSIONS)`
    2. Norm of the complex number is taken.
    Shape: `(OTHER_DIMENSIONS)`

    Input
    -----
    p4 : `torch.Tensor`
        The 4-momenta with shape (2, OTHER_DIMENSIONS , 4)
        p4 is represented by `[p, q]`, where p is the real component, and q is the imaginary component.

    Output
    -----
    m : `torch.Tensor`
        The norm square of p4
        Recall p^2 = - m
        Shape: `(OTHER_DIMENSIONS)`
    """
    p4_real_sq = torch.pow(p4[0], 2)
    m_real = 2 * p4_real_sq[..., 0] - p4_real_sq.sum(dim=-1)
    pq = p4[0] * p4[1]
    m_im = 2 * (2 * pq[..., 0] - pq.sum(dim=-1))
    return torch.sqrt(torch.pow(m_real, 2) + torch.pow(m_im, 2))


def normsq_canonical(p4):
    """
    1. p is expressed in the basis of spherical harmonics, which is naturally defined in the field of complex numbers.
    2. The norm squared is computed.
    3. The result is converted to real.
    """
    p4 = p_cplx_to_rep(p4)
    p_sq = repdot(p4, p4)[(1, 1)]
    return torch.pow(p_sq[0], 2) + torch.pow(p_sq[1], 2)


def normsq_real(p4):
    """
    Compute the norm of complex p4 and then find the Lorentz norm squared.
    1. Norm of p4 is taken so that it only contains real components.
    Shape: `(OTHER_DIMENSIONS,4)`
    2. The Lorentz norm is computed using the Minkowskian metric (+, -, -, -).
    Shape: `(OTHER_DIMENSIONS)`

    Input
    -----
    p4 : `torch.Tensor`
        The 4-momenta with shape `(2, OTHER_DIMENSIONS, 4)`
        p4 = [p, q], where p is the real component, and q is the imaginary component.

    Output
    -----
    m : `torch.Tensor`
        The norm square of p4
        Recall p^2 = - m
        Shape: `(OTHER_DIMENSIONS)`
    """
    p4_norm = torch.sqrt(torch.pow(p4[0], 2) + torch.pow(p4[1], 2))
    return 2 * p4_norm[..., 0] - p4_norm.sum(dim=-1)


def normsq_p3(p4):
    """
    Calculate the norm square of the 3-momentum. This is useful when particle mass is neglible.
    1. Norm of p4 is taken so that it only contains real components.
    Shape: `(OTHER_DIMENSIONS, 4)`
    2. The norm of the 3-momenta is computed (in 3D Euclidean metric in )
    Shape: `(OTHER_DIMENSIONS)`
    """
    p4_real = p4[0]
    p4_im = p4[1]
    p_real = torch.sum(torch.pow(p4_real, 2)[..., 1:], dim=-1)
    p_im = torch.sum(torch.pow(p4_im, 2)[..., 1:], dim=-1)
    return p_real + p_im


def pairwise_distance(p, q, loss_norm_choice, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Compute the pairwise between jet 4-momenta p and q

    Parameters
    ------
    p, q: `torch.Tensor`
        The 4-momenta of shape `(2, batch_size, num_particles, 4)`,
        where num_particles *could* be different for p and q.

    Output
    -------
    dist : `torch.Tensor`
        The matrix that represents distance between each particle in p and q.
        Shape : `(batch_size, num_particles, num_particles)`
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

    if loss_norm_choice.lower() == 'real':
        dist = torch.sqrt(normsq_real(p1 - q1))
    elif loss_norm_choice.lower() == 'canonical':
        dist = torch.sqrt(normsq_canonical(p1 - q1))
    elif loss_norm_choice.lower() in ['cplx', 'complex']:
        dist = torch.sqrt(normsq_cplx(p1 - q1))
    else:
        dist = torch.sqrt(normsq_p3(p1 - q1))
    return dist