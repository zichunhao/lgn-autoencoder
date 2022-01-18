import torch

def pairwise_distance_sq(p, q, norm_choice='cartesian',
                         device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Compute the squared pairwise distance between jet 4-momenta p and q.

    Args
    ----
    p, q: `torch.Tensor`
        The 3- or 4-momenta of shape `(batch_size, num_particles, 3)` or `(batch_size, num_particles, 4)`,
        where num_particles *could* be different for p and q.
    norm_choice : str
        The metric choice for distance.
        Option:
            - 'cartesian': (+, +, +, +) (mandatory for 3-momenta)
            - 'minkowskian': (+, -, -, -)
            - 'polar': (E, pt, eta, phi) paired with metric (+, +, +, +)

    Return
    ------
    dist : `torch.Tensor`
        The matrix that represents distance between each particle in p and q.
        Shape : `(batch_size, num_particles, num_particles)`
    """
    if (p.shape[0] != q.shape[0]):
        raise ValueError(
            f"The batch size of p and q are not equal! Found: {p.shape[0]=}, {q.shape[0]=}.")
    if (p.shape[-1] not in [3, 4]):
        raise ValueError(
            f"p should consist of 3- or 4-vectors. Found: {p.shape[-1]=}.")
    if (q.shape[-1] not in [3, 4]):
        raise ValueError(
            f"q should consist of 3- or 4-vectors. Found: {q.shape[-1]=}.")
    if (q.shape[-1] != p.shape[-1]):
        raise ValueError(
            f"Dimension of q ({q.shape[-1]}) does not match with dimension of p ({p.shape[-1]}).")
    if (q.shape[-1] == 3):
        norm_choice = 'cartesian'

    batch_size = p.shape[0]
    num_row = p.shape[-2]
    num_col = q.shape[-2]
    vec_dim = p.shape[-1]

    p1 = p.repeat(1, 1, num_col).view(
        batch_size, -1, num_col, vec_dim).to(device)
    q1 = q.repeat(1, num_row, 1).view(
        batch_size, num_row, -1, vec_dim).to(device)

    return normsq(p1-q1, norm_choice=norm_choice)


def normsq(p, norm_choice='cartesian'):
    if norm_choice.lower() == 'minkowskian':
        return normsq_minkowskian(p)
    if norm_choice.lower() == 'polar':
        return normsq_polar(p)
    else:
        return normsq_cartesian(p)


def normsq_minkowskian(p):
    psq = torch.pow(p, 2)
    return 2 * psq[..., 0] - psq.sum(dim=-1)


def normsq_cartesian(p):
    return torch.sum(torch.pow(p, 2), dim=-1)


def normsq_polar(p):
    psq = torch.pow(p, 2)
    return 2 * psq[..., 0] - psq.sum(dim=-1)
