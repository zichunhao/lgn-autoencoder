import torch


def apply_lorentz_transformation(
    mat: torch.Tensor,
    p4: torch.Tensor
) -> torch.Tensor:
    '''
    Applies a Lorentz transformation to a 4-vector.

    :param mat: The Lorentz transformation matrix. Shape: (4, 4).
    :param p4: The 4-vector to transform.
        Supported formats:
        - batched jets: (batch_size, num_particles, 4)
        - a single jet: (num_particles, 4)
        - a single 4-vector: (4,)

    :return: The transformed 4-vector.
    '''
    # type checks
    if not isinstance(p4, torch.Tensor):
        raise TypeError(f"p4 must be a torch.Tensor. Found: {type(p4)}")

    if not isinstance(mat, torch.Tensor):
        raise TypeError(f"mat must be a torch.Tensor. Found: {type(mat)}")

    # shape checks
    if p4.shape[-1] != 4:
        if p4.shape[-1] == 3:
            import logging
            logging.warning(
                '3-vector(s) is(are) passed to apply_lorentz_transformation.'
                'Expanding to 4-vector(s) with the assumption that particles are massless.'
            )
            E = torch.norm(p4, dim=-1, keepdim=True)
            p4 = torch.cat((E, p4), dim=-1)
        else:
            raise ValueError(f"p4 must be 4-vectors. Found: {p4.shape[-1]=}")
    else:  # correct dimensions
        pass

    # apply transformation
    if len(p4.shape) == 3:  # (batch_size, num_particles, 4)
        return torch.einsum('ij,bnj->bni', mat, p4)
    if len(p4.shape) == 2:  # (num_particles, 4)
        return torch.einsum('ij,nj->ni', mat, p4)
    if len(p4.shape) == 1:  # a single 4-vector
        return mat @ p4
    # invalid dimensions
    raise ValueError(
        f"p4 must have the dimensions (batch_size, num_particles, 4), (num_particles, 4), or (4, ). Found: {p4.shape=}"
    )
