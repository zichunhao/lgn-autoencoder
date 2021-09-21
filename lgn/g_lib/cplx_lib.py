import torch

#########  Weight mixing  ###########


def mix_zweight_zvec(weight, part, zdim=0):
    """
    Apply the linear matrix in `GWeight` and a part of a `GVec`.

    Parameters
    ----------
    weight : `torch.Tensor`
        A tensor of mixing weights to apply to `part`.
    part : `torch.Tensor`
        Part of `GVec` to multiply by scalars.

    """
    weight_r, weight_i = weight.unbind(zdim)
    part_r, part_i = part.unbind(zdim)

    return torch.stack([weight_r@part_r - weight_i@part_i,
                        weight_i@part_r + weight_r@part_i], dim=zdim)


def mix_zweight_zscalar(weight, part, zdim=0):
    """
    Apply the linear matrix in `GWeight` and a part of a `GScalar`.

    Parameters
    ----------
    scalar : `torch.Tensor`
        A tensor of mixing weights to apply to `part`.
    part : `torch.Tensor`
        Part of `GScalar` to multiply by scalars.

    """
    # Must permute first two dimensions
    weight_r, weight_i = weight.transpose(0, 1).unbind(zdim)
    part_r, part_i = part.unbind(zdim)

    # # Since the dimension to be mixed in part is the right-most,
    return torch.stack([part_r@weight_r - part_i@weight_i,
                        part_r@weight_i + part_i@weight_r], dim=zdim)


#########  Multiply  ###########

def mul_zscalar_zirrep(scalar, part, rdim=-1, zdim=0):
    """
    Multiply the part of a `GScalar` and a part of a `GVec`.

    Parameters
    ----------
    scalar : `torch.Tensor`
        A tensor of scalars to apply to `part`.
    part : `torch.Tensor`
        Part of `GVec` to multiply by scalars.

    """
    scalar_r, scalar_i = scalar.unsqueeze(rdim).unbind(zdim)
    part_r, part_i = part.unbind(zdim)

    return torch.stack([part_r * scalar_r - part_i * scalar_i,
                        part_r * scalar_i + part_i * scalar_r], dim=zdim)


def mul_zscalar_zscalar(scalar1, scalar2, zdim=0):
    """
    Complex multiply the part of a `GScalar` and a part of a
    different `GScalar`.

    Parameters
    ----------
    scalar1 : `torch.Tensor`
        First tensor of scalars to multiply.
    scalar2 : `torch.Tensor`
        Second tensor of scalars to multiply.
    zdim : int
        Dimension for which complex multiplication is defined.


    """
    scalar1_r, scalar1_i = scalar1.unbind(zdim)
    scalar2_r, scalar2_i = scalar2.unbind(zdim)

    return torch.stack([scalar1_r*scalar2_r - scalar1_i*scalar2_i,
                        scalar1_r*scalar2_i + scalar1_i*scalar2_r], dim=zdim)
