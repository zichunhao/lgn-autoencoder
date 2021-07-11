import torch
import torch.nn as nn
from utils.norm_sq import convert_to_complex, pairwise_distance, norm_sq

class ChamferLoss(nn.Module):
    """
    Parameters
    ----------
    loss_norm_choice : `str`
        The choice to compute the norm squared of the complex 4-vector.
        Optional, default: `'p3'`
        Options:
        - `'p3'`
            1. Norm of p4 is taken so that it only contains real components.
            Shape: `(OTHER_DIMENSIONS,4)`
            2. The norm of the 3-momenta is computed (in Cartesian metric)
            Shape: `(OTHER_DIMENSIONS)`
        - `'real'`
            1. Norm of p4 is taken so that it only contains real components.
            Shape: `(batch_size, num_particles, 4)`
            2. The Lorentz norm is computed using the Minkowskian metric (+, -, -, -).
            Note that the result can be negative even though it is called the 'norm squared.'
            Shape: `(batch_size, num_particles)`
        - `'cplx'` or '`complex'`
            1. Lorentz norm is computed first using the Minkowskian metric (+, -, -, -).
            Shape: `(2, batch_size, num_particles)`
            2. Norm of the complex number is taken. Note that the result will be non-negative.
            Shape: `(batch_size, num_particles)`
        - `'canonical'`
            1. p is expressed in the basis of spherical harmonics, which is naturally defined in the field of complex numbers.
            2. The norm squared is computed.
            3. The result is converted to real.
    """

    def __init__(self, loss_norm_choice='p3', device=None):
        super(ChamferLoss, self).__init__()
        if loss_norm_choice.lower() not in ['real', 'cplx', 'complex', 'canonical', 'p3', 'polar']:
            raise ValueError("loss_norm_choice can only be one of ['real', 'cplx', 'canonical', 'p3', 'polar']. "
                             f"Found: {loss_norm_choice}")

        self.loss_norm_choice = loss_norm_choice
        self.device = device if (device is not None) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, p, q, jet_features=False):
        """
        The forward pass to compute the chamfer loss of the point-cloud like jets.

        Parameters
        ------
        p : `torch.Tensor`
            The **generated** jets 4-momenta.
            Shape: `(2, batch_size, num_particles, 4)`
        q : `torch.Tensor`
            The **target** jets 4-momenta.
            Shape: `(2, batch_size, num_particles, 4) or (batch_size, num_particles, 4)`
        """

        if (len(p.shape) != 4) or (p.shape[0] != 2):
            raise ValueError(f'Invalid dimension: {p.shape}. The first argument should be complex generated momenta.')
        if len(q.shape) == 3:
            q = convert_to_complex(q)
        elif len(q.shape) == 4:
            pass
        else:
            raise ValueError(f'Invalid dimension: {q.shape}. The second argument should be the jet target momenta.')

        dist = torch.pow(pairwise_distance(p, q, norm_choice=self.loss_norm_choice, device=self.device), 2)

        min_dist_pq = torch.min(dist, dim=-1)
        min_dist_qp = torch.min(dist, dim=-2)  # Equivalent to permuting the last two axis

        # Adapted from Steven Tsan https://github.com/stsan9/AnomalyDetection4Jets/blob/emd/code/loss_util.py#L3
        chamfer_loss = torch.sum(min_dist_pq.values + min_dist_qp.values)

        if jet_features:
            jet_p = torch.sum(p[0], dim=-2)
            jet_q = torch.sum(q[0], dim=-2)
            jet_loss = norm_sq(jet_p - jet_q).sum()
        else:
            jet_loss = 0

        return chamfer_loss + jet_loss


############################################## Unused ##############################################
def reshape_generated_ps(generated_ps):
    """
    Reshape the generated 4-momenta to `(batch_size, num_particles, 2, 4)`, the second last axis is the complex dimension.
    """
    return generated_ps.permute(1, 2, 0, 3)  # (batch_size, num_particles, 2, 4)
