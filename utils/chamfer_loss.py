import torch

import torch.nn as nn

class ChamferLoss(nn.Module):
    def __init__(self, device):
        super(ChamferLoss, self).__init__()

        self.device = device if (device is not None) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, p, q):
        """
        The forward pass to compute the chamfer loss of the point-cloud like jets.

        Inputs
        ------
        p : `torch.Tensor`
            The **generated** jets 4-momenta.
            Shape: (2, batch_size, num_particles, 4).
        q : `torch.Tensor`
            The **target** jets 4-momenta.
            Shape: (2, batch_size, num_particles, 4) or (batch_size, num_particles, 4)
        """

        if (len(p.shape) != 4) or (p.shape[0] != 2):
            raise ValueError(f'Invalid dimension: {p.shape}. The first argument should be complex generated momenta.')
        if len(q.shape) == 3:
            q = convert_to_complex(q)
        elif len(q.shape) == 4:
            pass
        else:
            raise ValueError(f'Invalid dimension: {q.shape}. The second argument should be the jet target momenta.')

        dist = self._pairwise_distance(p, q)

        min_dist_pq = torch.min(dist, dim=-1)
        min_dist_qp = torch.min(dist, dim=-2)  # Equivalent to permute the last two axis

        # Adapted from Steven Tsan https://github.com/stsan9/AnomalyDetection4Jets/blob/emd/code/loss_util.py#L3
        chamfer_loss = torch.sum(min_dist_pq.values + min_dist_qp.values)

        return chamfer_loss

    def _pairwise_distance(self, p, q):
        """
        Compute the pairwise between 4-momenta p and q

        Inputs
        ------
        p, q: `torch.Tensor`
            The 4-momenta of shape (2, batch_size, num_particles, 4),
            where num_particles *could* be different for p and q.

        Outputs
        -------
        dist : `torch.Tensor`
            The matrix that represents distance between each particle in p and q
            Shape : (batch_size, num_particles, num_particles)
        """
        if (p.shape[1] != q.shape[1]):
            raise ValueError(f"The batch size of p and q are not equal! p.shape[0] is {p.shape[1]}, whereas q.shape[0] is {q.shape[0]}!")
        if (p.shape[-1] != 4):
            raise ValueError( f"p should consist of 4-vectors, but p.shape[-1] is {p.shape[-1]}.")
        if (q.shape[-1] != 4):
            raise ValueError( f"q should consist of 4-vectors, but q.shape[-1] is {q.shape[-1]}.")

        batch_size = p.shape[1]
        num_row = p.shape[-2]
        num_col = q.shape[-2]
        vec_dim = 4  # 4-vectors

        p1 = p.repeat(1, 1, 1, num_col).view(2, batch_size, -1, num_col, vec_dim).to(self.device)
        q1 = q.repeat(1, 1, num_row, 1).view(2, batch_size, num_row, -1, vec_dim).to(self.device)
        dist = normsq(p1 - q1)
        return dist

def convert_to_complex(real_ps):
    """
    Convert real jet 4-momenta from the dataset to complex number (with imaginary dimension begin 0 tensors)

    Input
    -----
    real_ps : `torch.Tensor`
        Real 4-momenta of the jets.
        Shape: (batch_size, num_particles, 4)
    Output
    ------
    The 4-momenta extended to the field of complex numbers, with 0s in the imaginary components.
    The shape is (2, batch_size, num_particles, 4)
    """
    return torch.stack((real_ps, torch.zeros_like(real_ps)), dim=0)

def normsq(p4):
    """
    Compute the norm squared p4^2 for complex p4.

    Input
    -----
    p4 : `torch.Tensor`
        The 4-momenta with shape (2, ... , 4)
        p4 = [p, q], where p is the real component, and q is the imaginary component.
    Output
    -----
    m : `torch.Tensor`
        The norm square of p4
        Recall p^2 = - m
        Shape: (2, ...)
    """
    p4_real_sq = torch.pow(p4[0], 2)
    m_real = 2 * p4_real_sq[..., 0] - p4_real_sq.sum(dim=-1)
    pq = p4[0] * p4[1]
    m_im = 2 * (2 * pq[..., 0] - pq.sum(dim=-1))
    return torch.sqrt(torch.pow(m_real, 2) + torch.pow(m_im, 2) + 1e-12)

######################################## Unused ########################################
def reshape_generated_ps(generated_ps):
    """
    Reshape the generated 4-momenta to (batch_size, num_particles, 2, 4), the second last axis is the complex dimension.
    """
    return generated_ps.permute(1,2,0,3)  # (batch_size, num_particles, 2, 4)
