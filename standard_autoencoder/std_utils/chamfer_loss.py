import torch
import torch.nn as nn
import logging

class ChamferLoss(nn.Module):
    
    def __init__(self, loss_norm_choice):
        super(ChamferLoss, self).__init__()
        self.loss_norm_choice = loss_norm_choice

    def forward(self, p, q, jet_features=False):
        """
        The forward pass to compute the chamfer loss of the point-cloud like jets.

        Parameters
        ------
        p : `torch.Tensor`
            The **reconstructed** jets 4-momenta.
            Shape: `(batch_size, num_particles, 4)`
        q : `torch.Tensor`
            The **target** jets 4-momenta.
            Shape: `(batch_size, num_particles, 4) or (batch_size, num_particles, 4)`
        """
        self.device = p.device

        dist = pairwise_distance_sq(p, q, norm_choice=self.loss_norm_choice, device=self.device)

        min_dist_pq = torch.min(dist, dim=-1)
        min_dist_qp = torch.min(dist, dim=-2)  # Equivalent to permuting the last two axis

        # Adapted from Steven Tsan https://github.com/stsan9/AnomalyDetection4Jets/blob/emd/code/loss_util.py#L3
        chamfer_loss = torch.sum(min_dist_pq.values + min_dist_qp.values)

        if jet_features:
            jet_p = torch.sum(p, dim=-2).to(self.device)
            jet_q = torch.sum(q, dim=-2).to(self.device)
            jet_loss = normsq(jet_p - jet_q, norm_choice=self.loss_norm_choice).sum()
        else:
            jet_loss = 0

        return chamfer_loss + jet_loss

def pairwise_distance_sq(p, q, norm_choice='cartesian',
                         device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Compute the squared pairwise distance between jet 4-momenta p and q.

    Parameters
    ------
    p, q: `torch.Tensor`
        The 4-momenta of shape `(batch_size, num_particles, 4)`,
        where num_particles *could* be different for p and q.
    norm_choice : `str`
        The metric choice for distance.
        Option:
            - 'cartesian': (+, +, +, +)
            - 'minkowskian': (+, -, -, -)
            - 'polar': (E, pt, eta, phi) paired with metric (+, +, +, +)

    Output
    -------
    dist : `torch.Tensor`
        The matrix that represents distance between each particle in p and q.
        Shape : `(batch_size, num_particles, num_particles)`
    """
    if (p.shape[0] != q.shape[0]):
        raise ValueError(f"The batch size of p and q are not equal! Found: {p.shape[0]=}, {q.shape[0]=}.")
    if (p.shape[-1] != 4):
        raise ValueError(f"p should consist of 4-vectors. Found: {p.shape[-1]=}.")
    if (q.shape[-1] != 4):
        raise ValueError(f"q should consist of 4-vectors. Found: {q.shape[-1]=}.")

    batch_size = p.shape[0]
    num_row = p.shape[-2]
    num_col = q.shape[-2]
    vec_dim = 4  # 4-vectors

    p1 = p.repeat(1, 1, num_col).view(batch_size, -1, num_col, vec_dim).to(device)
    q1 = q.repeat(1, num_row, 1).view(batch_size, num_row, -1, vec_dim).to(device)
    
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

