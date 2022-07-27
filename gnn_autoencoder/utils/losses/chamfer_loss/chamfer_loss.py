import torch
import torch.nn as nn
from .distance_sq import pairwise_distance_sq, normsq


class ChamferLoss(nn.Module):

    def __init__(self, loss_norm_choice):
        super(ChamferLoss, self).__init__()
        self.loss_norm_choice = loss_norm_choice

    def forward(self, p, q, jet_features_weight=1):
        """
        The forward pass to compute the chamfer loss of the point-cloud like jets.

        Parameters
        ----------
        p : `torch.Tensor`
            The **reconstructed** jets 4-momenta.
            Shape: `(batch_size, num_particles, 4)`
        q : `torch.Tensor`
            The **target** jets 4-momenta.
            Shape: `(batch_size, num_particles, 4) or (batch_size, num_particles, 4)`
        """
        self.device = p.device

        dist = pairwise_distance_sq(
            p, q, norm_choice=self.loss_norm_choice, 
            device=self.device
        )

        min_dist_pq = torch.min(dist, dim=-1)
        # Equivalent to permuting the last two axis
        min_dist_qp = torch.min(dist, dim=-2)

        # Adapted from Steven Tsan https://github.com/stsan9/AnomalyDetection4Jets/blob/emd/code/loss_util.py#L3
        chamfer_loss = torch.sum(min_dist_pq.values + min_dist_qp.values)

        if jet_features_weight != 0:
            jet_p = torch.sum(p, dim=-2).to(self.device)
            jet_q = torch.sum(q, dim=-2).to(self.device)
            jet_loss = normsq(
                jet_p - jet_q, norm_choice=self.loss_norm_choice).sum()
            chamfer_loss += jet_features_weight * jet_loss
        return jet_loss
