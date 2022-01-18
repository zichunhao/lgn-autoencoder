import torch
import torch.nn as nn
from .utils import get_p_polar, get_p_cartesian, get_polar_rel, check_p_dim
from scipy import optimize


class HungarianMSELoss(nn.Module):
    """Permutation invariant MSE Loss for jets using scipy.optimiz (Hungarian algorithm)."""

    def __init__(self):
        super(HungarianMSELoss, self).__init__()

    def forward(self, recons, target, abs_coord=True, polar_coord=False):
        self.abs_coord = abs_coord
        self.polar_coord = polar_coord
        self.device = recons.device
        return jet_mse_loss(recons, target, abs_coord=abs_coord, polar_coord=polar_coord)


def jet_mse_loss(recons, target, abs_coord=True, polar_coord=False):
    """Get the permutation invariant MSE Loss for jets (Hungarian algorithm).

    Parameters
    ----------
    recons : torch.Tensor
        The reconstructed particle momenta in absolute Cartesian coordinates.
        Shape: (batch_size, num_particles, 3) or (batch_size, num_particles, 4)
    target : torch.Tensor
        The target particle momenta in absolute Cartesian coordinates.
        Shape: (batch_size, num_particles, 3) or (batch_size, num_particles, 4)
    abs_coord : bool, optional
        Whether to use absolute coordinate.
        If False, relative coordinates (with respect to the target) will be used.
        Default: True
    polar_coord: bool, optional
        Whether to calculate the MSE in polar coordinates (pt, eta, phi).
        Default: False

    Returns
    -------
    The particle-wise MSE loss after finding the match bewteen target and reconstructed/generated jet.
    """
    recons, target = preprocess(recons, target, abs_coord=abs_coord, polar_coord=polar_coord)
    recons._requires_grad = True
    device = recons.device

    cost = torch.cdist(recons, target).cpu().detach().numpy()
    matching = [optimize.linear_sum_assignment(cost[i])[1] for i in range(len(cost))]

    recons_shuffle = torch.zeros(recons.shape).to(device).to(recons.dtype)
    for i in range(len(matching)):
        recons_shuffle[i] = recons[i, matching[i]]

    mse = nn.MSELoss()
    loss = mse(recons_shuffle, target)
    return loss


def preprocess(recons, target, abs_coord=True, polar_coord=False):
    """Preprocess target and recons by converting jets into desired coordinates.
    Options:
        - absolute Cartesian coordinates
        - absolute polar coordinates
        - relative Cartesian coordinates
        - relative polar coordinates
    """
    check_p_dim(target)
    check_p_dim(recons)

    device = recons.device
    target = target.to(device)

    if abs_coord:
        # Absolute polar coordinates
        if polar_coord:
            target = get_p_polar(target)
            recons = get_p_polar(recons)
            return recons, target

        # Absolute Cartesian coordinates
        return recons, target

    else:  # Relative coordinate
        target_jet = target.sum(dim=-2)
        target = get_polar_rel(target, target_jet, input_cartesian=True)
        recons = get_polar_rel(recons, target_jet, input_cartesian=True)

        # Relative polar coordinates
        if polar_coord:
            return recons, target

        # Convert to relative Cartesian coordiantes
        target = get_p_cartesian(target)
        recons = get_p_cartesian(recons)
        # Relative Cartesian coordinates
        return recons, target
