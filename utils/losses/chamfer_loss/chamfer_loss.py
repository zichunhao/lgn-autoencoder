from typing import Optional
from torch import nn
import torch
from .distance_sq import cdist


class ChamferLoss(nn.Module):
    def __init__(self, device: Optional[torch.device] = None):
        super(ChamferLoss, self).__init__()
        self.device = (
            device
            if (device is not None)
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, jet_features: bool = False):
        dist = cdist(x, y, device=self.device)

        # Computer chamfer loss
        min_dist_xy = torch.min(dist, dim=-1).values
        # Equivalent to permuting the last two axis
        min_dist_yx = torch.min(dist, dim=-2).values
        chamfer_loss = torch.sum((min_dist_xy + min_dist_yx) / 2)

        if jet_features:
            jet_x = x.sum(dim=-2)
            jet_y = y.sum(dim=-2)
            jet_loss = nn.MSELoss()(jet_x, jet_y)
            return chamfer_loss + jet_loss
        else:
            return chamfer_loss
