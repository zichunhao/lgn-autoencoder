import torch
import torch.nn as nn

class ChamferLoss(nn.Module):
    def __init__(self, device):
        super(ChamferLoss, self).__init__()

        self.device = device

    def forward(self, x, y):
        dist = pairwise_distance(x, y, self.device)

        min_dist_xy = torch.min(dist, dim=-1)
        min_dist_yx = torch.min(dist, dim=-2)  # Equivalent to permute the last two axis

        # Adapted from Steven Tsan https://github.com/stsan9/AnomalyDetection4Jets/blob/emd/code/loss_util.py#L3
        loss = torch.sum(min_dist_xy.values + min_dist_yx.values)

        return loss

def pairwise_distance(x, y, device):
    assert (x.shape[0] == y.shape[0]), f"The batch size of x and y are not equal! x.shape[0] is {x.shape[0]}, whereas y.shape[0] is {y.shape[0]}!"
    assert (x.shape[-1] == y.shape[-1]), f"Feature dimension of x and y are not equal! x.shape[-1] is {x.shape[-1]}, whereas y.shape[-1] is {y.shape[-1]}!"

    batch_size = x.shape[0]
    num_row = x.shape[1]
    num_col = y.shape[1]
    vec_dim = x.shape[-1]

    x1 = x.repeat(1, 1, num_col).view(batch_size, -1, num_col, vec_dim).to(device)
    y1 = y.repeat(1, num_row, 1).view(batch_size, num_row, -1, vec_dim).to(device)

    dist = torch.norm(x1 - y1 + 1e-12, dim=-1)

    return dist
