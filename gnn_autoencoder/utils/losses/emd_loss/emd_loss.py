"""
Slightly adapted from Raghav Kansal's code (https://github.com/rkansal47/emd_loss)
"""
import torch
import torch.nn as nn
from .qpth.qp import QPFunction
from utils.utils import get_p_polar


# derived from https://github.com/icoz69/DeepEMD/blob/master/Models/models/emd_utils.py
def emd_inference_qpth(distance_matrix, weight1, weight2, device, form='QP', l2_strength=0.0001, add_energy_diff=True, eps=1e-12):
    """
    to use the QP solver QPTH to derive EMD (LP problem),
    one can transform the LP problem to QP,
    or omit the QP term by multiplying it with a small value,i.e. l2_strngth.
    :param distance_matrix: nbatch * element_number * element_number
    :param weight1: nbatch  * weight_number
    :param weight2: nbatch  * weight_number
    :return:
    emd distance: nbatch*1
    flow : nbatch * weight_number *weight_number
    """

    nbatch = distance_matrix.shape[0]
    nelement_distmatrix = distance_matrix.shape[1] * distance_matrix.shape[2]
    nelement_weight1 = weight1.shape[1]
    nelement_weight2 = weight2.shape[1]

    # reshape dist matrix too (nbatch, 1, n1 * n2)
    Q_1 = distance_matrix.view(-1, 1, nelement_distmatrix).double()
    # print(Q_1)

    if form == 'QP':  # converting to QP - after testing L2 reg performs marginally better than QP
        # version: QTQ
        Q = torch.bmm(Q_1.transpose(2, 1), Q_1).double() + eps * torch.eye(nelement_distmatrix).double().unsqueeze(0).repeat(nbatch, 1, 1)  # 0.00001 *
        p = torch.zeros(nbatch, nelement_distmatrix).double().to(device)
    elif form == 'L2':  # regularizing a trivial Q term with l2_strength
        # version: regularizer
        Q = (l2_strength * torch.eye(nelement_distmatrix).double()).unsqueeze(0).repeat(nbatch, 1, 1).to(device)
        p = distance_matrix.view(nbatch, nelement_distmatrix).double()
    else:
        raise ValueError('Unknown form')

    weight1 = weight1.to(device)
    weight2 = weight2.to(device)

    # h = [0 ... 0 w1 w2]
    h_1 = torch.zeros(nbatch, nelement_distmatrix).double().to(device)
    h_2 = torch.cat([weight1, weight2], 1).double().to(device)
    h = torch.cat((h_1, h_2), 1)

    G_1 = -torch.eye(nelement_distmatrix).double().unsqueeze(0).repeat(nbatch, 1, 1).to(device)
    G_2 = torch.zeros([nbatch, nelement_weight1 + nelement_weight2, nelement_distmatrix]).double().to(device)
    # sum_j(xij) = si
    for i in range(nelement_weight1):
        G_2[:, i, nelement_weight2 * i:nelement_weight2 * (i + 1)] = 1
    # sum_i(xij) = dj
    for j in range(nelement_weight2):
        G_2[:, nelement_weight1 + j, j::nelement_weight2] = 1

    # xij>=0, sum_j(xij) <= si,sum_i(xij) <= dj, sum_ij(x_ij) = min(sum(si), sum(dj))
    G = torch.cat((G_1, G_2), 1)
    A = torch.ones(nbatch, 1, nelement_distmatrix).double().to(device)
    b = torch.min(torch.sum(weight1, 1), torch.sum(weight2, 1)).unsqueeze(1).double()
    flow = QPFunction(verbose=-1)(Q, p, G, h, A, b)

    if add_energy_diff:
        energy_diff = torch.abs(torch.sum(weight1, dim=1) - torch.sum(weight2, dim=1))

    emd_score = torch.sum((Q_1).squeeze() * flow, 1)
    if add_energy_diff:
        emd_score += energy_diff

    return emd_score, flow.view(-1, nelement_weight1, nelement_weight2)


def emd_loss(jet_out, jet_target, eps=1e-12, form='L2', l2_strength=0.0001,
             return_flow=False, device=None):
    """
    batched Energy Mover's Distance between jet_out and jet_target

    Parameters
    ----------
    jet_out : `torch.Tensor`
        output momenta
        4-momenta of `(2, batch_size, num_particles, 4)` if complexified
    jet_target : `torch.Tensor`
        target momenta
        4-momenta of shape `(batch_size, num_particles, 4)` or `(2, batch_size, num_particles, 4)` if complexified
    return_flow : bool
        Optional, default: False
        Whether to the flow as well as the EMD score

    Return
    ------
    emd distance : torch.Tensor with shape (batch_size, 1)
    if return_flow:
        flow : torch.Tensor with shape (batch_size, num_particles, num_particles)
    """

    if (len(jet_out.shape) == 4) and (jet_out.shape[0] == 2) and (jet_out.shape[-1] == 4):
        jet_out = jet_out[0]  # real component only
        if len(jet_target.shape) == 4:  # complexified
            jet_target = jet_target[0]
    elif (len(jet_out.shape) == 3):  # real components only
        pass
    else:
        assert ValueError(f"Invalid shape of jet_out! Found: {jet_out.shape=}")

    if device is None:
        device = jet_out.device
    else:
        jet_out = jet_out.to(device)

    jet_target = jet_target.to(device)

    # Convert to polar coordinate (eta, phi, pt)
    jet_out = get_p_polar(jet_out, eps=eps)
    jet_target = get_p_polar(jet_target, eps=eps)

    diffs = -(jet_target[:, :, :2].unsqueeze(2) - jet_out[:, :, :2].unsqueeze(1)) + eps
    dists = torch.norm(diffs, dim=3)

    emd_score, flow = emd_inference_qpth(dists, jet_target[:, :, 2], jet_out[:, :, 2],
                                         device, form=form, l2_strength=l2_strength)

    return (emd_score.sum(), flow) if return_flow else emd_score.sum()


class EMDLoss(nn.Module):
    def __init__(self, eps=1e-16, form='L2', l2_strength=0.0001):
        """
        eps : float
            eps used in computation
        form : str
        l2_strength : float
        device : str or None
        """
        super(EMDLoss, self).__init__()
        self.eps = eps
        self.form = form
        self.l2_strength = l2_strength

    def forward(self, jet_out, jet_target, return_flow=False):
        return emd_loss(jet_out, jet_target, eps=self.eps, form=self.form,
                        l2_strength=self.l2_strength, return_flow=return_flow)
