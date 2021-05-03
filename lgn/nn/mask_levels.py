import torch
import torch.nn as nn


class MaskLevel(nn.Module):

    """
    Mask level for implementing hard and soft cutoffs. With the current
    architecutre, we have all-to-all communication.

    This mask takes relative position vectors p_{ij} = p_i - p_j
    and implements either a hard cutoff, a soft cutoff, or both.
    The soft cutoffs can also be made learnable.

    Parameters
    ----------
    num_channels : int
        Number of channels to mask out.
    hard_cut_rad : float
        Hard cutoff radius beyond which two nodes will never communicate.
    soft_cut_rad : float
        Soft cutoff radius used in cutoff function.
    soft_cut_width : float
        Soft cutoff width if 'sigmoid' form of cutoff cuntion is used.
    cutoff_type : list of str
        Optional, default: None, in which case it will be set to ['learn']
        What types of cutoffs to use. Choices are 'hard', 'soft', 'learn',
        'learn_rad', and/or 'learn_width'.
    gaussian_mask : bool
        Optional, default: False
        Whether to mask using gaussians instead of sigmoids.
    eps : float
        Optional, default: 1e-3
        Numerical minimum to use in case learnable cutoff parameters are driven towards zero.
    device : torch.device
        Optional, default: torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        The device to which the module is initialized.
    dtype : torch.dtype
        Optional, default: torch.float64
        The data type to which the module is initialized.
    """
    def __init__(self, num_channels, hard_cut_rad, soft_cut_rad, soft_cut_width, cutoff_type,
                 gaussian_mask=False, eps=1e-3,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 dtype=torch.float64):
        if device is None:
            device = torch.device('cpu')
        super(MaskLevel, self).__init__()

        self.gaussian_mask = gaussian_mask
        self.num_channels = num_channels
        
        
        self.zero = torch.tensor(0, device=device, dtype=dtype)
        self.eps = torch.tensor(eps, device=device, dtype=dtype)  # Numerical minimum for learnable cutoff

        # Initialize hard/soft cutoff to None as default.
        self.hard_cut_rad = None
        self.soft_cut_rad = None
        self.soft_cut_width = None

        if 'hard' in cutoff_type:
            self.hard_cut_rad = hard_cut_rad

        if ('soft' in cutoff_type) or ('learn' in cutoff_type) or ('learn_rad' in cutoff_type) or ('learn_width' in cutoff_type):

            self.soft_cut_rad = soft_cut_rad*torch.ones(num_channels, device=device, dtype=dtype).view((1, 1, 1, -1))
            self.soft_cut_width = soft_cut_width*torch.ones(num_channels, device=device, dtype=dtype).view((1, 1, 1, -1))

            if ('learn' in cutoff_type) or ('learn_rad' in cutoff_type):
                self.soft_cut_rad = nn.Parameter(self.soft_cut_rad)

            if ('learn' in cutoff_type) or ('learn_width' in cutoff_type):
                self.soft_cut_width = nn.Parameter(self.soft_cut_width)


    """
    Forward pass to mask the input.

    Parameters
    ----------
    edge_net : torch.Tensor or GVec
        Edge scalars or edge GVec to apply mask to.
    edge_mask : torch.Tensor
        Mask to account for padded batches.
    norms : torch.Tensor
        Pairwise distance matrices.
    sq_norms : torch.Tensor
        Pairwise distance norm squared matrices.
    Returns
    -------
    edge_net : torch.Tensor
        Input edge_net with mask applied.
    """
    def forward(self, edge_net, edge_mask, norms, sq_norms):

        # Use hard cut
        if self.hard_cut_rad is not None:
            edge_mask = (edge_mask * (norms < self.hard_cut_rad))

        edge_mask = edge_mask.to(self.dtype).unsqueeze(-1).to(self.dtype)

        # Use soft cut
        if self.soft_cut_rad is not None:
            cut_width = torch.max(self.eps, self.soft_cut_width.abs())
            cut_rad = torch.max(self.eps, self.soft_cut_rad.abs())

            if self.gaussian_mask:  # Use Gaussian mask
                edge_mask = edge_mask * torch.exp(-(sq_norms.unsqueeze(-1)/cut_rad.pow(2)))
            else:  # Use sigmoid mask
                edge_mask = edge_mask * torch.sigmoid((cut_rad - norms.unsqueeze(-1))/cut_width)

        edge_mask = edge_mask.unsqueeze(-1)
        edge_net = edge_net * edge_mask

        return edge_net
