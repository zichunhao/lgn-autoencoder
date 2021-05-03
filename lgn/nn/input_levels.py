import torch
import torch.nn as nn

from lgn.nn.generic_levels import BasicMLP
from lgn.nn.position_levels import RadPolyTrig
from lgn.nn.mask_levels import MaskLevel
from lgn.g_lib import GTau, GVec


class InputLinear(nn.Module):
    """
    Input linear layer.

    This module applies a simple linear mixing matrix to a one-hot of node
    embeddings based upon the number of node types.

    Parameters
    ----------
    channels_in : int
        Number of input features before mixing.
    channels_out : int
        Number of output features after mixing.
    bias : bool
        Optional, default: True
        Whether to include bias terms in the linear mixing level.
    device : torch.device
        Optional, default: torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        The device to which the module is initialized.
    dtype : torch.dtype
        Optional, default: torch.float64
        The data type to which the module is initialized.
    """
    def __init__(self, channels_in, channels_out, bias=True,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 dtype=torch.float64):
        super(InputLinear, self).__init__()

        self.channels_in = channels_in
        self.channels_out = channels_out
        self.bias = bias

        self.lin = nn.Linear(channels_in, 2*channels_out, bias=bias)
        self.lin.to(device=device, dtype=dtype)

        self.zero = torch.tensor(0, dtype=dtype, device=device)

    """
    Forward pass for InputLinear.

    Parameters
    ----------
    input_scalars : torch.Tensor
        Input scalar features.
    node_mask : torch.Tensor
        Mask used to account for padded nodes for unequal batch sizes.

    Return
    -------
    A GVec that represents the processed node features
    to be used as input to Clebsch-Gordan layers as part of LGN.
    """
    def forward(self, input_scalars, node_mask, *ignore):

        node_mask = node_mask.unsqueeze(-1)
        # masking
        out = torch.where(node_mask, self.lin(input_scalars), self.zero)
        # Put zdim at axis 0
        out = out.view(input_scalars.shape[0:2] + (self.channels_out, 1, 2)).permute(4, 0, 1, 2, 3)  # The 2 is the complex dimension
        return GVec({(0, 0): out})

    @property
    def tau(self):
        return GTau({(0, 0): self.channels_out})


class InputMPNN(nn.Module):
    """
    Module to create node feature vectors at the input level.

    Parameters
    ----------
    channels_in : int
        Number of input features before mixing.
    channels_out : int
        Number of output features after mixing.
    num_layers : int
        Number of message passing layers.
    soft_cut_rad : float
        Optional, default: None
        Radius of the soft cutoff used in the radial position functions.
    soft_cut_width : float
        Optional, default: None
        Width of the soft cutoff used in the 'sigmoid' radial position functions.
    hard_cut_rad : float
        Optional, default: None
        Radius of the hard cutoff used in the radial position functions.
    cutoff_type : list of str
        Optional, default: None, in which case it will be set to ['learn']
        What types of cutoffs to use. Choices are 'hard', 'soft', 'learn',
        'learn_rad', and/or 'learn_width'.
    channels_mlp : int
        Optional, default: -1, in which case it will be set to max(channels_in, channels_out)
        Number of channels in the hidden layer.
    nun_hidden : int
        Optional, default: 1
        Number of hidden layers.
    layer_width : int
        Optional: default: 256
        Number of perceptrons per hidden layer.
    activation : str
        Optional, default: 'leakyrelu'
        The type of activation function to use.
        Options are ‘leakyrelu’(for nn.LeakyReLU()), ‘relu’(for nn.ReLU()),
        ‘elu’(for nn.ELU()), ‘sigmoid’(for nn.Sigmoid()),‘logsigmoid’(for nn.LogSigmoid()),
        and ‘atan’(for torch.atan(input))
    basis_set : iterable
        Optional, default: None, in which case it will be set to (3, 3)
        The number of bases to use.
    bias : bool
        Optional, default: True
        Whether to include bias terms in the linear mixing level.
    device : torch.device
        Optional, default: None, in which case we will use
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        The device to which the module is initialized.
    dtype : torch.dtype
        Optional, default: torch.float64
        The data type to which the module is initialized.
    """
    def __init__(self, channels_in, channels_out, num_layers=1,
                 soft_cut_rad=None, soft_cut_width=None, hard_cut_rad=None, cutoff_type=None,
                 channels_mlp=-1, num_hidden=1, layer_width=256,
                 activation='leakyrelu', basis_set=None,
                 device=None, dtype=torch.float64):

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if cutoff_type is None:
            cutoff_type = ['learn']
        if basis_set is None:
            basis_set = (3, 3)
        if device is None:
            device = torch.device('cpu')

        super(InputMPNN, self).__init__()

        self.soft_cut_rad = soft_cut_rad
        self.soft_cut_width = soft_cut_width
        self.hard_cut_rad = hard_cut_rad

        # List of channels at each level. The factor of two accounts for
        # the fact that the passed messages are concatenated with the input states.
        channels_lvls = [channels_in] + [channels_mlp]*(num_layers-1) + [2*channels_out]

        self.channels_in = channels_in
        self.channels_out = channels_out
        if channels_mlp < 0:
            self.channels_mlp = max(channels_in, channels_out)
        else:
            self.channels_mlp = channels_mlp

        # Set up
        self.mlps = nn.ModuleList()
        self.masks = nn.ModuleList()
        self.rad_filts = nn.ModuleList()

        for chan_in, chan_out in zip(channels_lvls[:-1], channels_lvls[1:]):
            # radial filter
            rad_filt = RadPolyTrig(0, basis_set, chan_in, mix='real', device=device, dtype=dtype)
            self.rad_filts.append(rad_filt)

            # mask
            mask = MaskLevel(1, hard_cut_rad, soft_cut_rad, soft_cut_width, ['soft', 'hard'], device=device, dtype=dtype)
            self.masks.append(mask)

            # MLP
            mlp = BasicMLP(2*chan_in, chan_out, activation=activation, num_hidden=num_hidden, layer_width=layer_width, device=device, dtype=dtype)
            self.mlps.append(mlp)

        
        

    def forward(self, features, node_mask, edge_features, edge_mask, norms, sq_norms):
        """
        Forward pass for InputMPNN layer.

        Parameters
        ----------
        features : torch.Tensor
            Input node features.
        node_mask : torch.Tensor
            Mask used to account for padded nodes for unequal batch sizes.
        edge_features : torch.Tensor
            Unused. Included only for pedagogical purposes.
        edge_mask : torch.Tensor
            Mask used to account for padded edges for unequal batch sizes.
        norms : torch.Tensor
            Matrix of relative distances between pairs of nodes.
        sq_norms : torch.Tensor
            Matrix of norm squared of the relative distances between pairs of nodes
            in momentum space.

        Returns
        -------
        A GVec object that stores the processed node features
        to be used as input to Clebsch-Gordan layers.
        Note that zdim is at axis 0.
        """

        node_mask = node_mask.unsqueeze(-1)  # reshape

        s = features.shape  # Get the shape of the input to reshape at the end

        # Loop over MPNN levels. There is no edge network here.
        # Instead, there is just masked radial functions which takes
        # the role of the adjacency matrix.
        for mlp, rad_filt, mask in zip(self.mlps, self.rad_filts, self.masks):
            # Construct the learnable radial functions
            rad = rad_filt(norms, edge_mask)
            rad = rad[0][..., 0].unsqueeze(-1)

            # Mask the position function if desired
            edge = mask(rad, edge_mask, norms, sq_norms)
            # Convert to a form that MatMul expects
            edge = edge.squeeze(-1)

            # Now pass messages using matrix multiplication with the edge features
            # Einsum b: batch, n: node, c: channel, x: to be summed over
            features_mp = torch.einsum('bnxc,bxc->bnc', edge, features)

            # Concatenate the passed messages with the original features
            features_mp = torch.cat([features_mp, features], dim=-1)

            # Apply a masked MLP
            features = mlp(features_mp, mask=node_mask)

        # The output are the MLP features reshaped into a set of complex numbers.
        out = features.view((2,)+s[0:2] + (self.channels_out, 1))

        return GVec({(0,0): out})

    @property
    def tau(self):
        return GTau({(0,0): self.channels_out})
