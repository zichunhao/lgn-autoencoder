import torch
import torch.nn as nn

from lgn.cg_lib import CGProduct
from lgn.nn.generic_levels import get_activation_fn
from lgn.nn import CatMixReps


class LGNNodeLevel(nn.Module):

    """
    The message passing step in LGNCG layers.

    Parameters
    ----------
    tau_in : `GTau`
        The multiplicity of the input features.
    tau_pos : `list` of `GTau` (or compatible)
        The multiplicity of the relative position vectors.
    maxdim : `int`
        The maximum weight of irreps to be accounted.
    num_channels : `list` of `int`
        The number of channels in the LGNNodeLevel.
    level_gain : `list` of `floats`
        The gain at each level. (args.level_gain = [1.])
    weight_init :  `str`
        The type of weight initialization. The choices are 'randn' and 'rand'.
    device : `torch.device`
        Optional, default: None, in which case we will use
            torch.device('cuda' if torch.cuda.is_available() else 'cpu').
        The device to which the module is initialized.
    dtype : `torch.dtype`
        Optional, default: torch.float64
        The data type to which the module is initialized.
    cg_dict : `CGDict`
        Optional, default: None
        The CG dictionary as a reference for taking CG decompositions.
    """

    def __init__(self, tau_in, tau_pos, maxdim, num_channels, level_gain, weight_init,
                 device=None, dtype=torch.float64, cg_dict=None):

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        super(LGNNodeLevel, self).__init__()
        self.maxdim = maxdim
        self.num_channels = num_channels

        self.tau_in = tau_in
        self.tau_pos = tau_pos

        # Operations linear in input reps
        # Self-interactions
        self.cg_power = CGProduct(tau_in, tau_in, maxdim=self.maxdim,
                                  device=device, dtype=dtype, cg_dict=cg_dict)
        tau_sq = self.cg_power.tau_out
        # Mutual interactions
        self.cg_aggregate = CGProduct(tau_in, tau_pos, maxdim=self.maxdim,
                                      aggregate=True, device=device, dtype=dtype, cg_dict=cg_dict)
        tau_ag = self.cg_aggregate.tau_out
        # Message aggregation
        self.cat_mix = CatMixReps([tau_ag, tau_in, tau_sq], num_channels, maxdim=self.maxdim,
                                  weight_init=weight_init, gain=level_gain, device=device, dtype=dtype)
        self.tau_out = self.cat_mix.taus_out

    """
    The forward pass for message aggregation.

    Parameters
    ----------
    node_feature :  GVec
        Node features.
    edge_feature: GVec
        Edge features.
    torch.Tensor with data type torch.byte
        Batch mask for node representations. Shape is (N_batch, N_node).

    Return
    ----------
    node_feature_out: GVec
        The updated node features.
    """

    def forward(self, node_feature, edge_feature, mask):
        # Mutual interactions
        reps_ag = self.cg_aggregate(node_feature, edge_feature)
        # Self-interactions
        reps_sq = self.cg_power(node_feature, node_feature)
        # Message aggregation
        node_feature_out = self.cat_mix([reps_ag, node_feature, reps_sq])
        return node_feature_out


class CGMLP(nn.Module):
    """
    The multilayer perceptron layers for LGNCG, for scalars only.

    Parameters
    ----------
    tau : `GTau`
        The multiplicity of the input (scalar) features.
    num_hidden : `int`
        The number of hidden layers.
    layer_width_mul : `int`
        The ratio between the layer width and num_scalars.
    activation : `str`
        Optional, default: 'sigmoid'
        The type of activation function to use.
        Options are ‘leakyrelu’(for nn.LeakyReLU()), ‘relu’(for nn.ReLU()),
        ‘elu’(for nn.ELU()), ‘sigmoid’(for nn.Sigmoid()),‘logsigmoid’(for nn.LogSigmoid()),
        and ‘atan’(for torch.atan(input))
    device : `torch.device`
        Optional, default: None, in which case we will use
            torch.device('cuda' if torch.cuda.is_available() else 'cpu').
        The device to which the module is initialized.
    dtype : `torch.dtype`
        Optional, default: torch.float64
        The data type to which the module is initialized.
    """

    def __init__(self, tau, num_hidden=3, layer_width_mul=2, activation='sigmoid',
                 device=None, dtype=torch.float64):

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        super(CGMLP, self).__init__()

        self.tau = tau
        num_scalars = 2 * tau[(0, 0)]  # Account for the complex dimension
        self.num_scalars = num_scalars
        layer_width = layer_width_mul * num_scalars

        # Linear layers
        self.linear = nn.ModuleList()
        self.linear.append(nn.Linear(num_scalars, layer_width))
        for layer in range(num_hidden - 1):
            self.linear.append(nn.Linear(layer_width, layer_width))
        if num_hidden > 0:
            self.linear.append(nn.Linear(layer_width, num_scalars))
        else:
            self.linear.append(nn.Linear(num_scalars, num_scalars))

        # Activation functions
        activation_fn = get_activation_fn(activation)
        self.activations = nn.ModuleList()
        for i in range(num_hidden):
            self.activations.append(activation_fn)

        # For calculating mask in the forward function
        self.zero = torch.tensor(0, device=device, dtype=dtype)

        self.to(device=device, dtype=dtype)

    def forward(self, node_feature_in, mask=None):
        """
        The forward function for the standard MLP.

        Parameters
        ----------
        node_feature_in : `GVec`
            Node features.
        mask : `torch.Tensor` with data type `torch.byte`
            Batch mask for node representations. Shape is (N_batch, N_node).

        Return
        ----------
        node_feature_out : `GVec`
            Output node features.
        """

        node_feature_out = node_feature_in
        # Extract the scalar features
        x = node_feature_out.pop((0, 0)).squeeze(-1)
        s = x.shape
        x = x.permute(1, 2, 3, 0).contiguous().view(s[1:3] + (self.num_scalars,))

        # Loop over a linear layer followed by a non-linear activation.
        for (lin, activation) in zip(self.linear, self.activations):
            x = activation(lin(x))
        # After last non-linearity, apply a final linear mixing layer
        x = self.linear[-1](x)

        # If mask is included, mask the output
        if mask is not None:
            x = torch.where(mask, x, self.zero)

        node_feature_out[(0, 0)] = x.view(s[1:]+(2,)).permute(3, 0, 1, 2).unsqueeze(-1)
        return node_feature_out

    """
    The scaling function for weights in the standard MLP.

    Parameter
    ----------
    scale : `float`
        Scaling parameter
    """

    def scale_weights(self, scale):
        self.linear[-1].weight *= scale
        if self.linear[-1].bias is not None:
            self.linear[-1].bias *= scale
