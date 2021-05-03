import torch
import torch.nn as nn

from lgn.cg_lib import CGModule
from lgn.g_lib import GTau, GScalar


class BasicMLP(nn.Module):

    """
    The general MLP. It operates only on the last axis of the data.

    Parameters
    ----------
    num_in : int
        Number of input channels
    num_out : int
        Number of output channels
    num_hidden : int,
        Optional, default: 1
        Number of hidden layers.
    layer_width : int
        Optional, default: 256
        Number of perceptrons per hidden layer.
    activation : str
        Optional, default: 'leakyrelu'
        The type of activation function to use.
        Options are ‘leakyrelu’(for nn.LeakyReLU()), ‘relu’(for nn.ReLU()),
        ‘elu’(for nn.ELU()), ‘sigmoid’(for nn.Sigmoid()),‘logsigmoid’(for nn.LogSigmoid()),
        and ‘atan’(for torch.atan(input))
    device : torch.device
        Optional, default: torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        The device to which the module is initialized.
    dtype : torch.dtype
        Optional, default: torch.float64
        The data type to which the module is initialized.
    """
    def __init__(self, num_in, num_out, num_hidden=1, layer_width=256, activation='leakyrelu',
                 device=None, dtype=torch.float64):

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        super(BasicMLP, self).__init__()

        self.num_in = num_in
        self.num_out = num_out

        # Linear layers
        self.linear = nn.ModuleList()
        self.linear.append(nn.Linear(num_in, layer_width))
        for _ in range(num_hidden - 1):
            self.linear.append(nn.Linear(layer_width, layer_width))
        self.linear.append(nn.Linear(layer_width, num_out))

        # Activation functions
        activation_fn = get_activation_fn(activation)
        self.activations = nn.ModuleList()
        for _ in range(num_hidden):
            self.activations.append(activation_fn)

        # Used for masking
        self.zero = torch.tensor(0, device=device, dtype=dtype)

        self.to(device=device, dtype=dtype)

    """
    The forward pass for standard MLP

    Parameters
    ----------
    x : GVec or torch.Tensor
        Input features.
    mask : torch.Tensor
        Mask for features.

    Return
    ----------
    x : GVec or torch.Tensor
        Updated features
    """
    def forward(self, x, mask=None):

        for (lin, activation) in zip(self.linear, self.activations):
            x = lin(x)
            x = activation(x)

        x = self.linear[-1](x)  # Final linear mixing layer

        # Mask the output if mask is specified
        if mask is not None:
            x = torch.where(mask, x, self.zero)

        return x

    """
    The scaling function for weights in the standard MLP.

    Parameter
    ----------
    scale : float or int
        Scaling parameter.
    """
    def scale_weights(self, scale):
        self.linear[-1].weight *= scale
        if self.linear[-1].bias is not None:
            self.linear[-1].bias *= scale

############# Helper function and class #############
def get_activation_fn(activation):
    activation = activation.lower()
    if activation == 'leakyrelu':
        activation_fn = nn.LeakyReLU()
    elif activation == 'relu':
        activation_fn = nn.ReLU()
    elif activation == 'elu':
        activation_fn = nn.ELU()
    elif activation == 'sigmoid':
        activation_fn = nn.Sigmoid()
    elif activation == 'logsigmoid':
        activation_fn = nn.LogSigmoid()
    elif activation == 'atan':
        activation_fn = ATan()
    else:
        raise ValueError(f'Activation function {activation} not implemented!')
    return activation_fn


class ATan(torch.nn.Module):
    def forward(self, input):
        return torch.atan(input)
