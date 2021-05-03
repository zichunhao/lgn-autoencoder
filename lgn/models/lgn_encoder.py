import torch
import logging
import sys
sys.path.insert(1, '../..')

from lgn.cg_lib import CGModule, normsq4
from lgn.g_lib import GTau

from lgn.models.lgn_graphnet import LGNGraphNet

class LGNEncoder(CGModule):
    """
    The encoder of the LGN autoencoder.

    Parameters
    ----------
    num_input_particles : `int`
        The number of input particles
    tau_latent_scalars : `int`
        Multiplicity of Lorentz scalars (0,0) in the latent_space.
    tau_latent_vectors : `int`
        Multiplicity of Lorentz 4-vectors (1,1) in the latent_space.
    maxdim : `list` of `int`
        Maximum weight in the output of CG products. (Expanded to list of
        length num_cg_levels)
    max_zf : `list` of `int`
        Maximum weight in the output of the spherical harmonics  (Expanded to list of
        length num_cg_levels)
    num_cg_levels : int
        Number of cg levels to use.
    num_channels : `list` of `int`
        Number of channels that the output of each CG are mixed to (Expanded to list of
        length num_cg_levels)
    weight_init : `str`
        The type of weight initialization. The choices are 'randn' and 'rand'.
    level_gain : list of floats
        The gain at each level. (args.level_gain = [1.])
    num_basis_fn : `int`
        The number of basis function to use.
    activation : `str`
        Optional, default: 'leakyrelu'
        The activation function for lgn.LGNCG
    scale : `float` or `int`
        Scaling parameter for node features.
    mlp : `bool`
        Optional, default: True
        Whether to include the extra MLP layer on scalar features in nodes.
    mlp_depth : `int`
        Optional, default: None
        The number of hidden layers in CGMLP.
    mlp_width : `list` of `int`
        Optional, default: None
        The number of perceptrons in each CGMLP layer
    device : `torch.device`
        Optional, default: None, in which case it will be set to
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        The device to which the module is initialized.
    dtype : `torch.dtype`
        Optional, default: torch.float64
        The data type to which the module is initialized.
    cg_dict : `CGDict`
        Optional, default: None
        Clebsch-gordan dictionary for taking the CG decomposition.
    """
    def __init__(self, num_input_particles, tau_particle_scalar, tau_particle_vector, tau_latent_scalars, tau_latent_vectors,
                 maxdim, num_basis_fn, max_zf, num_cg_levels, num_channels, weight_init, level_gain,
                 activation='leakyrelu', scale=1., mlp=True, mlp_depth=None, mlp_width=None,
                 device=None, dtype=torch.float64, cg_dict=None):

        # The extra element accounts for the output channels
        assert len(num_channels) > num_cg_levels, f"num_channels ({num_channels}) must have a length larger than than num_cg_levels ({num_cg_levels})!"

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if dtype is None:
            dtype = torch.float64

        logging.info(f'Initializing encoder with device: {device} and dtype: {dtype}')

        level_gain = expand_var_list(level_gain, num_cg_levels)
        maxdim = expand_var_list(maxdim, num_cg_levels)
        max_zf = expand_var_list(max_zf, num_cg_levels)
        num_channels = expand_var_list(num_channels, num_cg_levels)

        super().__init__(maxdim=max(maxdim + max_zf), device=device, dtype=dtype, cg_dict=cg_dict)

        self.num_input_particles = num_input_particles
        self.input_basis = 'cartesian'
        self.num_latent_particles = 1
        self.tau_latent = GTau({(0, 0): tau_latent_scalars, (1, 1): tau_latent_vectors})
        self.num_cg_levels = num_cg_levels
        self.num_basis_fn = num_basis_fn
        self.scale = scale

        self.graph_net = LGNGraphNet(num_input_particles=self.num_input_particles, input_basis=self.input_basis,
                                     tau_input_scalars=tau_particle_scalar, tau_input_vectors=tau_particle_vector,
                                     num_output_partcles=self.num_latent_particles, tau_output_scalars=tau_latent_scalars, tau_output_vectors=tau_latent_vectors,
                                     max_zf=max_zf, maxdim=maxdim, num_cg_levels=self.num_cg_levels, num_channels=num_channels,
                                     weight_init=weight_init, level_gain=level_gain, num_basis_fn=self.num_basis_fn,
                                     activation=activation, scale=scale, mlp=mlp, mlp_depth=mlp_depth, mlp_width=mlp_width,
                                     device=device, dtype=dtype, cg_dict=cg_dict)

    def forward(self, data, covariance_test=False):
        # Get data
        node_scalars, node_ps, node_mask, edge_mask = self.prepare_input(data)

        latent_features, node_mask, edge_mask = self.graph_net(node_scalars, node_ps, node_mask, edge_mask)

        return latent_features, edge_mask, edge_mask

    """
    Extract input from data.

    Parameters
    ----------
    data : `dict`
        The jet data.

    Returns
    -------
    node_scalars : `torch.Tensor`
        Tensor of scalars for each node.
    node_ps: : `torch.Tensor`
        Momenta of the nodes
    node_mask : `torch.Tensor`
        Node mask used for batching data.
    edge_mask: `torch.Tensor`
        Edge mask used for batching data.
    """
    def prepare_input(self, data):

        node_ps = data['p4'].to(device=self.device, dtype=self.dtype) * self.scale

        data['p4'].requires_grad_(True)

        node_mask = data['node_mask'].to(device=self.device, dtype=torch.uint8)
        edge_mask = data['edge_mask'].to(device=self.device, dtype=torch.uint8)

        scalars = torch.ones_like(node_ps[:,:, 0]).unsqueeze(-1)
        scalars = normsq4(node_ps).abs().sqrt().unsqueeze(-1)

        if 'scalars' in data.keys():
            scalars = torch.cat([scalars, data['scalars'].to(device=self.device, dtype=self.dtype)], dim=-1)

        return scalars, node_ps, node_mask, edge_mask

############################## Helpers ##############################
"""
Expand variables in a list

Parameters
----------
var : `list`, `int`, or `float`
    The variables
num_cg_levels : `int`
    Number of cg levels to use.

Return
------
var_list : `list`
    The list of variables. The length will be num_cg_levels.
"""
def expand_var_list(var, num_cg_levels):
    if type(var) == list:
        var_list = var + (num_cg_levels - len(var)) * [var[-1]]
    elif type(var) in [float, int]:
        var_list = [var] * num_cg_levels
    else:
        raise ValueError(f'Incorrect type of variables: {type(var)}. ' \
                         'The allowed data types are list, float, or int')
    return var_list
