import torch
import logging
import sys
sys.path.insert(1, '../..')

from lgn.cg_lib import CGModule, normsq4
from lgn.g_lib import GTau

from lgn.models.lgn_graphnet import LGNGraphNet

class LGNEncoder(LGNGraphNet):
    """
    The encoder of the LGN autoencoder.

    Parameters
    ----------
    num_input_particles : `int`
        The number of input particles
    tau_particle_scalar : `int`
        The multiplicity of scalars per particle
        For the hls4ml 150-p jet data, it should be 1 (namely the particle mass -p^2).
    tau_particle_vector : `int`
        The multiplicity of vectors per particle.
        For the hls4ml 150-p jet data, it should be 1 (namely the particle 4-momentum).
    tau_latent_scalars : `int`
        Multiplicity of Lorentz scalars (0,0) in the latent_space.
    tau_latent_vectors : `int`
        Multiplicity of Lorentz 4-vectors (1,1) in the latent_space.
    num_basis_fn : `int`
        The number of basis function to use.
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
    level_gain : `list` of `floats`
        The gain at each level. (args.level_gain = [1.])
    num_latent_particles : `int`
        Optional, default : 1
        The number of particles of jets in the latent space.
    activation : `str`
        Optional, default: 'leakyrelu'
        The activation function for lgn.LGNCG
    scale : `float` or `int`
        Optional, default: 1.
        Scaling parameter for input node features.
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
                 maxdim, num_basis_fn, max_zf, num_cg_levels, num_channels, weight_init, level_gain, num_latent_particles=1,
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

        super().__init__(num_input_particles=num_input_particles, input_basis='cartesian',
                         tau_input_scalars=tau_particle_scalar, tau_input_vectors=tau_particle_vector,
                         num_output_partcles=num_latent_particles, tau_output_scalars=tau_latent_scalars, tau_output_vectors=tau_latent_vectors,
                         max_zf=max_zf, maxdim=max(maxdim + max_zf), num_cg_levels=num_cg_levels, num_channels=num_channels,
                         weight_init=weight_init, level_gain=level_gain, num_basis_fn=num_basis_fn,
                         activation=activation, mlp=mlp, mlp_depth=mlp_depth, mlp_width=mlp_width,
                         device=device, dtype=dtype, cg_dict=cg_dict)

        self.scale = scale
        self.tau_latent = self.tau_output

    '''
    The forward pass of the LGN GNN.

    Parameters
    ----------
    data : `dict`
        The dictionary that stores the hls4ml data, with relevant keys 'p4', 'node_mask', and 'edge_mask'.
    covariance_test : `bool`
        Optional, default: False
        If False, return prediction (scalar reps) only.
        If True, return both predictions and full node features, where the full node features
        will be used to test Lorentz covariance.

    Returns
    -------
    node_features : `dict`
        The dictionary that stores all relevant irreps.
    node_mask : `torch.Tensor`
        The mask of node features. (Unchanged)
    edge_mask : `torch.Tensor`
        The mask of edge features. (Unchanged)
    '''
    def forward(self, data, covariance_test=False):
        # Get data
        node_scalars, node_ps, node_mask, edge_mask = self._prepare_input(data)

        # Can be simplied as self.graph_net(node_scalars, node_ps, node_mask, edge_mask, covariance_test)
        if not covariance_test:
            latent_features, node_mask, edge_mask = super(LGNEncoder, self).forward(node_scalars, node_ps, node_mask, edge_mask, covariance_test)
            return latent_features, edge_mask, edge_mask
        else:
            latent_features, node_mask, edge_mask, nodes_all = super(LGNEncoder, self).forward(node_scalars, node_ps, node_mask, edge_mask, covariance_test)
            return latent_features, edge_mask, edge_mask, nodes_all
    """
    Extract input from data.

    Parameters
    ----------
    data : `dict`
        The jet data.

    Returns
    -------
    scalars : `torch.Tensor`
        Tensor of scalars for each node.
    node_ps: : `torch.Tensor`
        Momenta of the nodes
    node_mask : `torch.Tensor`
        Node mask used for batching data.
    edge_mask: `torch.Tensor`
        Edge mask used for batching data.
    """
    def _prepare_input(self, data):

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
