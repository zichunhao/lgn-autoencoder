import torch
import logging
import sys
sys.path.insert(1, '../..')

from lgn.cg_lib import CGModule, ZonalFunctionsRel, ZonalFunctions, normsq4
from lgn.g_lib import GTau

from lgn.models.lgn_cg import LGNCG

from lgn.nn import RadialFilters
from lgn.nn import InputLinear, MixReps

class Decoder(CGModule):
    """
    The encoder of the LGN autoencoder.

    Parameters
    ----------
    num_output_scalars : `int`
        Multiplicity of Lorentz scalars (0,0) in the output per jet.
    num_output_vectors : `int`
        Multiplicity of Lorentz 4-vectors (1,1) in the output per jet.
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
    p4_into_CG : `bool`
        Optional, default: False
        Whether or not to feed in 4-momenta themselves to the first CG layer,
        in addition to scalars.
            - If true, MixReps will be used for the input linear layer of the model.
            - If false, IntputLinear will be used.
    add_beams : `bool`
        Optional, default: False
        Append two proton beams of the form (m^2,0,0,+-1) to each event
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
    def __init__(self, num_latent_scalars, num_latent_vectors,
                 maxdim, max_zf, num_cg_levels, num_channels,
                 weight_init, level_gain, num_basis_fn,
                 activation='leakyrelu', p4_into_CG=True,
                 add_beams=False, scale=1., mlp=True, mlp_depth=None, mlp_width=None,
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

        self.num_cg_levels = num_cg_levels
        self.num_channels = num_channels
        self.scale = scale
        self.p4_into_CG = p4_into_CG
        self.tau_latent = GTau({(0,0): num_latent_scalars, (1,1): num_latent_vectors})

        # spherical harmonics
        if p4_into_CG:
            # Express input momenta in the bases of spherical harmonics
            self.zonal_fns_in = ZonalFunctions(max(max_zf), dtype=dtype,
                                               device=device, cg_dict=cg_dict)
        # relative position in momentum space
        self.zonal_fns = ZonalFunctionsRel(max(max_zf), dtype=dtype,
                                           device=device, cg_dict=cg_dict)

        # Position functions
        self.rad_funcs = RadialFilters(max_zf, num_basis_fn, num_channels, num_cg_levels,
                                       device=device, dtype=dtype)
        tau_pos = self.rad_funcs.tau

        if num_cg_levels:
            if add_beams:
                num_scalars_in = 2
            else:
                num_scalars_in = 1
        else:
            num_scalars_in = 150+2  # number of particles per jet (after padding)

        num_scalars_out = num_channels[0]

        # Input linear layer: self.input_func_node
        if not num_cg_levels:
            self.input_func_node = InputLinear(num_scalars_in, num_scalars_out,
                                               device=device, dtype=dtype)
        else:
            tau_in = GTau({**{(0,0): num_scalars_in}, **{(l,l): 1 for l in range(1, max_zf[0] + 1)}})
            tau_out = GTau({(l,l): num_scalars_out for l in range(max_zf[0] + 1)})
            self.input_func_node = MixReps(tau_in, tau_out, device=device, dtype=dtype)

        tau_input_node = self.input_func_node.tau

        # CG layers
        self.lgn_cg = LGNCG(maxdim, max_zf, tau_input_node, tau_pos,
                            num_cg_levels, num_channels, level_gain, weight_init,
                            mlp=mlp, mlp_depth=mlp_depth, mlp_width=mlp_width,
                            activation=activation, device=self.device, dtype=self.dtype, cg_dict=self.cg_dict)

        tau_cg_levels_node = self.lgn_cg.tau_levels_node

        # Output layers
        self.mix_reps = MixReps(tau_cg_levels_node[-1], self.tau_latent, device=device, dtype=dtype)

    def forward(self, data, covariance_test=False):
        # Get data
        node_scalars, node_ps, node_mask, edge_mask = prepare_input(data, self.scale, self.num_cg_levels,
                                                                    device=self.device, dtype=self.dtype)

        # Calculate Zonal functions
        if self.p4_into_CG:
            zonal_functions_in, _, _ = self.zonal_fns_in(node_ps)
            # all input are so far reals, so [real, imaginary] = [scalars, 0]
            zonal_functions_in[(0, 0)] = torch.stack([node_scalars.unsqueeze(-1),
                                                      torch.zeros_like(node_scalars.unsqueeze(-1))])
        zonal_functions, norms, sq_norms = self.zonal_fns(node_ps, node_ps)

        # Input layer
        if self.num_cg_levels > 0:
            rad_func_levels = self.rad_funcs(norms, edge_mask * (norms != 0).byte())
            # Feed scalars only
            if not self.p4_into_CG:
                node_reps_in = self.input_func_node(node_scalars, node_mask)
            # Feed both scalars and 4-momenta
            else:
                node_reps_in = self.input_func_node(zonal_functions_in)
        else:
            rad_func_levels = []
            node_reps_in = self.input_func_node(node_scalars, node_mask)

        # CG layer
        nodes_all = self.lgn_cg(node_reps_in, node_mask, rad_func_levels, zonal_functions)

        # Output
        # size for each representation: (2, batch_size, num_input_particles, tau_rep, dim_rep)
        # (0,0): (2, batch_size, num_input_particles, tau_scalars, 1)
        # (1,1): (2, batch_size, num_input_particles, tau_vectors, 4)
        node_feature = self.mix_reps(nodes_all[-1])

        # size for each representation: (2, batch_size, num_input_particles * tau_rep, dim_rep)
        node_feature = {weight: reps.view(2, reps.shape[1], -1, reps.shape[-1]) for weight, reps in node_feature.items()}

        return node_feature

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
