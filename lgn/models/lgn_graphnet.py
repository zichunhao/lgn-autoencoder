import torch
import logging
import sys
sys.path.insert(1, '../..')

from lgn.cg_lib import CGModule, ZonalFunctionsRel, ZonalFunctions
from lgn.g_lib import GTau

from lgn.models.lgn_cg import LGNCG

from lgn.nn import RadialFilters
from lgn.nn import InputLinear, MixReps

class LGNGraphNet(CGModule):
    """
    The LGN Graph Neural Network architecture. Encoder and decoders are variations of this.
    However, it is not directly used in LGNEncoder and LGNDecoder because they have different
    forward passes and therefore cannot be represented.

    Parameters
    ----------
    input_basis : `str`
        The basis of the input.
        Choices:
        - 'cartesian': Cartesian coordinates
        - 'canonical': The set of spherical harmonics (zonal functions).
    num_input_particles : `int`
        The number of particles in the input.
    tau_output_scalars : `int`
        Multiplicity of Lorentz scalars (0,0) in the output.
    tau_output_vectors : `int`
        Multiplicity of Lorentz 4-vectors (1,1) in the output.
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
    def __init__(self, num_input_particles, input_basis, tau_input_scalars, tau_input_vectors,
                 num_output_partcles, tau_output_scalars, tau_output_vectors, num_basis_fn,
                 maxdim, max_zf, num_cg_levels, num_channels, weight_init, level_gain,
                 activation='leakyrelu', scale=1., mlp=True, mlp_depth=None, mlp_width=None,
                 device=None, dtype=torch.float64, cg_dict=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if dtype is None:
            dtype = torch.float64

        # We parition the output so that there are num_input_particles particles
        assert (num_input_particles * tau_output_scalars % num_output_partcles == 0), \
        f'num_output_partcles {num_output_partcles} has to be a factor of num_input_particles * tau_output_scalars {num_input_particles * tau_output_scalars}!'
        assert (num_input_particles * tau_output_vectors % num_output_partcles == 0), \
        f'num_output_partcles {num_output_partcles} has to be a factor of num_input_particles * tau_output_vectors {num_input_particles * tau_output_vectors}!'

        level_gain = expand_var_list(level_gain, num_cg_levels)
        maxdim = expand_var_list(maxdim, num_cg_levels)
        max_zf = expand_var_list(max_zf, num_cg_levels)
        num_channels = expand_var_list(num_channels, num_cg_levels)

        super().__init__(maxdim=max(maxdim + max_zf), device=device, dtype=dtype, cg_dict=cg_dict)

        self.input_basis = input_basis
        self.num_cg_levels = num_cg_levels
        self.num_channels = num_channels
        self.scale = scale
        self.num_output_particles = num_output_partcles

        # Express input momenta in the basis of spherical harmonics
        self.zonal_fns_in = ZonalFunctions(max(max_zf), basis=self.input_basis,
                                           dtype=dtype, device=device, cg_dict=cg_dict)
        # relative position in momentum space
        self.zonal_fns = ZonalFunctionsRel(max(max_zf), basis=self.input_basis,
                                           dtype=dtype, device=device, cg_dict=cg_dict)

        # Position functions
        self.rad_funcs = RadialFilters(max_zf, num_basis_fn, num_channels, num_cg_levels,
                                       device=device, dtype=dtype)
        tau_pos = self.rad_funcs.tau

        # Input linear layer: Prepare input to the CG layers
        tau_in = GTau({(0,0): tau_input_scalars, (1,1): tau_input_vectors})
        tau_out = GTau({(l,l): num_channels[0] for l in range(max_zf[0] + 1)})
        self.input_func_node = MixReps(tau_in, tau_out, device=device, dtype=dtype)

        tau_input_node = self.input_func_node.tau

        # CG layers
        self.lgn_cg = LGNCG(maxdim, max_zf, tau_input_node, tau_pos,
                            num_cg_levels, num_channels, level_gain, weight_init,
                            mlp=mlp, mlp_depth=mlp_depth, mlp_width=mlp_width,
                            activation=activation, device=self.device, dtype=self.dtype, cg_dict=self.cg_dict)

        self.tau_cg_levels_node = self.lgn_cg.tau_levels_node

        # Output layers
        self.tau_output = GTau({(0,0): tau_output_scalars, (1,1): tau_output_vectors})
        self.mix_reps = MixReps(self.tau_cg_levels_node[-1], self.tau_output, device=device, dtype=dtype)

    def forward(self, node_scalars, node_ps, node_mask, edge_mask, covariance_test=False):
        # Calculate Zonal functions (edge features)
        zonal_functions_in, _, _ = self.zonal_fns_in(node_ps)
        # Cartesian basis is used for the input data
        # All input are real, so [real, imaginary] = [scalars, 0]
        zonal_functions_in[(0, 0)] = torch.stack([node_scalars.unsqueeze(-1),
                                                  torch.zeros_like(node_scalars.unsqueeze(-1))])
        zonal_functions, norms, sq_norms = self.zonal_fns(node_ps, node_ps)

        # Input layer
        if self.num_cg_levels > 0:
            rad_func_levels = self.rad_funcs(norms, edge_mask * (norms != 0).byte())
            node_reps_in = self.input_func_node(zonal_functions_in)
        else:
            rad_func_levels = []
            node_reps_in = self.input_func_node(node_scalars, node_mask)

        # CG layer
        nodes_all = self.lgn_cg(node_reps_in, node_mask, rad_func_levels, zonal_functions)

        # Output layer: output node features to latent space.
        # Size for each rep: (2, batch_size, num_input_particles, tau_rep, dim_rep)
        # (0,0): (2, batch_size, num_input_particles, tau_scalars, 1)
        # (1,1): (2, batch_size, num_input_particles, tau_vectors, 4)
        node_features = nodes_all[-1]
        node_features = self.mix_reps(node_features) # node_all[-1] is the updated feature in the last layer

        # Size for each reshaped rep: (2, batch_size, num_output_particles, tau_rep, dim_rep)
        node_features = {weight: reps.view(2, reps.shape[1], self.num_output_particles, -1, reps.shape[-1]) for weight, reps in node_features.items()}
        return node_features, node_mask, edge_mask

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
