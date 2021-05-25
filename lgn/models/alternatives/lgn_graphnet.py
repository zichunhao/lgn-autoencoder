import torch

from lgn.cg_lib import CGModule, ZonalFunctionsRel, ZonalFunctions
from lgn.g_lib import GTau

from lgn.models.lgn_cg import LGNCG

from lgn.nn import RadialFilters
from lgn.nn import MixReps
from lgn.g_lib.g_vec import GVec


class LGNGraphNet(CGModule):
    """
    The LGN Graph Neural Network architecture. Encoder and decoders are variations of this.

    Parameters
    ---------
    num_input_particles : `int`
        The number of particles in the input.
    input_basis : `str`
        The basis of the input.
        Options:
        - 'cartesian': Cartesian coordinates
        - 'canonical': The set of spherical harmonics (zonal functions).
    tau_input_scalars : int
        The multiplicity of the input scalar.
    tau_input_vectors : int
        The multiplicity of the input vectors.
    num_output_particles : int
        The number of particles in the output.
        - For Encoder, it will be set to 1, which means that the latent space is equivalent to
          one particle.
        - For Decoder, it will be set to 150 (the number of particles per jet in the hls4ml 150-p data)
    tau_output_scalars : `int`
        Multiplicity of Lorentz scalars (0,0) in the output.
    tau_output_vectors : `int`
        Multiplicity of Lorentz 4-vectors (1,1) in the output.
    num_basis_fn : `int`
        The number of basis function to use.
    maxdim : `list` of `int`
        Maximum weight in the output of CG products, expanded or truncated to list of
        length len(num_channels) - 1.
    num_channels : `list` of `int`
        Number of channels that the output of each CG are mixed to.
    max_zf : `list` of `int`
        Maximum weight in the output of the spherical harmonics, expanded or truncated to list of
        length len(num_channels) - 1.
    weight_init : `str`
        The type of weight initialization. The choices are 'randn' and 'rand'.
    level_gain : list of floats
        The gain at each level. (args.level_gain = [1.])
    activation : `str`
        Optional, default: 'leakyrelu'
        The activation function for lgn.LGNCG
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
                 num_output_particles, tau_output_scalars, tau_output_vectors, num_basis_fn,
                 maxdim, num_channels, max_zf, weight_init, level_gain,
                 activation='leakyrelu', mlp=True, mlp_depth=None, mlp_width=None,
                 device=None, dtype=torch.float64, cg_dict=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if dtype is None:
            dtype = torch.float64

        num_cg_levels = len(num_channels) - 1

        # We parition the output so that there are num_input_particles particles
        assert (num_input_particles * tau_output_scalars % num_output_particles == 0), \
            f'num_output_particles {num_output_particles} has to be a factor of num_input_particles * tau_output_scalars {num_input_particles * tau_output_scalars}!'
        assert (num_input_particles * tau_output_vectors % num_output_particles == 0), \
            f'num_output_particles {num_output_particles} has to be a factor of num_input_particles * tau_output_vectors {num_input_particles * tau_output_vectors}!'

        level_gain = adapt_var_list(level_gain, num_cg_levels)
        maxdim = adapt_var_list(maxdim, num_cg_levels)
        max_zf = adapt_var_list(max_zf, num_cg_levels)

        super().__init__(maxdim=max(maxdim + max_zf), device=device, dtype=dtype, cg_dict=cg_dict)

        # Member varibles
        self.input_basis = input_basis
        self.num_cg_levels = num_cg_levels
        self.num_basis_fn = num_basis_fn
        self.max_zf = max_zf
        self.num_channels = num_channels
        self.num_output_particles = num_output_particles
        self.mlp = mlp
        self.mlp_depth = mlp_depth
        self.mlp_width = mlp_width
        self.activation = activation

        # Express input momenta in the basis of spherical harmonics
        self.zonal_fns_in = ZonalFunctions(max(self.max_zf), basis=self.input_basis,
                                           dtype=dtype, device=device, cg_dict=cg_dict)
        # Relative position in momentum space
        self.zonal_fns = ZonalFunctionsRel(max(self.max_zf), basis=self.input_basis,
                                           dtype=dtype, device=device, cg_dict=cg_dict)

        # Position functions
        self.rad_funcs = RadialFilters(self.max_zf, self.num_basis_fn, self.num_channels, self.num_cg_levels,
                                       device=self.device, dtype=self.dtype)
        tau_pos = self.rad_funcs.tau

        # Input linear layer: Prepare input to the CG layers
        tau_in = GTau({(0, 0): tau_input_scalars, (1, 1): tau_input_vectors})
        # A dictionary of multiplicities in the model (updated as the model is built)
        self.tau_dict = {'input': tau_in}
        tau_out = GTau({(l, l): num_channels[0] for l in range(max_zf[0] + 1)})
        self.input_func_node = MixReps(tau_in, tau_out, device=device, dtype=dtype)

        tau_input_node = self.input_func_node.tau

        # CG layers
        self.lgn_cg = LGNCG(maxdim, self.max_zf, tau_input_node, tau_pos, self.num_cg_levels, self.num_channels,
                            level_gain, weight_init, mlp=self.mlp, mlp_depth=self.mlp_depth, mlp_width=self.mlp_width,
                            activation=self.activation, device=self.device, dtype=self.dtype, cg_dict=self.cg_dict)

        self.tau_cg_levels_node = self.lgn_cg.tau_levels_node
        self.tau_dict['cg_layers'] = self.tau_cg_levels_node.copy()

        # Output layers
        self.tau_cg_levels_node[-1] = GTau({weight: int(value * num_input_particles / num_output_particles)
                                            for weight, value in self.tau_cg_levels_node[-1]})
        self.tau_dict['reshape'] = self.tau_cg_levels_node[-1]
        self.tau_output = GTau({(0, 0): tau_output_scalars, (1, 1): tau_output_vectors})
        self.tau_dict['output'] = self.tau_output
        self.mix_reps = MixReps(
            self.tau_cg_levels_node[-1], self.tau_output, device=self.device, dtype=self.dtype)

    def forward(self, node_scalars, node_ps, node_mask, edge_mask, covariance_test=False):
        '''
        The forward pass of the LGN GNN.

        Parameters
        ----------
        node_scalars : `GTensor` or `torch.Tensor`
            The node scalar features.
            Shape: (2, batch_size, num_input_particles, tau_input_scalars, 1).
        node_ps : `GTensor` or `torch.Tensor`
            The node 4-vector features.
            Shape: (2, batch_size, num_input_particles, tau_input_scalars, 4).
        node_mask : `torch.Tensor`
            The mask of node features.
        edge_mask : `torch.Tensor`
            The mask of edge features.

        Returns
        -------
        node_features : `dict`
            The dictionary that stores all relevant irreps.
        '''
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

        # Size for each reshaped rep: (2, batch_size, num_output_particles, tau_rep, dim_rep)
        node_features = GVec({weight: reps.view(2, reps.shape[1], self.num_output_particles, -1, reps.shape[-1])
                              for weight, reps in node_features.items()})
        # Mix
        # node_all[-1] is the updated feature in the last layer
        node_features = self.mix_reps(node_features)
        if not covariance_test:
            return node_features
        else:
            nodes_all.append(GVec(node_features))
            return node_features, nodes_all


############################## Helpers ##############################
"""
Adapt variables.
- If type(var) is `float` or `int`, adapt it to [var] * num_cg_levels.
- If type(var) is `list`, adapt the length to num_cg_levels.

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


def adapt_var_list(var, num_cg_levels):
    if type(var) == list:
        if len(var) < num_cg_levels:
            var_list = var + (num_cg_levels - len(var)) * [var[-1]]
        elif len(var) == num_cg_levels:
            var_list = var
        elif len(var) > num_cg_levels:
            var_list = var[:num_cg_levels - 1]
        else:
            raise ValueError(f'Invalid length of var: {len(var)}')
    elif type(var) in [float, int]:
        var_list = [var] * num_cg_levels
    else:
        raise ValueError(f'Incorrect type of variables: {type(var)}. '
                         'The allowed data types are list, float, or int')
    return var_list
