import torch
import logging

from lgn.cg_lib import CGModule, ZonalFunctionsRel, ZonalFunctions
from lgn.cg_lib.zonal_functions import normsq4, rep_to_p
from lgn.g_lib import GTau, GVec

from lgn.models.lgn_cg import LGNCG
from lgn.nn import RadialFilters
from lgn.nn import MixReps

from lgn.models.utils import adapt_var_list


class LGNEncoder(CGModule):
    """
    The encoder of the LGN autoencoder.

    Parameters
    ----------
    num_input_particles : `int`
        The number of input particles.
    tau_input_scalars : `int`
        The multiplicity of scalars per particle.
        For the hls4ml 150-p jet data, it should be 1 (namely the particle invariant mass -p^2).
    tau_input_vectors : `int`
        The multiplicity of vectors per particle.
        For the hls4ml 150-p jet data, it should be 1 (namely the particle 4-momentum).
    tau_latent_scalars : `int`
        Multiplicity of Lorentz scalars (0,0) in the latent_space.
    tau_latent_vectors : `int`
        Multiplicity of Lorentz 4-vectors (1,1) in the latent_space.
    maxdim : `list` of `int`
        Maximum weight in the output of CG products, expanded or truncated to list of
        length len(num_channels) - 1.
    num_basis_fn : `int`
        The number of basis function to use.
    num_channels : `list` of `int`
        Number of channels that the outputs of each CG layer are mixed to.
    max_zf : `list` of `int`
        Maximum weight in the output of the spherical harmonics, expanded or truncated to list of
        length len(num_channels) - 1.
    weight_init : `str`
        The type of weight initialization. The choices are 'randn' and 'rand'.
    level_gain : `list` of `floats`
        The gain at each level. (args.level_gain = [1.])
    jet_features : `bool`
        Optional, default: False
        Whether to incorporate jet momenta into the model.
    map_to_latent : `str`
        Optional, default: 'mean'
        The way of mapping the graph to latent space.
        Choices:
            - 'sum': sum over all nodes.
            - 'mix': linearly mix each isotypic component of node features.
            - 'mean': taking the mean over all nodes.
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
        Optional, default: None, in which case it will be set to torch.float64
        The data type to which the module is initialized.
    cg_dict : `CGDict`
        Optional, default: None
        Clebsch-gordan dictionary for taking the CG decomposition.
    """

    def __init__(self, num_input_particles, tau_input_scalars, tau_input_vectors,
                 tau_latent_scalars, tau_latent_vectors, maxdim, num_basis_fn,
                 num_channels, max_zf, weight_init, level_gain, jet_features=False,
                 map_to_latent='sum', activation='leakyrelu',
                 scale=1., mlp=True, mlp_depth=None, mlp_width=None,
                 device=None, dtype=None, cg_dict=None):

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if dtype is None:
            dtype = torch.float64

        num_cg_levels = len(num_channels) - 1

        if map_to_latent.lower() not in ['mix', 'sum', 'mean', 'average']:
            raise NotImplementedError(f"map_to_latent can only one of ('mix', 'sum', 'mean'). Found: {map_to_latent}")

        level_gain = adapt_var_list(level_gain, num_cg_levels)
        maxdim = adapt_var_list(maxdim, num_cg_levels)
        max_zf = adapt_var_list(max_zf, num_cg_levels)

        super().__init__(maxdim=max(maxdim + max_zf), device=device, dtype=dtype, cg_dict=cg_dict)
        logging.info(f'Initializing encoder with device: {self.device} and dtype: {self.dtype}')

        # Member varibles
        self.num_input_particles = num_input_particles
        self.input_basis = 'cartesian'
        self.num_cg_levels = num_cg_levels
        self.num_basis_fn = num_basis_fn
        self.max_zf = max_zf
        self.num_channels = num_channels
        self.jet_features = jet_features
        self.map_to_latent = map_to_latent
        self.mlp = mlp
        self.mlp_depth = mlp_depth
        self.mlp_width = mlp_width
        self.activation = activation

        if jet_features:
            self.num_input_particles += 1

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
        tau_in = GTau({**{(0, 0): tau_input_scalars, (1, 1): tau_input_vectors},
                       **{(l, l): 1 for l in range(2, max_zf[0] + 1)}})
        # A dictionary of multiplicities in the model (updated as the model is built)
        self.tau_dict = {'input': tau_in}
        # tau_out = GTau({weight: num_channels[0] for weight in [(0, 0), (1, 1)]})
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
        # Mix to latent nodes
        if self.map_to_latent.lower() == 'mix':
            self.tau_cg_levels_node[-1] = GTau({weight: int(value * num_input_particles)
                                                for weight, value in self.tau_cg_levels_node[-1]})

        self.tau_output = {weight: 1 for weight in self.tau_cg_levels_node[-1].keys()}
        self.tau_output[(0, 0)] = tau_latent_scalars
        self.tau_output[(1, 1)] = tau_latent_vectors
        self.tau_dict['latent'] = self.tau_output
        self.mix_reps = MixReps(self.tau_cg_levels_node[-1], self.tau_output, device=self.device, dtype=self.dtype)

        self.scale = scale
        self.tau_latent = self.tau_output

        logging.info(f'Encoder initialized. Number of parameters: {sum(p.nelement() for p in self.parameters())}')

    def forward(self, data, covariance_test=False):
        '''
        The forward pass of the LGN GNN.

        Parameters
        ----------
        data : `dict`
            The dictionary that stores the hls4ml data, with relevant keys 'p4', 'node_mask', and 'edge_mask'.
        covariance_test : `bool`
            Optional, default: False
            If False, return prediction (scalar reps) only.
            If True, return both generated output and full node features, where the full node features
            will be used to test Lorentz covariance.

        Returns
        -------
        node_features : `dict`
            The dictionary that stores all relevant irreps.
        node_mask : `torch.Tensor`
            The mask of node features. (Unchanged)
        edge_mask : `torch.Tensor`
            The mask of edge features. (Unchanged)
        If covariance_test is True, also:
            nodes_all : `list` of `GVec`
                The full node features in the encoder.
        '''
        # Extract node features and masks
        node_scalars, node_ps, node_mask, edge_mask = self._prepare_input(data)

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
        node_features = nodes_all[-1]

        # Size for each reshaped rep: (2, batch_size, 1, tau_rep, dim_rep)
        if self.map_to_latent.lower() == 'mix':
            node_features = GVec({weight: reps.view(2, reps.shape[1], 1, -1, reps.shape[-1])
                                  for weight, reps in node_features.items()})
        # Mix
        # node_all[-1] is the updated feature in the last layer
        latent_features = self.mix_reps(node_features)
        latent_features = {weight: latent_features[weight] for weight in [(0, 0), (1, 1)]}  # Truncate higher order irreps than (1, 1)

        if self.map_to_latent.lower() == 'sum':
            latent_features = GVec({weight: torch.sum(value, dim=-3).unsqueeze(dim=-3)
                                    for weight, value in latent_features.items()})
        elif self.map_to_latent.lower() in ['mean', 'average']:
            latent_features = GVec({weight: torch.mean(value, dim=-3).unsqueeze(dim=-3)
                                    for weight, value in latent_features.items()})

        latent_features_canonical = GVec({weight: val.clone()
                                          for weight, val in latent_features.items()})
        latent_features[(1, 1)] = rep_to_p(latent_features[(1, 1)])  # Convert to Cartesian coordinates

        if not covariance_test:
            return latent_features
        else:
            nodes_all.append(GVec(node_features))
            nodes_all.append(GVec(latent_features_canonical))
            return latent_features, nodes_all

    def _prepare_input(self, data):
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

        node_ps = data['p4'].to(device=self.device, dtype=self.dtype) * self.scale
        if self.jet_features:
            node_ps = torch.cat((node_ps, torch.sum(node_ps, dim=-2).unsqueeze(-1)), dim=-2)

        data['p4'].requires_grad_(True)

        # Calculate node masks and edge masks
        if 'labels' in data:
            node_mask = data['labels'].to(device=self.device)
            node_mask = node_mask.to(torch.uint8)
        else:
            node_mask = data['p4'][..., 0] != 0
            node_mask = node_mask.to(device=self.device, dtyle=torch.uint8)
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)

        scalars = torch.ones_like(node_ps[:, :, 0]).unsqueeze(-1)
        scalars = normsq4(node_ps).abs().sqrt().unsqueeze(-1)

        if 'scalars' in data.keys():
            scalars = torch.cat([scalars, data['scalars'].to(device=self.device, dtype=self.dtype)], dim=-1)

        return scalars, node_ps, node_mask, edge_mask
