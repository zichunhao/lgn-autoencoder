import logging
import torch.nn as nn

from lgn.models import LGNNodeLevel, CGMLP
from lgn.cg_lib import CGModule

class LGNCG(CGModule):

    """
    The initializer of the LGN CG layers.

    Parameters
    ----------
    maxdim : `int`
        The maximum weight of irreps to be accounted.
    max_zf : `int`
        The maximum weight of zonal functions
    tau_in : `GTau`
        The multiplicity of the input features.
    tau_pos : `list` of `GTau` (or compatible)
        The multiplicity of the relative position vectors.
    num_cg_levels : `int`
        The number of steps of message passing.
    num_channels : `list` of `int`
        The number of channels in the LGNNodeLevel.
    level_gain : `list` of `floats`
        The gain at each level. (args.level_gain = [1.])
    weight_init : `str`
        The type of weight initialization. The choices are 'randn' and 'rand'.
    mlp : `bool`
        Optional, default: True
        Whether to include the extra MLP layer on scalar features in nodes.
    mlp_depth : `int`
        Optional, default: None
        The number of hidden layers in CGMLP.
    mlp_width : list of `int`
        Optional, default: None
        The number of perceptrons in each CGMLP layer
    activation : `str`
        Optional, default: 'leakyrelu'
        The type of activation function to use.
        Options are ‘leakyrelu’(for nn.LeakyReLU()), ‘relu’(for nn.ReLU()),
        ‘elu’(for nn.ELU()), ‘sigmoid’(for nn.Sigmoid()),‘logsigmoid’(for nn.LogSigmoid()),
        and ‘atan’(for torch.atan(input))
    device : `torch.device`
        Optional, default: None
        The device to which the module is initialized.
    dtype : `torch.dtype`
        Optional, default: None
        The data type to which the module is initialized.
    cg_dict : `CGDict`
        Optional, default: None
        The CG dictionary as a reference for taking CG decompositions.
    """
    def __init__(self, maxdim, max_zf, tau_in, tau_pos, num_cg_levels, num_channels,
                 level_gain, weight_init, mlp=True, mlp_depth=None, mlp_width=None, activation='leakyrelu',
                 device=None, dtype=None, cg_dict=None):

        super().__init__(device=device, dtype=dtype, cg_dict=cg_dict)
        device, dtype, cg_dict = self.device, self.dtype, self.cg_dict

        self.max_zf = max_zf
        self.mlp = mlp
        tau_node_in = tau_in.tau if type(tau_in) is CGModule else tau_in

        logging.info(f'tau_node_in in LGNCG: {tau_node_in}')

        node_levels = nn.ModuleList()
        if mlp:
            mlp_levels = nn.ModuleList()  # the extra MLP layer after the LGNCG layer

        tau_node = tau_node_in # initial tau


        for layer in range(num_cg_levels):
            node_lvl = LGNNodeLevel(tau_node, tau_pos[layer], maxdim[layer], num_channels[layer+1],
                                    level_gain[layer], weight_init, device=device, dtype=dtype, cg_dict=cg_dict)
            node_levels.append(node_lvl)

            if mlp:
                mlp_lvl = CGMLP(node_lvl.tau_out, activation=activation, num_hidden=mlp_depth,
                                layer_width_mul=mlp_width, device=device, dtype=dtype)
                mlp_levels.append(mlp_lvl)

            tau_node = node_lvl.tau_out # update tau

            logging.info(f'layer {layer} tau_node: {tau_node}')

        self.node_levels = node_levels
        if mlp:
            self.mlp_levels = mlp_levels

        self.tau_levels_node = [tau_node_in] + [level.tau_out for level in node_levels]

    """
    The forward pass of the LGN CG layers.

    Parameters
    ----------
    node_feature :  `GVec`
        Node features.
    node_mask : torch.Tensor with data type torch.byte
        Batch mask for node representations. Shape is (N_batch, N_node).
    rad_funcs : `list` of `GScalars`
        The (possibly learnable) radial filters.
    edge_mask : `torch.Tensor`
        Matrix of the magnitudes of relative position vectors of pairs of nodes
        in momentmum space. Shape is (N_batch, N_node, N_node).
    zonal_functions : `GVec`
        Representation of spherical harmonics calculated from the relative
        position vectors p_{ij} = sqrt{(p_i - p_j)^2} * sign((p_i - p_j)^2).

    Returns
    -------
    nodes_features : `list` of `GVecs`
        The concatenated list of the representations outputted at each round of message passing.
    """
    def forward(self, node_feature, node_mask, rad_funcs, zonal_functions):

        assert len(self.node_levels) == len(rad_funcs), f'The number of layer {len(self.node_levels)} and the number of available radial functions {len(rad_funcs)} are not equal!'

        # message passing
        nodes_features = [node_feature]  # node features in each round of message passing; to be concatenated
        if self.mlp:
            for idx, (node_level, mlp_level) in enumerate(zip(self.node_levels, self.mlp_levels)):
                edge = rad_funcs[idx] * zonal_functions  # edge features
                node_feature = node_level(node_feature, edge, node_mask)  # message aggregation
                node_feature = mlp_level(node_feature)  # the additional MLP layer
                nodes_features.append(node_feature)

        else:
            for idx, node_level in enumerate(self.node_levels):
                edge = rad_funcs[idx] * zonal_functions
                node_feature = node_level(node_feature, edge, node_mask)
                nodes_features.append(node_feature)

        return nodes_features
