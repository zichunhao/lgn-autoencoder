from typing import Dict, List, Tuple, Union
import torch
import numpy as np
import logging

from lgn.cg_lib import CGModule, ZonalFunctionsRel, ZonalFunctions
from lgn.cg_lib.cg_dict import CGDict
from lgn.cg_lib.zonal_functions import normsq4, rep_to_p, repdot
from lgn.g_lib import GTau, GVec

from lgn.models.lgn_cg import LGNCG
from lgn.nn import RadialFilters
from lgn.nn import MixReps

from lgn.models.utils import adapt_var_list, resolve_tau_latent

IMPLEMENTED_AGGREGATIONS = ("mix", "sum", "mean", "average", "min", "max")


class LGNEncoder(CGModule):
    """
    The encoder of the LGN autoencoder.
    """

    def __init__(
        self,
        num_input_particles: int,
        tau_input_scalars: int,
        tau_input_vectors: int,
        maxdim: int,
        num_basis_fn: int,
        num_channels: List[int],
        max_zf: List[int],
        weight_init: List[float],
        level_gain: List[float],
        activation: str = "leakyrelu",
        mlp: bool = True,
        mlp_depth: int = None,
        mlp_width: int = None,
        scale: float = 1.0,
        jet_features: bool = False,
        map_to_latent: str = "mean",
        tau_latent: dict = None,
        tau_latent_scalars: int = None,
        tau_latent_vectors: int = None,
        device: torch.device = None,
        dtype: torch.dtype = None,
        cg_dict: CGDict = None,
    ):
        """
        Parameters
        ----------
        num_input_particles : int
            The number of input particles.
        tau_input_scalars : int
            The multiplicity of scalars per particle.
            e.g. For the hls4ml 150p or 30p jet data, it is 1 (namely the particle invariant mass -p^2).
        tau_input_vectors : int
            The multiplicity of vectors per particle.
            e.g. For the hls4ml 150p or 30p jet data, it is 1 (namely the particle 4-momentum).
        maxdim : list of int
            Maximum weight in the output of CG products, expanded or truncated to list of
            length len(num_channels) - 1.
        num_basis_fn : int
            The number of basis function to use.
        num_channels : list of int
            Number of channels that the outputs of each CG layer are mixed to.
        max_zf : list of int
            Maximum weight in the output of the spherical harmonics, expanded or truncated to list of
            length len(num_channels) - 1.
        weight_init : str
            The type of weight initialization. The choices are 'randn' and 'rand'.
        level_gain : list of floats
            The gain at each level. (args.level_gain = [1.])
        jet_features : bool
            Optional, default: False
            Whether to incorporate jet momenta into the model.
        map_to_latent : str
            Optional, default: 'mean'
            The way of mapping the graph to latent space (aggregation method).
            Choices:
                - 'mix': linearly mix each isotypic component of node features.
                - 'sum': sum over all nodes.
                - 'mean': taking the mean over all nodes.
                - 'min': taking the minimum over all nodes.
                - 'max': taking the maximum over all nodes.
                - Any combination of ('sum', 'mean', 'min', 'max') with '+' to add all features.
                - Any combination of ('sum', 'mean', 'min', 'max') with '&' to concatenate all features.
        tau_latent : dict of {(k, n): int}, list of str, or None
            Optional, default: None
            Multiplicity of each Lorentz irrep (k, n) to keep in the latent space, e.g.
            ``{(0, 0): 1, (1, 1): 8, (2, 2): 2}``. A list of ``"k,n:mult"`` tokens is also
            accepted. Every requested irrep must be produced by the last CG layer, which
            is controlled by ``maxdim`` (only (0,0) and (1,1) exist for maxdim=2; higher
            irreps such as (2,2), (0,2), (2,0) require maxdim >= 3). If None, the latent
            space falls back to the legacy ``tau_latent_scalars``/``tau_latent_vectors``.
        tau_latent_scalars : int
            Optional, default: None
            Legacy shortcut for the multiplicity of Lorentz scalars (0,0) in the latent
            space. Used only when ``tau_latent`` is None.
        tau_latent_vectors : int
            Optional, default: None
            Legacy shortcut for the multiplicity of Lorentz 4-vectors (1,1) in the latent
            space. Used only when ``tau_latent`` is None.
        activation : str
            Optional, default: 'leakyrelu'
            The activation function for lgn.LGNCG
        scale : float or int
            Optional, default: 1.
            Scaling parameter for input node features.
        mlp : bool
            Optional, default: True
            Whether to include the extra MLP layer on scalar features in nodes.
        mlp_depth : int
            Optional, default: None
            The number of hidden layers in CGMLP.
        mlp_width : list of int
            Optional, default: None
            The number of perceptrons in each CGMLP layer
        device : torch.device
            Optional, default: None, in which case it will be set to
                torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            The device to which the module is initialized.
        dtype : torch.dtype
            Optional, default: None, in which case it will be set to torch.float64
            The data type to which the module is initialized.
        cg_dict : CGDict
            Optional, default: None
            Clebsch-gordan dictionary for taking the CG decomposition.
        """

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if dtype is None:
            dtype = torch.float64

        # Latent-space irreps as a {(k, n): multiplicity} dict (with legacy fallback).
        tau_latent = resolve_tau_latent(
            tau_latent, tau_latent_scalars, tau_latent_vectors
        )

        num_cg_levels = len(num_channels) - 1

        # if map_to_latent.lower() not in IMPLEMENTED_AGGREGATIONS:
        #     raise NotImplementedError(
        #         f"map_to_latent can only one of {IMPLEMENTED_AGGREGATIONS}. "
        #         f"Found: {map_to_latent}"
        #     )

        level_gain = adapt_var_list(level_gain, num_cg_levels)
        maxdim = adapt_var_list(maxdim, num_cg_levels)
        max_zf = adapt_var_list(max_zf, num_cg_levels)

        super().__init__(
            maxdim=max(maxdim + max_zf), device=device, dtype=dtype, cg_dict=cg_dict
        )
        misc_info = {"dtype": self.dtype, "device": self.device}
        logging.info(
            f"Initializing encoder with device: {self.device} and dtype: {self.dtype}"
        )

        # Member variables
        self.num_input_particles = num_input_particles
        self.input_basis = "cartesian"
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
            tau_input_scalars += 1  # invariant mass of the jet

        # Express input momenta in the canonical basis
        self.zonal_fns_in = ZonalFunctions(
            maxdim=max(self.max_zf),
            basis=self.input_basis,
            cg_dict=cg_dict,
            **misc_info,
        )
        # Relative position in momentum space
        self.zonal_fns = ZonalFunctionsRel(
            maxdim=max(self.max_zf),
            basis=self.input_basis,
            cg_dict=cg_dict,
            **misc_info,
        )

        # Position functions
        self.rad_funcs = RadialFilters(
            max_zf=self.max_zf,
            num_basis_fn=self.num_basis_fn,
            num_channels_out=self.num_channels,
            num_levels=self.num_cg_levels,
            **misc_info,
        )
        tau_pos = self.rad_funcs.tau

        # Input linear layer: Prepare input to the CG layers
        tau_in = GTau(
            {
                **{(0, 0): tau_input_scalars, (1, 1): tau_input_vectors},
                **{(l, l): 1 for l in range(2, max_zf[0] + 1)},
            }
        )
        # A dictionary of multiplicities in the model (updated as the model is built)
        self.tau_dict = {"input": tau_in}
        # tau_out = GTau({weight: num_channels[0] for weight in [(0, 0), (1, 1)]})
        tau_out = GTau({(l, l): num_channels[0] for l in range(max_zf[0] + 1)})
        self.input_func_node = MixReps(tau_in, tau_out, **misc_info)

        tau_input_node = self.input_func_node.tau

        # CG layers
        self.lgn_cg = LGNCG(
            maxdim=maxdim,
            max_zf=self.max_zf,
            tau_in=tau_input_node,
            tau_pos=tau_pos,
            num_cg_levels=self.num_cg_levels,
            num_channels=self.num_channels,
            level_gain=level_gain,
            weight_init=weight_init,
            mlp=self.mlp,
            mlp_depth=self.mlp_depth,
            mlp_width=self.mlp_width,
            activation=self.activation,
            cg_dict=self.cg_dict,
            **misc_info,
        )

        self.tau_cg_levels_node = self.lgn_cg.tau_levels_node
        self.tau_dict["cg_layers"] = self.tau_cg_levels_node.copy()

        # Output layers
        # Mix to latent nodes
        if self.map_to_latent.lower() == "mix":
            self.tau_cg_levels_node[-1] = GTau(
                {
                    weight: int(value * num_input_particles)
                    for weight, value in self.tau_cg_levels_node[-1]
                }
            )

        # Only irreps produced by the last CG layer can be kept in the latent space.
        available_weights = list(self.tau_cg_levels_node[-1].keys())
        missing = [weight for weight in tau_latent if weight not in available_weights]
        if missing:
            raise ValueError(
                f"Requested latent irreps {missing} are not produced by the last CG "
                f"layer. Available irreps: {sorted(available_weights)}. Increase "
                "`maxdim` or choose from the available irreps (only (0,0) and (1,1) "
                "exist for maxdim=2; higher irreps require maxdim >= 3)."
            )

        # `mix_reps` mixes every available irrep (multiplicity 1 by default); the
        # requested latent irreps get their configured multiplicity. The forward pass
        # then keeps only the requested `self.latent_weights`.
        self.tau_output = {weight: 1 for weight in available_weights}
        for weight, mult in tau_latent.items():
            self.tau_output[weight] = int(mult)
        self.latent_weights = list(tau_latent.keys())
        self.tau_latent = GTau({weight: int(mult) for weight, mult in tau_latent.items()})
        self.tau_dict["latent"] = dict(self.tau_latent)
        self.mix_reps = MixReps(
            self.tau_cg_levels_node[-1], self.tau_output, **misc_info
        )

        self.scale = scale

        self.__num_param = sum(
            p.nelement() for p in self.parameters() if p.requires_grad
        )

    def l1_norm(self) -> torch.Tensor:
        return sum(p.abs().sum() for p in self.parameters())

    def l2_norm(self) -> torch.Tensor:
        return sum(torch.pow(p, 2).sum() for p in self.parameters())

    def forward(
        self,
        data: Union[Dict[str, torch.Tensor], torch.Tensor, np.ndarray],
        covariance_test: bool = False,
    ) -> Union[GVec, Tuple[GVec, List[GVec]]]:
        """
        The forward pass of the LGN GNN.

        Parameters
        ----------
        data : `dict`
            The dictionary that stores the hls4ml data, with relevant keys 'p4', 'node_mask', and 'edge_mask'.
        covariance_test : bool
            Optional, default: False
            If False, return prediction (scalar reps) only.
            If True, return both generated output and full node features, where the full node features
            will be used to test Lorentz covariance.

        Returns
        -------
        node_features : GVec
            The dictionary that stores all relevant irreps.
            For each key, the value is a tensor of shape
            `(2, batch_size, num_particles, tau, feature_dim)`.
        If covariance_test is True, also:
            nodes_all : list of `GVec`
                The full node features in the encoder.
        """
        # Extract node features and masks
        node_scalars, node_ps, node_mask, edge_mask = self._prepare_input(data)

        # Calculate Zonal functions (edge features)
        zonal_functions_in, _, _ = self.zonal_fns_in(node_ps)
        # Cartesian basis is used for the input data
        # All input are real, so [real, imaginary] = [scalars, 0]
        zonal_functions_in[(0, 0)] = torch.stack(
            [node_scalars.unsqueeze(-1), torch.zeros_like(node_scalars.unsqueeze(-1))]
        )
        zonal_functions, norms, sq_norms = self.zonal_fns(node_ps, node_ps)

        # Input layer
        if self.num_cg_levels > 0:
            rad_func_levels = self.rad_funcs(norms, edge_mask * (norms != 0).byte())
            node_reps_in = self.input_func_node(zonal_functions_in)
        else:
            rad_func_levels = []
            node_reps_in = self.input_func_node(node_scalars, node_mask)

        # CG layer
        nodes_all = self.lgn_cg(
            node_reps_in, node_mask, rad_func_levels, zonal_functions
        )

        # Output layer: output node features to latent space.
        # Size for each rep: (2, batch_size, num_input_particles, tau_rep, dim_rep)
        node_features = nodes_all[-1]

        # Size for each reshaped rep: (2, batch_size, 1, tau_rep, dim_rep)
        if self.map_to_latent.lower() == "mix":
            node_features = GVec(
                {
                    weight: reps.view(2, reps.shape[1], 1, -1, reps.shape[-1])
                    for weight, reps in node_features.items()
                }
            )
        # Mix
        # node_all[-1] is the updated feature in the last layer
        latent_features = self.mix_reps(node_features)
        # Keep only the requested latent irreps (any (k, n) produced by the last CG layer)
        latent_features = GVec(
            {weight: latent_features[weight] for weight in self.latent_weights}
        )

        # latent_features = self._aggregate(latent_features)
        # The 4-vector irrep (1, 1) is expressed in Cartesian coordinates; every other
        # irrep is kept in the canonical basis (it has no Cartesian counterpart).
        if (1, 1) in latent_features.keys():
            latent_features[(1, 1)] = rep_to_p(latent_features[(1, 1)])
        latent_features = aggregate(self.map_to_latent, latent_features)

        if not covariance_test:
            return latent_features
        else:
            return latent_features, nodes_all

    def _prepare_input(
        self, data: Union[Dict[str, torch.Tensor], torch.Tensor, np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        # convert ot dict
        if isinstance(data, torch.Tensor):
            p4 = data
            data = dict()
            data["p4"] = p4
        elif isinstance(data, np.ndarray):
            p4 = torch.from_numpy(data)
            data = dict()
            data["p4"] = p4

        node_ps = data["p4"].to(device=self.device, dtype=self.dtype) * self.scale
        if self.jet_features:
            jet_p = torch.sum(node_ps, dim=-2).unsqueeze(-2)
            node_ps = torch.cat((node_ps, jet_p), dim=-2)

        scalars = normsq4(node_ps).abs().sqrt().unsqueeze(-1)
        if self.jet_features:
            jet_mass = (
                normsq4(torch.sum(node_ps, dim=-2))
                .unsqueeze(-1)
                .unsqueeze(-1)
                .repeat(1, self.num_input_particles, 1)
            )
            scalars = torch.cat((scalars, jet_mass), dim=-1)

        # Calculate node masks and edge masks
        if "labels" in data:
            node_mask = data["labels"].to(device=self.device)
            node_mask = node_mask.to(torch.uint8)
        elif "masks" in data:
            node_mask = data["masks"].to(device=self.device)
            node_mask = node_mask.to(torch.uint8)
        elif "mask" in data:
            node_mask = data["mask"].to(device=self.device)
            node_mask = node_mask.to(torch.uint8)
        else:
            node_mask = data["p4"][..., 0] != 0
            node_mask = node_mask.to(device=self.device, dtype=torch.uint8)
        if self.jet_features:
            jet_mask = torch.ones_like(node_mask[..., 0:1])
            jet_mask = jet_mask.to(device=self.device, dtype=torch.uint8)
            node_mask = torch.cat((node_mask, jet_mask), dim=-1)
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)

        if "scalars" in data.keys():
            # (0,0) representation
            scalars = torch.cat(
                [scalars, data["scalars"].to(device=self.device, dtype=self.dtype)],
                dim=-1,
            )

        return scalars, node_ps, node_mask, edge_mask

    @property
    def num_learnable_parameters(self) -> int:
        return self.__num_param


def aggregate(map_to_latent: str, latent_features: Union[GVec, torch.Tensor]) -> GVec:
    """Aggregate to the latent space."""
    if map_to_latent.lower() == "sum":
        return GVec(
            {
                weight: torch.sum(value, dim=-3, keepdim=True)
                for weight, value in latent_features.items()
            }
        )

    elif map_to_latent.lower() in ("mean", "average"):
        return GVec(
            {
                weight: torch.mean(value, dim=-3, keepdim=True)
                for weight, value in latent_features.items()
            }
        )

    elif map_to_latent.lower() == "max":
        return GVec(
            {
                weight: get_max_features(value, weight)
                for weight, value in latent_features.items()
            }
        )

    elif map_to_latent.lower() == "min":
        return GVec(
            {
                weight: get_min_features(value, weight)
                for weight, value in latent_features.items()
            }
        )

    elif map_to_latent.lower() == "mix":  # will be processed in the next step
        return latent_features

    # simply add different latent features
    # TODO: learnable parameters based on Lorentz scalars
    elif "+" in map_to_latent.lower():
        if "mix" in map_to_latent.lower():
            raise NotImplementedError(
                "Adding with mix aggregation not implemented yet."
            )
        methods = map_to_latent.split("+")
        if len(methods) < 1:
            raise ValueError(f"No aggregation method specified: {map_to_latent}.")
        weights = latent_features.keys()
        features = [aggregate(method, latent_features) for method in methods]

        return GVec(
            {
                weight: sum([feature[weight] for feature in features]) / len(methods)
                for weight in weights
            }
        )

    elif "&" in map_to_latent:
        if "mix" in map_to_latent.lower():
            raise NotImplementedError(
                "Concatenating with mix aggregation not implemented yet."
            )
        methods = map_to_latent.split("&")
        if len(methods) < 1:
            raise ValueError(f"No aggregation method specified: {map_to_latent}.")
        weights = latent_features.keys()
        features = [aggregate(method, latent_features) for method in methods]
        return GVec(
            {
                weight: torch.cat([feature[weight] for feature in features], dim=3)
                for weight in weights
            }
        )

    else:
        raise NotImplementedError(f"{map_to_latent} is not implemented.")


def get_msq(p4: torch.Tensor, keep_dim=False):
    """Get mass squared of a 4-momentum."""
    E, p3 = p4[..., 0], p4[..., 1:]
    msq = E**2 - torch.norm(p3, dim=-1) ** 2
    if keep_dim:
        return msq.unsqueeze(-1)
    return msq


def gather_righthand(src, index, check=True):
    """
    Index a tensor src based on a tensor index.
    Source: https://stackoverflow.com/a/68198072
    """
    index = index.long()
    i_dim = index.dim()
    s_dim = src.dim()
    t_dim = i_dim - 1
    if check:
        if s_dim <= i_dim:
            raise ValueError(
                f"src.dim() ({src.dim()}) <= index.dim() {index.dim()}."
                "src dimension must be larger than index dimension."
            )
        for d in range(0, t_dim):
            if src.size(d) != index.size(d):
                raise ValueError(
                    f"src.shape[{d}] ({src.shape[d]}) != index.shape[{d}] ({index.shape[d]})."
                    "src and index must have the same shape."
                )
    index_new_shape = list(src.shape)
    index_new_shape[t_dim] = index.shape[t_dim]
    for _ in range(i_dim, s_dim):
        index = index.unsqueeze(-1)

    # only this two line matters
    index_expand = index.expand(index_new_shape)
    # only this two line matters
    return torch.gather(src, dim=t_dim, index=index_expand)


def get_invariant_scalar(feature: torch.Tensor, weight) -> torch.Tensor:
    """
    Real part of the canonical-basis Lorentz-invariant norm-squared of an irrep.

    Ranking particles by this invariant keeps the min/max aggregation Lorentz-
    equivariant for a general irrep. The invariant is broadcast over the complex
    dimension so the same particle is selected for the real and imaginary parts.

    Parameters
    ----------
    feature : torch.Tensor
        A single irrep of shape (2, batch_size, num_particles, tau, feature_dim).
    weight : tuple of int
        The irrep weight (k, n).

    Returns
    -------
    torch.Tensor
        Ranking scalar of shape (2, batch_size, num_particles, tau).
    """
    inv = repdot({weight: feature}, {weight: feature})[weight][0].squeeze(-1)
    return inv.unsqueeze(0).expand(2, *inv.shape)


def get_min_features(feature: torch.Tensor, weight=None):
    # (2, batch_size, num_particles, tau, feature_dim)
    # -> (2, batch_size, tau, num_particles, feature_dim)
    features_permute = feature.permute(0, 1, 3, 2, 4)
    if weight not in (None, (0, 0), (1, 1)):
        # General irrep: rank by the Lorentz-invariant norm-squared.
        scalar = get_invariant_scalar(feature, weight)
    elif feature.shape[-1] == 1:
        scalar = feature.min(dim=-1).values
    elif feature.shape[-1] == 4:
        scalar = get_msq(feature, keep_dim=False)
    else:
        raise NotImplementedError(
            f"feature dimension {feature.shape[-1]} not supported yet"
        )
    indices = torch.min(scalar, dim=-2).indices.unsqueeze(-1)

    # aggregated_permuted = gather_righthand(features_permute, indices)

    # (2, batch_size, tau, num_particles, feature_dim)
    # -> (2, batch_size, num_particles, tau, feature_dim)
    return gather_righthand(features_permute, indices).permute(0, 1, 3, 2, 4)


def get_max_features(feature: torch.Tensor, weight=None):
    """
    Get the maximum features per tau based on the reference and a map function.
    """
    # (2, batch_size, num_particles, tau, feature_dim)
    # -> (2, batch_size, tau, num_particles, feature_dim)
    features_permute = feature.permute(0, 1, 3, 2, 4)
    if weight not in (None, (0, 0), (1, 1)):
        # General irrep: rank by the Lorentz-invariant norm-squared.
        scalar = get_invariant_scalar(feature, weight)
    else:
        scalar = get_msq(feature, keep_dim=False)
    indices = torch.max(scalar, dim=-2).indices.unsqueeze(-1)
    if weight in (None, (0, 0), (1, 1)):
        if feature.shape[-1] == 1:
            scalar = feature.max(dim=-1).values
        elif feature.shape[-1] == 4:
            scalar = get_msq(feature, keep_dim=False)
        else:
            raise NotImplementedError(
                f"feature dimension {feature.shape[-1]} not supported yet"
            )

    # aggregated_permuted = gather_righthand(features_permute, indices)

    # (2, batch_size, tau, num_particles, feature_dim)
    # -> (2, batch_size, num_particles, tau, feature_dim)
    return gather_righthand(features_permute, indices).permute(0, 1, 3, 2, 4)
