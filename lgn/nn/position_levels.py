import torch
import torch.nn as nn

from math import pi

from lgn.g_lib import GTau, GScalar

class RadPolyTrig(nn.Module):
    """
    A variation/generalization of spherical bessel functions.
    Rather than than introducing the bessel functions explicitly we just write out a basis
    that can produce them. Then, when apply a weight mixing matrix to reduce the number of channels
    at the end.

    Parameters
    ----------
    max_zf : int
        Maximum weight to use for the spherical harmonics.
    num_basis_fn : int
        The number of basis function to use.
    num_channels : int
        The number of channels of features.
    mix: bool or str
        Optional, default: True
        The rule to mix radial components.
        If type is bool,
            if True, the channel will be mixed to complex shapes; and
            if False, the channel will be mixed to real shapes.
        If type is str,
            the choices are 'cplx' for mixing the radial components to complex shapes,
            'real' for mixing the radial components to real shapes, and
            'None' for not mixing the radial components.
    device : torch.device
        Optional, default: None, in which case it will set to
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        The device to which the module is initialized.
    dtype : torch.dtype
        Optional, default: torch.float64
        The data type to which the module is initialized.
    """
    def __init__(self, max_zf, num_basis_fn, num_channels, mix=True,
                 device=None, dtype=torch.float64):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super(RadPolyTrig, self).__init__()

        self.max_zf = max_zf

        self.num_basis_fn = num_basis_fn
        self.num_channels = num_channels


        self.basis_size = (1, 1, 1, 2 * num_basis_fn)

        # Learnable parameters
        self.a = torch.randn(self.basis_size).to(device=device, dtype=dtype)
        self.b = torch.randn(self.basis_size).to(device=device, dtype=dtype)
        self.c = torch.randn(self.basis_size).to(device=device, dtype=dtype)

        self.a = nn.Parameter(self.a)
        self.b = nn.Parameter(self.b)
        self.c = nn.Parameter(self.c)

        # If desired, mix the radial components to a desired shape
        self.mix = mix
        if (mix == 'cplx') or (mix is True):  # default
            self.linear = nn.ModuleList([nn.Linear(2 * self.num_basis_fn, 2 * self.num_channels).to(device=device, dtype=dtype) for _ in range(max_zf + 1)])
            self.radial_types = (num_channels,) * (max_zf)
        elif mix == 'real':
            self.linear = nn.ModuleList([nn.Linear(2 * self.num_basis_fn, self.num_channels).to(device=device, dtype=dtype) for _ in range(max_zf + 1)])
            self.radial_types = (num_channels,) * (max_zf)
        elif (mix == 'none') or (mix is False):
            self.linear = None
            self.radial_types = (self.num_basis_fn,) * (max_zf)
        else:
            raise ValueError('Can only specify mix = real, cplx, or none! {}'.format(mix))

        self.zero = torch.tensor(0, device=device, dtype=dtype)

    """
    The forward pass for generating the learnable scalar functions.

    Parameters
    ----------
    norms : torch.Tensor
        The norm of the position vector p_{ij}.
    edge_mask : torch.Tensor
        Matrix of the magnitudes of relative position vectors of pairs of nodes
        in momentmum space. Shape is (N_batch, N_node, N_node).

    Return
    ----------
    A GScalar object that stores a dict of radial functions as values
    and weights (l,l) as keys.
    """
    def forward(self, norms, edge_mask):
        # Shape to resize to at the end
        s = norms.shape

        # Mask and reshape
        edge_mask = (edge_mask.byte()).unsqueeze(-1)
        norms = norms.unsqueeze(-1)

        # Lorentzian-bell radial functions: a + 1 / (b + c^2 p^2) when not masked
        rad_trig = torch.where(edge_mask, self.b * (torch.ones_like(self.b) + (self.c * norms).pow(2)).pow(-1) + self.a, self.zero).unsqueeze(-1)
        rad_prod = rad_trig.view(s + (1, 2 * self.num_basis_fn,))

        # Apply linear mixing function, if desired
        if self.mix == 'cplx' or (self.mix is True):
            if len(s) == 3:
                radial_functions = [linear(rad_prod).view(s + (self.num_channels, 2)).permute(4, 0, 1, 2, 3) for linear in self.linear]
            elif len(s) == 2:
                radial_functions = [linear(rad_prod).view(s + (self.num_channels, 2)).permute(3, 0, 1, 2) for linear in self.linear]
        elif self.mix == 'real':
            radial_functions = [linear(rad_prod).view(s + (self.num_channels,)) for linear in self.linear]
        elif (self.mix == 'none') or (self.mix is False):
            radial_functions = [rad_prod.view(s + (self.num_basis_fn, 2)).permute(4, 0, 1, 2, 3)] * (self.max_zf)

        return GScalar({(l, l): radial_function for l, radial_function in enumerate(radial_functions)})


class RadialFilters(nn.Module):
    """
    Generate a set of learnable scalar functions for the aggregation/point-wise
    convolution step.

    One set of radial filters is created for each irrep (l = 0, ..., max_zf).

    Parameters
    ----------
    max_zf : int
        Maximum weight to use for the spherical harmonics.
    basis_set : iterable of int
        Parameters of basis set to use.
        See RadPolyTrig for more details.
    num_channels_out : list of int
        Number of output channels to mix the resulting function.
    num_levels : int
        Number of CG levels in the LGN.
    mix: bool or str
        Optional, default: True
        The rule to mix radial components.
        If type is bool,
            if True, the channel will be mixed to complex shapes; and
            if False, the channel will be mixed to real shapes.
        If type is str,
            the choices are 'cplx' for mixing the radial components to complex shapes,
            'real' for mixing the radial components to real shapes, and
            'None' for not mixing the radial components.
    device : torch.device
        Optional, default: None, in which case it will be set to
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        The device to which the module is initialized.
    dtype : torch.dtype
        Optional, default: torch.float64
        The data type to which the module is initialized.
    """
    def __init__(self, max_zf, num_basis_fn, num_channels_out, num_levels, mix=True, device=None, dtype=torch.float64):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super(RadialFilters, self).__init__()

        self.num_levels = num_levels
        self.max_zf = max_zf

        rad_funcs = [RadPolyTrig(max_zf[level], num_basis_fn, num_channels_out[level], mix=mix, device=device, dtype=dtype) for level in range(self.num_levels)]
        self.rad_funcs = nn.ModuleList(rad_funcs)
        self.tau = [{(l, l): rad_func.radial_types[l - 1] for l in range(0, maxzf + 1)} for rad_func, maxzf in zip(self.rad_funcs, max_zf)]

        if len(self.tau) > 0:
            self.num_rad_channels = self.tau[0][(1, 1)]
        else:
            self.num_rad_channels = 0

        # Other things



        self.zero = torch.tensor(0, device=device, dtype=dtype)

    def forward(self, norms, base_mask):
        """
        The forward pass of the network.

        Parameters
        ----------
        norms : torch.Tensor
            Pairwise distance matrix between nodes.
        base_mask : torch.Tensor
            Masking tensor with 1s on locations that correspond to active edges
            and zero otherwise.

        Return
        -------
        rad_func_vals :  list of RadPolyTrig
            Values of the radial functions.
        """

        return [rad_func(norms, base_mask) for rad_func in self.rad_funcs]
