import torch
from torch.nn import Module
from lgn.cg_lib import CGModule
from lgn.g_lib import g_torch, GWeight, GTau, GScalar, GVec


class MixReps(CGModule):
    """
    The module to linearly mix a representation.
    The input musst have predefined types tau_in and tau_out.

    Parameters
    ----------
    tau_in : GTau (or compatible object).
        Input tau of representation.
    tau_out : GTau (or compatible object) or int
        Output tau of representation. If type(tau_out) == int,
        the output type will be set to tau_out for each
        parameter in the network.
    real : bool
        Optional, default: False
        Whether to use purely real mixing weights.
    weight_init : str
        Optional, default: 'randn'
        The type of weight initialization. The choices are 'randn' and 'rand'.
    gain : float
        Optional, default: 1
        Gain to scale initialized weights to.
    device : torch.device
        Optional, default: None, in which case it will be set to
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        The device to which the module is initialized.
    dtype : torch.dtype
        Optional, default: torch.float64
        The data type to which the module is initialized.
    """

    def __init__(self, tau_in, tau_out, real=False, weight_init='randn', gain=1, device=None, dtype=torch.float64):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        super().__init__(device=device, dtype=dtype)

        # input multiplicity
        tau_in = GTau(tau_in)
        # Remove irreps with zero multiplicity
        tau_in = {key: val for key, val in tau_in.items() if val}

        # output multiplicity
        tau_out = GTau(tau_out) if type(tau_out) is not int else tau_out
        if type(tau_out) is int:
            tau_out = {key: tau_out for key, val in tau_in.items() if val}

        self.tau_in = tau_in
        self.tau_out = GTau(tau_out)
        self.real = real
        self.weight_init = weight_init

        # weights
        if self.weight_init == 'randn':
            weights = GWeight.randn(self.tau_in, self.tau_out, device=self.device, dtype=self.dtype)
        elif self.weight_init == 'rand':
            weights = GWeight.rand(self.tau_in, self.tau_out, device=self.device, dtype=self.dtype)
        else:
            raise NotImplementedError(
                f"weight_init can only be 'randn' or 'rand'; other choices are not implemented yet ({self.weight_init})!")

        gain = {key: torch.tensor([gain / max(shape) / (10 ** key[0] if key[0] == key[1] else 1), 0],
                                  device=self.device, dtype=self.dtype).view(2, 1, 1)
                for key, shape in weights.shapes.items()}
        gain = GScalar(gain)

        weights = gain * weights
        self.weights = weights.as_parameter()

    def forward(self, reps):
        """
        The forward function for linearly mixing a represention.

        Parameters
        ----------
        rep : list of `torch.Tensor`
            Representation to mix.

        Returns
        -------
        rep : list of `torch.Tensor`
            Mixed representation.
        """
        if isinstance(reps, dict):
            reps = GVec(reps)

        if not GTau.from_rep(reps) == self.tau_in:
            raise ValueError(f'Tau of input reps, {GTau.from_rep(reps)}, does not match initialized tau, {self.tau_in}!')

        return g_torch.mix(self.weights, reps)

    @property
    def tau(self):
        return self.tau_out


class CatReps(Module):
    """
    Module to concanteate a list of reps. Specify input type for error checking
    and to allow network to fit into main architecture.

    Parameters
    ----------
    taus_in : list of GTau or compatible.
        List of taus of input reps.
    maxdim : int
        Optional, None, in which case it will be set to the max_weight + 1
        Maximum weight to include in the concatenation.
    """

    def __init__(self, taus_in, maxdim=None):
        super().__init__()

        self.taus_in = taus_in = [GTau(tau) for tau in taus_in if tau]

        if maxdim is None:
            maxdim = max(sum(dict(i for tau in taus_in for i in tau.items()), ())) + 1  # max(j) + 1
        self.maxdim = maxdim

        self.taus_in = taus_in
        self.tau_out = {}
        for tau in taus_in:
            for key, val in tau.items():
                if val > 0:
                    if max(key) <= maxdim - 1:
                        self.tau_out.setdefault(key, 0)
                        self.tau_out[key] += val
        self.tau_out = GTau(self.tau_out)

        self.all_keys = list(self.tau_out.keys())

    def forward(self, reps):
        """
        The forward that concatenates a list of reps.

        Parameters
        ----------
        reps : list of GTensor subclasses
            List of representations to concatenate.

        Return
        -------
        reps_cat : list of torch.Tensor
            List of concateated reps
        """
        # Drops None
        reps = [rep for rep in reps if rep is not None]

        # Error checking
        reps_taus_in = [rep.tau for rep in reps]
        if reps_taus_in != self.taus_in:
            raise ValueError(
                f'Taus of input reps {reps_taus_in} do not match the predefined taus_in {self.taus_in}!')

        # Keep reps up to maxdim
        if self.maxdim is not None:
            reps = [rep.truncate(self.maxdim) for rep in reps]

        return g_torch.cat(reps)

    @property
    def tau(self):
        return self.tau_out


class CatMixReps(CGModule):
    """
    Module to concatenate and mix a list of reps CatReps and MixReps.

    Parameters
    ----------
    tau_in : GTau (or compatible object).
        Input tau of representation.
    tau_out : GTau (or compatible object) or int
        Output tau of representation. If type(tau_out) == int,
        the output type will be set to tau_out for each
        parameter in the network.
    maxdim : int
        Optional, None, in which case it will be set to the max_weight + 1
        Maximum weight to include in the concatenation.
    real : bool
        Optional, default: False
        Whether to use purely real mixing weights.
    weight_init : str
        Optional, default: 'randn'
        The type of weight initialization. The choices are 'randn' and 'rand'.
    gain : float
        Optional, default:: 1
        Gain to scale initialized weights to.
    device : torch.device
        Optional, default: None, in which case it will be set to
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        The device to which the module is initialized.
    dtype : torch.dtype
        Optional, default: torch.float64
        The data type to which the module is initialized.

    """

    def __init__(self, taus_in, tau_out, maxdim=None,
                 real=False, weight_init='randn', gain=1, device=None, dtype=torch.float64):

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        super().__init__(device=device, dtype=dtype)

        self.cat_reps = CatReps(taus_in, maxdim=maxdim)
        self.mix_reps = MixReps(self.cat_reps.tau, tau_out,
                                real=real, weight_init=weight_init, gain=gain,
                                device=device, dtype=dtype)

        self.taus_in = taus_in
        self.taus_out = GTau(self.mix_reps.tau)

    def forward(self, reps_in):
        """
        The forward function for concatenating and then
        linearly mixing a list of reps.

        Parameters
        ----------
        reps_in : list of torch.Tensor
            List of input representations.

        Returns
        -------
        reps_out : list of torch.Tensor
            Representation as a result of combining and mixing input reps.
        """

        reps_cat = self.cat_reps(reps_in)
        reps_out = self.mix_reps(reps_cat)
        return reps_out

    @property
    def tau(self):
        return self.tau_out
