import torch
from torch import nn

from lgn.cg_lib import CGDict


class CGModule(nn.Module):
    """
    Clebsch-Gordan module. This functions identically to a normal PyTorch
    nn.Module, except for it adds the ability to specify a
    Clebsch-Gordan dictionary, and has additional tracking behavior set up
    to allow the CG Dictionary to be compatible with the DataParallel module.

    If `cg_dict` is specified upon instantiation, then the specified
    `cg_dict` is set as the Clebsch-Gordan dictionary for the CG module.

    If `cg_dict` is not specified, and `maxdim` is specified, then CGModule
    will attempt to set the local `cg_dict` based upon the global
    `lgn.cg_lib.global_cg_dicts`. If the dictionary has not been initialized
    with the appropriate `dtype`, `device`, and `maxdim`, it will be initialized
    and stored in the `global_cg_dicts`, and then set to the local `cg_dict`.

    In this way, if there are many modules that need `CGDicts`, only a single
    `CGDict` will be initialized and automatically set up.

    Parameters
    ----------
    cg_dict : `CGDict`, optional
        Specify an input CGDict to use for Clebsch-Gordan operations.
    maxdim : int, optional
        Maximum weight to initialize the Clebsch-Gordan dictionary.
    device : `torch.torch.device`, optional
        Device to initialize the module and Clebsch-Gordan dictionary to.
    dtype : `torch.torch.dtype`, optional
        Data type to initialize the module and Clebsch-Gordan dictionary to.
    """

    def __init__(
        self, cg_dict=None, maxdim=None, device=None, dtype=None, *args, **kwargs
    ):
        self._init_device_dtype(device, dtype)
        self._init_cg_dict(cg_dict, maxdim)

        super().__init__(*args, **kwargs)

    def _init_device_dtype(self, device, dtype):
        """
        Initialize the default device and data type.

        device : `torch.torch.device`, optional
            Set device for CGDict and related. If unset defaults to torch.device('cpu').

        dtype : `torch.torch.dtype`, optional
            Set device for CGDict and related. If unset defaults to torch.float64.

        """
        if device is None:
            self._device = torch.device("cpu")
        else:
            self._device = device

        if dtype is None:
            self._dtype = torch.float64
        else:
            if not (
                dtype == torch.half or dtype == torch.float64 or dtype == torch.double
            ):
                raise ValueError(
                    "CG Module only takes internal data types of half/float/double. Got: {}".format(
                        dtype
                    )
                )
            self._dtype = dtype

    def _init_cg_dict(self, cg_dict, maxdim):
        """
        Initialize the Clebsch-Gordan dictionary.

        If cg_dict is set, check the following::
        - The dtype of cg_dict matches with self.
        - The devices of cg_dict matches with self.
        - The desired :maxdim: <= :cg_dict.maxdim: so that the CGDict will contain
            all necessary coefficients

        If :cg_dict: is not set, but :maxdim: is set, get the cg_dict from a
        dict of global CGDict() objects.
        """
        # If cg_dict is defined, check it has the right properties
        if cg_dict is not None:
            if cg_dict.dtype != self.dtype:
                raise ValueError(
                    f"CGDict dtype ({cg_dict.dtype}) not match CGModule() device ({self.dtype})"
                )

            if cg_dict.device != self.device:
                raise ValueError(
                    f"CGDict device ({cg_dict.device}) not match CGModule() device ({self.device})"
                )

            if maxdim is None:
                Warning(
                    "maxdim is not defined, setting maxdim based upon CGDict maxdim ({}!".format(
                        cg_dict.maxdim
                    )
                )

            elif maxdim > cg_dict.maxdim:
                Warning(
                    "CGDict maxdim ({}) is smaller than CGModule() maxdim ({}). Updating!".format(
                        cg_dict.maxdim, maxdim
                    )
                )
                cg_dict.update_maxdim(maxdim)

            self.cg_dict = cg_dict
            self._maxdim = maxdim

        # If cg_dict is not defined, but
        elif cg_dict is None and maxdim is not None:
            self.cg_dict = CGDict(maxdim=maxdim, device=self.device, dtype=self.dtype)
            self._maxdim = maxdim

        else:
            self.cg_dict = None
            self._maxdim = None

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    @property
    def maxdim(self):
        return self._maxdim

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)

        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)

        if self.cg_dict is not None:
            self.cg_dict.to(device=device, dtype=dtype)

        if device is not None:
            self._device = device

        if dtype is not None:
            self._dtype = dtype

        return self

    def cuda(self, device=None):
        if device is None:
            device = torch.device("cuda")
        elif device in range(torch.cuda.device_count()):
            device = torch.device("cuda:{}".format(device))
        else:
            ValueError("Incorrect choice of device!")

        super().cuda(device=device)

        if self.cg_dict is not None:
            self.cg_dict.to(device=device)

        self._device = device

        return self

    def cpu(self):
        super().cpu()

        if self.cg_dict is not None:
            self.cg_dict.to(device=torch.device("cpu"))

        self._device = torch.device("cpu")

        return self

    def half(self):
        super().half()

        if self.cg_dict is not None:
            self.cg_dict.to(dtype=torch.half)

        self._dtype = torch.half

        return self

    def float(self):
        super().float()

        if self.cg_dict is not None:
            self.cg_dict.to(dtype=torch.float64)

        self._dtype = torch.float64

        return self

    def double(self):
        super().double()

        if self.cg_dict is not None:
            self.cg_dict.to(dtype=torch.double)

        self._dtype = torch.double

        return self
