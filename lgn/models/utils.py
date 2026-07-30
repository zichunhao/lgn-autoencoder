from typing import Dict, Iterable, Optional, Tuple, Union

import torch


def parse_tau_latent_spec(
    tokens: Union[str, Iterable[str]]
) -> Dict[Tuple[int, int], int]:
    """
    Parse a latent-irrep specification into a ``{(k, n): multiplicity}`` dict.

    Each token describes one irrep as ``"k,n:mult"`` (surrounding parentheses and
    whitespace are ignored), e.g. ``"0,0:1"``, ``"(1,1):8"``, ``"2,2:2"``.

    Parameters
    ----------
    tokens : str or iterable of str
        A single token or an iterable of tokens.

    Returns
    -------
    dict of {tuple: int}
        Mapping from irrep weight ``(k, n)`` to its multiplicity. Non-positive
        multiplicities are dropped.
    """
    if isinstance(tokens, str):
        tokens = [tokens]

    tau_latent = {}
    for token in tokens:
        raw = token.strip().replace("(", "").replace(")", "").replace(" ", "")
        if not raw:
            continue
        try:
            weight_str, mult_str = raw.split(":")
            k, n = (int(x) for x in weight_str.split(","))
            mult = int(mult_str)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Could not parse latent-irrep token {token!r}. "
                "Expected the format 'k,n:mult' (e.g. '1,1:8')."
            ) from e
        if mult > 0:
            tau_latent[(k, n)] = tau_latent.get((k, n), 0) + mult
    return tau_latent


def resolve_tau_latent(
    tau_latent: Optional[Union[Dict[Tuple[int, int], int], Iterable[str]]] = None,
    tau_latent_scalars: Optional[int] = None,
    tau_latent_vectors: Optional[int] = None,
) -> Dict[Tuple[int, int], int]:
    """
    Normalize the latent-space irrep multiplicities to a ``{(k, n): multiplicity}`` dict.

    The general ``tau_latent`` argument takes precedence. As a backward-compatible
    fallback, the legacy ``tau_latent_scalars`` (the ``(0, 0)`` irrep) and
    ``tau_latent_vectors`` (the ``(1, 1)`` irrep) are used when ``tau_latent`` is
    not given.

    Parameters
    ----------
    tau_latent : dict of {tuple: int}, iterable of str, or None
        The latent irreps and their multiplicities. Accepts a ready-made dict or a
        list of ``"k,n:mult"`` tokens (see :func:`parse_tau_latent_spec`).
    tau_latent_scalars : int, optional
        Legacy multiplicity of Lorentz scalars ``(0, 0)`` in the latent space.
    tau_latent_vectors : int, optional
        Legacy multiplicity of Lorentz 4-vectors ``(1, 1)`` in the latent space.

    Returns
    -------
    dict of {tuple: int}
        The normalized latent multiplicities (only strictly positive entries).
    """
    if tau_latent is not None:
        if isinstance(tau_latent, dict):
            resolved = {
                tuple(weight): int(mult)
                for weight, mult in tau_latent.items()
                if int(mult) > 0
            }
        else:
            resolved = parse_tau_latent_spec(tau_latent)
    else:
        resolved = {}
        if tau_latent_scalars:
            resolved[(0, 0)] = int(tau_latent_scalars)
        if tau_latent_vectors:
            resolved[(1, 1)] = int(tau_latent_vectors)

    if not resolved:
        raise ValueError(
            "The latent space is empty. Specify at least one irrep with positive "
            "multiplicity via `tau_latent` (or `tau_latent_scalars`/`tau_latent_vectors`)."
        )
    return resolved


def adapt_var_list(var, num_cg_levels):
    """
    Adapt variables to a list of length num_cg_levels.
    - If type(var) is float or int, adapt it to [var] * num_cg_levels.
    - If type(var) is list, adapt the length to num_cg_levels.

    Inputs
    -----
    var : list, int, or float
        The variables
    num_cg_levels : int
        Number of cg levels to use.

    Outputs
    ------
    var_list : list
        The list of variables. The length will be num_cg_levels.
    """
    if type(var) == list:
        if len(var) < num_cg_levels:
            var_list = var + (num_cg_levels - len(var)) * [var[-1]]
        elif len(var) == num_cg_levels:
            var_list = var
        elif len(var) > num_cg_levels:
            var_list = var[: num_cg_levels - 1]
        else:
            raise ValueError(f"Invalid length of var: {len(var)}")
    elif type(var) in [float, int]:
        var_list = [var] * num_cg_levels
    else:
        raise ValueError(
            f"Incorrect type of variables: {type(var)}. "
            "The allowed data types are list, float, or int"
        )
    return var_list


def detectnan(data):
    if isinstance(data, torch.Tensor):
        detectnan_tensor(data)
    if isinstance(data, dict):
        detectnan_dict(data)
    if isinstance(data, list):
        detectnan_list(data)


def detectnan_tensor(data):
    if isinstance(data, torch.Tensor):
        if (data != data).any():
            print(data)
            raise RuntimeError("NaN data!")


def detectnan_dict(data):
    for weight in data.keys():
        if (data[weight] != data[weight]).any():
            print(f"key = {weight}")
            print(data[weight])
            raise RuntimeError("NaN data!")


def detectnan_list(data):
    for i in range(len(data)):
        if isinstance(data[i], dict):
            d = data[i]
            detectnan_dict(d)
        if isinstance(data[i], torch.Tensor):
            detectnan_tensor(data[i])
        if isinstance(data[i], list):
            detectnan_list(data[i])
