import torch

def adapt_var_list(var, num_cg_levels):
    """
    Adapt variables to a list of length num_cg_levels.
    - If type(var) is `float` or `int`, adapt it to [var] * num_cg_levels.
    - If type(var) is `list`, adapt the length to num_cg_levels.

    Inputs
    -----
    var : `list`, `int`, or `float`
        The variables
    num_cg_levels : `int`
        Number of cg levels to use.

    Outputs
    ------
    var_list : `list`
        The list of variables. The length will be num_cg_levels.
    """
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
        raise ValueError(f'Incorrect type of variables: {type(var)}. ' \
                         'The allowed data types are list, float, or int')
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
            raise RuntimeError('NaN data!')

def detectnan_dict(data):
    for weight in data.keys():
        if (data[weight] != data[weight]).any():
            print(f"key = {weight}")
            print(data[weight])
            raise RuntimeError('NaN data!')

def detectnan_list(data):
    for i in range(len(data)):
        if isinstance(data[i], dict):
            d = data[i]
            detectnan_dict(d)
        if isinstance(data[i], torch.Tensor):
            detectnan_tensor(data[i])
        if isinstance(data[i], list):
            detectnan_list(data[i])
