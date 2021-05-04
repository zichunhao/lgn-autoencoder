"""
Adapt variables to a list of length num_cg_levels.
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
        raise ValueError(f'Incorrect type of variables: {type(var)}. ' \
                         'The allowed data types are list, float, or int')
    return var_list
