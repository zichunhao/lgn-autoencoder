import torch
import math
import logging

ERROR_MSG = "A physical value of beta must be between -1 and 1 (exclusive)."


def boost(beta: float, dir: str = "z") -> torch.Tensor:
    '''
    Get the boost matrix for a given rapidity :math:`beta = v / c` and axis of boost.
    
    :param beta: rapidity :math:`beta = v / c`
    :param dir: axis of boost. Default to 'z'.
    :raises ValueError: if :math:`beta` is not between -1 and 1 (exclusive).
    :return: boost matrix
    '''
    dir = dir.lower()
    if dir == 'x':
        return boost_x(beta)
    if dir == 'y':
        return boost_y(beta)
    if dir == 'z':
        return boost_z(beta)
    
    logging.warning(f"{dir} is not a valid direction. Defaulting to z.")
    return boost_z(beta)


def boost_x(beta: float):
    if abs(beta) >= 1:
        raise ValueError(ERROR_MSG + f" Found: {beta=}")
    gamma = 1 / math.sqrt(1 - beta**2)
    return torch.tensor([
        [gamma,       -gamma*beta, 0, 0],
        [-gamma*beta, gamma,       0, 0],
        [0,           0,           1, 0],
        [0,           0,           0, 1]
    ])
    
def boost_y(beta: float):
    if abs(beta) >= 1:
        raise ValueError(ERROR_MSG + f" Found: {beta=}")
    gamma = 1 / math.sqrt(1 - beta**2)
    return torch.tensor([
        [gamma,       0, -gamma*beta, 0],
        [0,           1, 0,           0],
        [-gamma*beta, 0, gamma,       0],
        [0,           0, 0,           1]
    ])
    
    
def boost_z(beta: float):
    if abs(beta) >= 1:
        raise ValueError(f"{ERROR_MSG}: {beta=}")
    gamma = 1 / math.sqrt(1 - beta**2)
    return torch.tensor([
        [gamma,       0, 0, -gamma*beta],
        [0,           1, 0, 0],
        [0,           0, 1, 0],
        [-gamma*beta, 0, 0, gamma]
    ])