import torch
import math
import logging
from typing import List, Union, Tuple

ERROR_MSG = "A physical value of beta must be between -1 and 1 (exclusive)."


def boost(
    beta: float, dir: Union[str, torch.Tensor, Tuple[int], List[int]] = "z"
) -> torch.Tensor:
    """
    Get the boost matrix for a given rapidity :math:`beta = v / c` and axis of boost.

    :param beta: rapidity :math:`beta = v / c`
    :type beta: float
    :param dir: axis of boost, specified as a 3D vector or one of ('x', 'y', 'z'). Default to 'z'.
    :type dir: Union[str, torch.Tensor, Tuple[int], List[int]]
    :raises ValueError: if :math:`beta` is not between -1 and 1 (exclusive).
    :return: boost matrix
    """

    if beta == 0:
        return torch.eye(4)
    if abs(beta) >= 1:
        raise ValueError(ERROR_MSG + f" Found: {beta=}")

    if isinstance(dir, str):
        dir = dir.lower()
        if dir == "x":
            return boost_x(beta)
        if dir == "y":
            return boost_y(beta)
        if dir == "z":
            return boost_z(beta)
        else:
            logging.warning(f"{dir} is not a valid direction. Defaulting to z.")
            return boost_z(beta)
    elif isinstance(dir, (tuple, list)):
        # normalize n
        n = math.sqrt(dir[0] ** 2 + dir[1] ** 2 + dir[2] ** 2)
        nhat = [ni / n for ni in dir]
        beta_x, beta_y, beta_z = beta * nhat[0], beta * nhat[1], beta * nhat[2]
        gamma = 1 / math.sqrt(1 - beta**2)
        return torch.tensor(
            [
                [gamma, -gamma * beta_x, -gamma * beta_y, -gamma * beta_z],
                [
                    -gamma * beta_x,
                    1 + (gamma - 1) * beta_x**2 / beta**2,
                    (gamma - 1) * beta_x * beta_y / beta**2,
                    (gamma - 1) * beta_x * beta_z / beta**2,
                ],
                [
                    -gamma * beta_y,
                    (gamma - 1) * beta_y * beta_x / beta**2,
                    1 + (gamma - 1) * beta_y**2 / beta**2,
                    (gamma - 1) * beta_y * beta_z / beta**2,
                ],
                [
                    -gamma * beta_z,
                    (gamma - 1) * beta_z * beta_x / beta**2,
                    (gamma - 1) * beta_z * beta_y / beta**2,
                    1 + (gamma - 1) * beta_z**2 / beta**2,
                ],
            ]
        )
    elif isinstance(dir, torch.Tensor):
        eps = torch.finfo(dir.dtype).eps
        dir = dir / torch.norm(dir, dim=-1, keepdim=True)
        beta_x, beta_y, beta_z = (beta * dir).unbind(-1)
        # [..., i, j]
        if (abs(beta) >= 1).any():
            raise ValueError(ERROR_MSG + f" Found: {beta=}")

        gamma = 1 / torch.sqrt(1 - beta**2)

        # row 0
        boost_0 = torch.stack(
            [gamma, -gamma * beta_x, -gamma * beta_y, -gamma * beta_z], dim=-1
        )

        # row 1
        boost_1 = torch.stack(
            [
                -gamma * beta_x,
                1 + (gamma - 1) * beta_x**2 / (beta**2 + eps),
                (gamma - 1) * beta_x * beta_y / (beta**2 + eps),
                (gamma - 1) * beta_x * beta_z / (beta**2 + eps),
            ],
            dim=-1,
        )

        # row 2
        boost_2 = torch.stack(
            [
                -gamma * beta_y,
                (gamma - 1) * beta_y * beta_x / (beta**2 + eps),
                1 + (gamma - 1) * beta_y**2 / (beta**2 + eps),
                (gamma - 1) * beta_y * beta_z / (beta**2 + eps),
            ],
            dim=-1,
        )

        # row 3
        boost_3 = torch.stack(
            [
                -gamma * beta_z,
                (gamma - 1) * beta_z * beta_x / (beta**2 + eps),
                (gamma - 1) * beta_z * beta_y / (beta**2 + eps),
                1 + (gamma - 1) * beta_z**2 / (beta**2 + eps),
            ],
            dim=-1,
        )
        return torch.stack([boost_0, boost_1, boost_2, boost_3], dim=-2)

    else:
        logging.warning(f"{dir} is not a valid direction. Defaulting to z.")
        return boost_z(beta)


def boost_x(beta: float) -> torch.Tensor:
    if abs(beta) >= 1:
        raise ValueError(ERROR_MSG + f" Found: {beta=}")
    gamma = 1 / math.sqrt(1 - beta**2)
    return torch.tensor(
        [
            [gamma, -gamma * beta, 0, 0],
            [-gamma * beta, gamma, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


def boost_y(beta: float) -> torch.Tensor:
    if abs(beta) >= 1:
        raise ValueError(ERROR_MSG + f" Found: {beta=}")
    gamma = 1 / math.sqrt(1 - beta**2)
    return torch.tensor(
        [
            [gamma, 0, -gamma * beta, 0],
            [0, 1, 0, 0],
            [-gamma * beta, 0, gamma, 0],
            [0, 0, 0, 1],
        ]
    )


def boost_z(beta: float) -> torch.Tensor:
    if abs(beta) >= 1:
        raise ValueError(f"{ERROR_MSG}: {beta=}")
    gamma = 1 / math.sqrt(1 - beta**2)
    return torch.tensor(
        [
            [gamma, 0, 0, -gamma * beta],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [-gamma * beta, 0, 0, gamma],
        ]
    )
