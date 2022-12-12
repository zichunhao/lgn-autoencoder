import torch
import math
import logging
from typing import List, Union, Tuple


def rot(
    theta: float,
    dir: Union[str, torch.Tensor, Tuple[float, float, float], List[float]] = "z",
) -> torch.Tensor:
    """
    Get the rotation matrix for a given angle and axis of spatial rotation.

    :param theta: angle of rotation
    :type theta: float
    :param dir: axis of rotation, specified as a 3D vector or one of ('x', 'y', 'z'). Default to 'z'.
    :type dir: Union[str, torch.Tensor, Tuple[float, float, float], List[float]]
    :return: rotation matrix
    """
    if isinstance(dir, str):
        dir = dir.lower()
        if dir == "x":
            return rot_x(theta)
        if dir == "y":
            return rot_y(theta)
        if dir == "z":
            return rot_z(theta)

        logging.warning(f"{dir} is not a valid direction. Defaulting to z.")
        return rot_z(theta)

    elif isinstance(dir, (Tuple[float, float, float], list)):
        if len(dir) != 3:
            raise ValueError(f"dir must be a 3D vector. Got: {len(dir)=}")
        # normalize the direction vector
        u = math.sqrt(dir[0] ** 2 + dir[1] ** 2 + dir[2] ** 2)
        if u == 0:
            raise ValueError(f"dir must be a non-zero vector. Got: {dir=}")
        ux, uy, uz = dir[0] / u, dir[1] / u, dir[2] / u

        return torch.tensor(
            [
                [1, 0, 0, 0],
                [
                    0,
                    math.cos(theta) + ux**2 * (1 - math.cos(theta)),
                    ux * uy * (1 - math.cos(theta)) - uz * math.sin(theta),
                    ux * uz * (1 - math.cos(theta)) + uy * math.sin(theta),
                ],
                [
                    0,
                    uy * ux * (1 - math.cos(theta)) + uz * math.sin(theta),
                    math.cos(theta) + uy**2 * (1 - math.cos(theta)),
                    uy * uz * (1 - math.cos(theta)) - ux * math.sin(theta),
                ],
                [
                    0,
                    uz * ux * (1 - math.cos(theta)) - uy * math.sin(theta),
                    uz * uy * (1 - math.cos(theta)) + ux * math.sin(theta),
                    math.cos(theta) + uz**2 * (1 - math.cos(theta)),
                ],
            ]
        )

    elif isinstance(dir, torch.Tensor):
        if dir.shape[-1] != 3:
            raise ValueError(f"dir must be a 3D vector. Got: {dir.shape[-1]=}")
        # normalize the direction vector
        u = dir / torch.norm(dir, dim=-1, keepdim=True)
        if torch.any(torch.isnan(u)):
            raise ValueError(f"dir must be a non-zero vector. Got: {dir=}")
        ux, uy, uz = u.unbind(-1)

        # row 0: (1, 0, 0, 0)
        R0 = torch.stack(
            [
                torch.ones_like(ux),
                torch.zeros_like(ux),
                torch.zeros_like(ux),
                torch.zeros_like(ux),
            ],
            dim=-1,
        )
        # row 1
        R1 = torch.stack(
            [
                torch.zeros_like(ux),
                math.cos(theta) + ux**2 * (1 - math.cos(theta)),
                ux * uy * (1 - math.cos(theta)) - uz * math.sin(theta),
                ux * uz * (1 - math.cos(theta)) + uy * math.sin(theta),
            ],
            dim=-1,
        )
        # row 2
        R2 = torch.stack(
            [
                torch.zeros_like(ux),
                uy * ux * (1 - math.cos(theta)) + uz * math.sin(theta),
                math.cos(theta) + uy**2 * (1 - math.cos(theta)),
                uy * uz * (1 - math.cos(theta)) - ux * math.sin(theta),
            ],
            dim=-1,
        )
        # row 3
        R3 = torch.stack(
            [
                torch.zeros_like(ux),
                uz * ux * (1 - math.cos(theta)) - uy * math.sin(theta),
                uz * uy * (1 - math.cos(theta)) + ux * math.sin(theta),
                math.cos(theta) + uz**2 * (1 - math.cos(theta)),
            ],
            dim=-1,
        )
        return torch.stack([R0, R1, R2, R3], dim=-2)

    else:
        logging.warning(f"{dir} is not a valid direction. Defaulting to z.")
        return rot_z(theta)


def rot_x(theta: float):
    return torch.tensor(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, math.cos(theta), -math.sin(theta)],
            [0, 0, math.sin(theta), math.cos(theta)],
        ]
    )


def rot_y(theta: float):
    return torch.tensor(
        [
            [1, 0, 0, 0],
            [0, math.cos(theta), 0, math.sin(theta)],
            [0, 0, 1, 0],
            [0, -math.sin(theta), 0, math.cos(theta)],
        ]
    )


def rot_z(theta: float):
    return torch.tensor(
        [
            [1, 0, 0, 0],
            [0, math.cos(theta), -math.sin(theta), 0],
            [0, math.sin(theta), math.cos(theta), 0],
            [0, 0, 0, 1],
        ]
    )
