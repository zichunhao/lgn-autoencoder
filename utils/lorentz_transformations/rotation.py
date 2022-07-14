import torch
import math
import logging

def rot(theta: float, dir: str = "z"):
    dir = dir.lower()
    if dir == 'x':
        return rot_x(theta)
    if dir == 'y':
        return rot_y(theta)
    if dir == 'z':
        return rot_z(theta)
    
    logging.warning(f"{dir} is not a valid direction. Defaulting to z.")
    return rot_z(theta)


def rot_x(theta: float):
    return torch.tensor([
        [1, 0, 0,               0],
        [0, 1, 0,               0],
        [0, 0, math.cos(theta), -math.sin(theta)],
        [0, 0, math.sin(theta), math.cos(theta)]
    ])
    
def rot_y(theta: float):
    return torch.tensor([
        [1, 0,                0, 0],
        [0, math.cos(theta),  0, math.sin(theta)],
        [0, 0,                1, 0],
        [0, -math.sin(theta), 0, math.cos(theta)]
    ])
    
def rot_z(theta: float):
    return torch.tensor([
        [1, 0,               0,                0],
        [0, math.cos(theta), -math.sin(theta), 0],
        [0, math.sin(theta), math.cos(theta),  0],
        [0, 0,               0,                1]
    ])