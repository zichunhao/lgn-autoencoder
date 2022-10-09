import logging
from typing import Tuple
import torch

EPS = 1e-16
METHODS = ('component_max', 'overall_max', 'jet_E')


def normalize_p4(
    p4: torch.Tensor, 
    method: str = 'normalize_p4_overall_max'
) -> Tuple[torch.Tensor, torch.Tensor]:
    if method.lower().replace(' ', '_').replace('-', '_') == 'component_max':
        return normalize_p4_component_max(p4)
    elif method.lower().replace(' ', '_').replace('-', '_') == 'overall_max':
        return normalize_p4_overall_max(p4)
    elif method.lower().replace(' ', '_').replace('-', '_') == 'jet_E'.lower():
        return normalize_p4_jet_E(p4)
    else:
        logging.warning(f'Normalization method {method} not recognized. Using component_max.')
        return normalize_p4_overall_max(p4)

def normalize_p4_component_max(p4: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalization using :math:`p^\mu \to p^\mu \max_{p \in J} (|p^\mu|)`. 
    That is, we normalize each component by the component-wise maximum of the 4-momenta within the jet.

    :param p4: 4-momentum tensor of shape (num_jets, num_particles, 4).
    :type p4: torch.Tensor
    :return: (p4_norm, factor)
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """    
    normalize_p4_factor = torch.abs(p4).amax(dim=-2, keepdim=True) + EPS
    p4 = p4 / normalize_p4_factor
    return p4, normalize_p4_factor

def normalize_p4_overall_max(p4: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalization using :math:`p \to p / \max_{\mu \in \{0,1,2,3\}} \max_{p \in J} (|p^\mu|)`.
    That is, we divide each entry of the 4-momentum by the overall maximum of the 4-momenta within the jet.

    :param p4: 4-momentum tensor of shape (num_jets, num_particles, 4).
    :type p4: torch.Tensor
    :return: (p4_norm, factor)
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """    
    normalize_p4_factor = torch.abs(p4).amax(dim=-1, keepdim=True).amax(dim=-2, keepdim=True) + EPS
    p4 = p4 / normalize_p4_factor
    return p4, normalize_p4_factor

def normalize_p4_jet_E(p4: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalization using the jet energy: :math:`p \to p / (\sum_{p \in J} p^0)`.

    :param p4: 4-momentum tensor of shape (num_jets, num_particles, 4).
    :type p4: torch.Tensor
    :return: (p4_norm, factor)
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    jet_p = p4.sum(dim=-2, keepdim=True)
    normalize_p4_factor = jet_p[..., 0].unsqueeze(-1).unsqueeze(-1) + EPS
    p4 = p4 / normalize_p4_factor
    return p4, normalize_p4_factor
    

