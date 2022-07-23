from typing import Optional
import torch
import torch.nn as nn
from .distance_sq import convert_to_complex, pairwise_distance_sq, norm_sq, cdist

SUPPORTED_NORMS = ('real', 'cplx', 'complex', 'canonical', 'p3', 'polar')

class ChamferLoss(nn.Module):

    def __init__(
        self, 
        loss_norm_choice: str = 'p3', 
        im: bool = False, 
        device: Optional[torch.device] = None
    ):
        """Chamfer loss.

        :param loss_norm_choice: he choice to compute the norm squared of 
        the complex 4-vector, defaults to 'p3'
            Options:
            - `'p3'`
                1. Norm of p4 is taken so that it only contains real components.
                Shape: `(OTHER_DIMENSIONS,4)`
                2. The norm of the 3-momenta is computed (in Cartesian metric)
                Shape: `(OTHER_DIMENSIONS)`
            - `'real'`
                1. Norm of p4 is taken so that it only contains real components.
                Shape: `(batch_size, num_particles, 4)`
                2. The Lorentz norm is computed using the Minkowskian metric (+, -, -, -).
                Note that the result can be negative even though it is called the 'norm squared.'
                Shape: `(batch_size, num_particles)`
            - `'cplx'` or '`complex'`
                1. Lorentz norm is computed first using the Minkowskian metric (+, -, -, -).
                Shape: `(2, batch_size, num_particles)`
                2. Norm of the complex number is taken. Note that the result will be non-negative.
                Shape: `(batch_size, num_particles)`
            - `'canonical'`
                1. p is expressed in the basis of spherical harmonics, which is naturally defined in the field of complex numbers.
                2. The norm squared is computed.
                3. The result is converted to real.
        :type loss_norm_choice: str, optional
        :param im: whether the imaginary component is considered, defaults to False.
        :type im: bool, optional
        :param device: the device to use, defaults to None
        :type device: Optional[torch.device], optional
        :raises ValueError: if the loss_norm_choice is not one of ('real', 'cplx', 'complex', 'canonical', 'p3', 'polar').
        """
        super(ChamferLoss, self).__init__()
        if loss_norm_choice.lower() not in SUPPORTED_NORMS:
            raise ValueError(f"loss_norm_choice can only be one of {str(SUPPORTED_NORMS)} "
                             f"Found: {loss_norm_choice}")

        self.loss_norm_choice = loss_norm_choice
        self.im = im
        self.device = device if (device is not None) else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

    def forward(
        self, 
        p: torch.Tensor, 
        q: torch.Tensor, 
        jet_features: bool = False
    ) -> torch.Tensor:
        """The forward pass to compute the chamfer loss of the point-cloud like jets.

        :param p: The generated/reconstructed jets 4-momenta.
            Shape: (2, batch_size, num_particles, 4)
        :type p: torch.Tensor
        :param q: The target jets 4-momenta.
            Shape: (2, batch_size, num_particles, 4) or (batch_size, num_particles, 4)
        :type q: torch.Tensor
        :param jet_features: whether the differences in jet momenta 
        are taken into account, defaults to False
        :type jet_features: bool, optional
        :type take_sqrt: bool, optional
        :raises ValueError: if p or q are not 3- or 4- vectors.
        :raises ValueError: if p or q do not have dimensions (2, batch_size, num_particles, 4) or (batch_size, num_particles, 4)
        :return: chamfer loss between p and q.
        :rtype: torch.Tensor
        """
                
        if len(p.shape) == 4 and p.shape[0] == 2:
            if not self.im:
                p = p[0]
        elif len(p.shape) == 3:
            if self.im:
                p = convert_to_complex(p)
        else:
            raise ValueError(f'Invalid dimension: {p.shape=}.')
        
        if len(q.shape) == 4 and q.shape[0] == 2:
            if not self.im:
                q = q[0]
        elif len(q.shape) == 3:
            if self.im:
                q = convert_to_complex(q)
        else:
            raise ValueError(f'Invalid dimension: {q.shape=}.')

        # standard Euclidean distance
        if ('p3' in self.loss_norm_choice.lower()):
            # Take (px, py, pz) from (E, px, py, pz) if necessary
            if p.shape[-1] == 4:
                p3 = p[..., 1:]
            else:
                raise ValueError(f'p must be 4-vectors. Found: {p.shape[-1]=}')
            if q.shape[-1] == 4:
                q3 = q[..., 1:]
            else:
                raise ValueError(f'q must be 4-vectors. Found: {q.shape[-1]=}')
            
            dist = cdist(p3, q3, device=self.device)
            # (2, batch_size, num_particles, 3)
            if (len(p3.shape) == 4) and (len(q3.shape) == 4):
                if not self.im:
                    dist = dist[0]
                else:  # norm
                    dist = torch.sqrt(
                        dist[0]**2 + dist[1]**2 + 1e-16
                    )

        else:  # Other cases
            dist = pairwise_distance_sq(
                p, q,
                norm_choice=self.loss_norm_choice,
                im=self.im,
                device=self.device
            )
            if self.loss_norm_choice == 'p3':  # Euclidean norm
                dist = torch.sqrt(dist + 1e-16)

        # Computer chamfer loss
        min_dist_pq = torch.min(dist, dim=-1).values
        # Equivalent to permuting the last two axis
        min_dist_qp = torch.min(dist, dim=-2).values

        # Adapted from Steven Tsan
        # https://github.com/stsan9/AnomalyDetection4Jets/blob/b31a9a2af927a79093079911070a45f14a833c14/code/loss_util.py#L27-L31
        chamfer_loss = torch.sum((min_dist_pq + min_dist_qp) / 2)

        if jet_features:
            jet_p = torch.sum(p, dim=-2)
            jet_q = torch.sum(q, dim=-2)
            jet_loss = norm_sq(jet_p - jet_q).sum()
            return chamfer_loss + jet_loss
        else:
            return chamfer_loss
