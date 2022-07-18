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
        jet_features: bool = False, 
        take_sqrt: bool = True
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
        :param take_sqrt: ether to take the square root of the loss, defaults to True.
            Used only when `self.loss_norm_choice` is 'p3'.
        :type take_sqrt: bool, optional
        :return: chamfer loss.
        :rtype: torch.Tensor
        """

        # if (len(p.shape) != 4) or (p.shape[0] != 2):
        #     raise ValueError(
        #         f'Invalid dimension: {p.shape}. The first argument should be complex generated momenta.'
        #     )
        if self.im:
            # complexify if necessary
            if len(q.shape) == 3:
                p = convert_to_complex(p)
            if len(q.shape) == 3:
                q = convert_to_complex(q)
        else:
            # take the real part if necessary
            if len(p.shape) == 4:
                p = p[0]
            if len(q.shape) == 4:
                p = q[0]

        # standard Euclidean distance
        if ('p3' in self.loss_norm_choice.lower()) and (take_sqrt and not self.im):
            # Take (px, py, pz) from (E, px, py, pz) if necessary
            if p.shape[-1] == 4:
                p3 = p[..., 1:]
            if q.shape[-1] == 4:
                q3 = q[..., 1:]
            dist = cdist(p3, q3, device=self.device)

            # (2, batch_size, num_particles, 3)
            if (len(p3.shape) == 4) and (len(q3.shape) == 4):
                if not self.im:
                    dist = dist[0]
                else:
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

        min_dist_pq = torch.min(dist, dim=-1)
        # Equivalent to permuting the last two axis
        min_dist_qp = torch.min(dist, dim=-2)

        # Adapted from Steven Tsan
        # https://github.com/stsan9/AnomalyDetection4Jets/blob/b31a9a2af927a79093079911070a45f14a833c14/code/loss_util.py#L27-L31
        chamfer_loss = torch.sum((min_dist_pq.values + min_dist_qp.values) / 2)

        if jet_features:
            jet_p = torch.sum(p, dim=-2)
            jet_q = torch.sum(q, dim=-2)
            jet_loss = norm_sq(jet_p - jet_q).sum()
        else:
            jet_loss = 0

        return chamfer_loss + jet_loss
