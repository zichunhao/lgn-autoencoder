from lgn.g_lib import g_tau, g_tensor

GTensor = g_tensor.GTensor
GTau = g_tau.GTau


class GScalar(GTensor):
    """
    Core class for creating and tracking G Scalars that
    are used to part-wise multiply `GVec`.

    At the core of each `GScalar` is a list of `torch.Tensors` with
    shape `(B, C, 2)`, where:

    * `B` is some number of batch dimensions.
    * `C` is the channels/multiplicity (tau) of each irrep.
    * `2` corresponds to the real/imaginary parts of the complex dimension.

    Parameters
    ----------

    data : List of of `torch.Tensor` with appropriate shape
        Input of a G Scalar.
    """

    @property
    def bdim(self):
        return slice(1, -1)

    @property
    def cdim(self):
        return -1

    @property
    def rdim(self):
        return None

    @property
    def zdim(self):
        return 0

    @staticmethod
    def _get_shape(batch, weight, channels):
        return (2,) + tuple(batch) + (channels,)

    def check_data(self, data):
        if any(part.numel() == 0 for part in data.values()):
            raise NotImplementedError(
                "Non-zero parts in GScalars not currrently enabled!"
            )

        shapes = {key: part.shape[self.bdim] for key, part in data.items()}
        if len(set(shapes.values())) > 1:
            raise ValueError("Batch dimensions are not identical!")

        if any(part.shape[self.zdim] != 2 for part in data.values()):
            raise ValueError(
                f"Complex dimension (dim={self.zdim}) of each tensor should have length 2! Found: {shapes[self.zdim]}"
            )
