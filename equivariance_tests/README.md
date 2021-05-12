# Equivariance Test Results
This directory shows the result of equivariance test. The rotation angles range from `0` to `2pi`, and the Lorentz factors range from `0` to `7382.4`. Two results are compared by definition of equivariance:
1. The input is transformed and then fed into the network.
2. The input is fed into the network, and the output is transformed.

The plots show the relative deviations between the two results for both generated features and internal node features.

## Summary of results
### Rotation equivariance test
- **Scalars**: The generated scalars are `0` regardless of the rotation angle. The relative deviations of internal scalars have a maximum on the order of `1e-10`.

- **Four-vectors**: The relative deviations of the generated 4-vectors are within the order of `1e-11`, and those of internal 4-vectors are within the order of `1e-10`.

### Boost equivariance test
- **Scalars**: The generated scalars are `0` regardless of the rotation angle. The relative deviations of internal scalars have a maximum on the order of `1e-10`.

- **Four-vectors**: Even at `gamma = 7382.4`, which corresponds to `v = 0.9999999817c`, the relative deviations of the internal and generated 4-vectors are within the order of `1e-4`. Boost is more sensible to floating point precision. As Bogatskiy et al. argues in [arXiv:2006.04780](https://arxiv.org/abs/2006.04780), double precision, which is what this project uses (`torch.float64`), has a better performance in equivariance test than single precision (e.g. `torch.float`).
