# Equivariance Test Results
This directory shows the result of equivariance test. The rotation angles range from `0` to `2pi`, and the Lorentz factors range from `0` to `11013.2`. Two results are compared by definition of equivariance:
1. The input is transformed and then fed into the network.
2. The input is fed into the network, and the output is transformed.

The plots show the relative deviations between the two results for both generated features and internal node features.

## Summary of results
### Rotation equivariance test
- **Scalars**: The generated scalars are `0` regardless of the rotation angle. The relative deviations of internal scalars all up to the order of `1e-14`.

- **Four-vectors**: The relative deviations of the generated 4-vectors are within the order of `1e-11`, and those of internal 4-vectors are within the order of `1e-10`.

### Boost equivariance test
- **Scalars**: The generated scalars are `0` regardless of the rotation angle. The relative deviations of internal scalars are all within the the order of `1e-7`.

- **Four-vectors**: Even at `gamma = 11013.2`, which corresponds to `v = 0.9999999917553856c`, the relative deviations of generated 4-vectors are within the order of `1e-5`, and those of the internal features remain within the order of `1e-3` (and most layers on `1e-4`).

- **Comment**: Boost is more sensible to floating point precision. As Bogatskiy et al. argue in [arXiv:2006.04780](https://arxiv.org/abs/2006.04780), double precision, which is used in this model (`torch.float64`), has a better performance in equivariance test than single precision (e.g. `torch.float`). The number of basis functions for expressing the spherical Bessel function might also impact the performance.
