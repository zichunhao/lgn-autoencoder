# Equivariance Test Results
This directory shows the result of equivariance test (from [`../lgn/models/autotest`](https://github.com/zichunhao/lgn-autoencoder/tree/main/lgn/models/autotest)). The rotation angles range from $0$ to $2\pi$, and the Lorentz factors range from $0$ to $11013.2$. Two results are compared by definition of equivariance:
1. The input is transformed and then fed into the network.
2. The input is fed into the network, and the output is transformed.


The relative deviations for a 4-vector output $p$ is defined as
$$
\delta(\Lambda) = \max_{p \in J} \max_{\mu \in \{0,1,2,3\}} \frac{
    \Lambda^\mu_\nu \mathtt{LGN}(p)_\nu
    - \mathtt{LGN}(\Lambda \cdot p)_\mu
}{\Lambda^\mu_\nu \mathtt{LGN}(p)_\nu},
$$
and that for scalar output $s$ is defined as
The plots show the relative deviations between the two results, defined as
$$
\delta(\Lambda) = \max_{s \in J} \frac{
    \Lambda \cdot \mathtt{LGN}(s)
    - \mathtt{LGN}(\Lambda \cdot s)
}{\Lambda \cdot \mathtt{LGN}(s)},
$$
for scalars, where $J$ stands for a jet, and $\mathtt{LGN}$ is the LGN neural network. Relative deviations are plotted for both generated features and internal node features in the equivariance tests. Because of the sensitivity of the test to numerical precisions, the momenta are converted to PeV.

Permutation invariance test is done by comparing the output of original data and the output of the shuffled data, which are expected to match exactly. Similarly, the relative deviation is defined as
$$
\delta(\Pi) = \max_{p \in J} \max_{\mu \in \{0,1,2,3\}} \frac{
    \Pi \cdot \mathtt{LGN}(p)_\nu
    - \mathtt{LGN}(\Pi \cdot p)_\mu
}{\Pi \cdot \mathtt{LGN}(p)_\nu},
$$
where $\Pi$ is the permutation matrix.

## Summary of results
### Rotation equivariance test
- **Scalars**: The relative deviations of the generated scalars are constantly $0$. The relative deviations of internal scalars all up $\mathcal{O}(10^{-18})$.

- **Four-vectors**: The relative deviations of the generated 4-vectors are within $\mathcal{O}(10^{-31})$, and those of internal 4-vectors are within the order of $\mathcal{O}(10^{-23})$.

### Boost equivariance test
- **Scalars**: The relative deviations of the generated scalars are $0$, and the relative deviations of internal scalars are all within the the order of $\mathcal{O}(10^{-5})$. This is because in the input layer, the masses are nearly $0$, so the result will be large when when a small number is in the denominator.

- **Four-vectors**: Even at $\gamma = 11013.2$, which corresponds to $v = 0.9999999917553856c$, the relative deviations of generated 4-vectors are within the order of $\mathcal{O}(10^{-15})$, and those of the internal features remain within the order of $\mathcal{O}(10^{-7})$.

- **Comment**: Boost is more sensible to floating point precision. As Bogatskiy et al. argue in [arXiv:2006.04780](https://arxiv.org/abs/2006.04780), double precision, which is used in this model (`torch.float64`), has a better performance in equivariance test than single precision (e.g. `torch.float`). The number of basis functions for expressing the spherical Bessel function might also impact the performance. The physically relevant region is $\gamma \in [1, 200]$, during which the model is fully equivariant up to small floating point errors.

It can be concluded that the autoencoder is equivariant with respect to Lorentz transformation within numerical precisions.


### Permutation invariance test
- **scalars**: The relative deviation of the generated scalars is $0$.

- **Four-vectors**: The relative deviation of the generated 4-vectors is on the order of $\mathcal{O}(10^{-13})$ or $\mathcal{O}(10^{-14})$.

It can be concluded that the autoencoder is invariant with respect to particle permutation within a jet within numerical precisions.
