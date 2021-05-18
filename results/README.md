# Results
## Poster
The file `OURS-Poster.pdf` is the poster for the [Online Undergraduate Research Symposium](https://ugresearch.ucsd.edu/conferences/ours/OURS%202021.html) (OURS) held at University of California, San Diego in 2021.
The poster does not include all the results of the model since the autoencoder model is not fully trained yet on the full `hls4ml` LHC jet dataset.

## Equivariance Tests
The directory [`/equivariance_tests`](https://github.com/zichunhao/lgn-autoencoder/tree/main/results/equivariance_tests) includes the result of equivariance test (from [`../lgn/models/autotest`](https://github.com/zichunhao/lgn-autoencoder/tree/main/lgn/models/autotest)). The rotation angles range from `0` to `2pi`, and the Lorentz factors range from `0` to `11013.2`. Two results are compared by definition of equivariance:
1. The input is transformed and then fed into the network.
2. The input is fed into the network, and the output is transformed.

The plots show the relative deviations of the two results for both generated features and internal node features. See the [`README`](https://github.com/zichunhao/lgn-autoencoder/blob/main/results/equivariance_tests/README.md) file in that directory for more details of the equivariance test results.
