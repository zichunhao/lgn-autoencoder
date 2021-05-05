# Models
## Descriptions
The files in this directory define the architecture of the Lorentz Group Network (LGN) Autoencoder. The encoder is defined as `LGNEncoder` in `encoder.py`, and the decoder as `LGNDecoder` in `decoder.py`. In `/alternatives` is an alternative implementation `LGNEncoder` based on a general LGN graph neural network architecture. However, since the decoder is quite different from the encoder, there is no point in building a general structure. Still, it can be used as a Lorentz group equivariant graph neural network, so it is kept in an isolated directory.

## Main Reference
The neural network architectures in `lgn_cg.py` and `lgn_levels.py` were cloned from [The Lorentz Group Network](https://github.com/fizisist/LorentzGroupNetwork) introduced by Bogatskiy et al. in [arXiv:2006.04780](https://arxiv.org/abs/2006.04780) with minor adaptations.
