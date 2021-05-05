# Lorentz Group Equivariant Jet Autoencoder
## Descriptions
This jet data generative model exploits the symmetry of the [Lorentz Group](https://en.wikipedia.org/wiki/Lorentz_group), a [Lie group](https://en.wikipedia.org/wiki/Lie_group) that represents a fundamental symmetry of spacetime and describes the dynamics of relativistic objects such as elementary particles in a particle physics experiment. The model is built using the architecture of the [The Lorentz Group Network](https://github.com/fizisist/LorentzGroupNetwork) introduced by Bogatskiy et al. in [arXiv:2006.04780](https://arxiv.org/abs/2006.04780) (see the `README` file in each directory for more details).

To achieve Lorentz equivariance, the model works on the [irreducible representations](https://en.wikipedia.org/wiki/Irreducible_representation) of the [Lorentz group](https://en.wikipedia.org/wiki/Representation_theory_of_the_Lorentz_group). For instance, Lorentz scalars are (0,0) representations, and 4-vectors such as the particle 4-momenta are (1/2,1/2) representations. Each representation has its transformation rules. That the model is equivariant implies that each parameter in the model will transform according to its corresponding transformation rule if the input undergoes a Lorentz transformation. In this way, the model can always generate data that satisfy the special relativity, and the latent space, since all internal parameters are Lorentz tensors, can possibly be more physically interpretable.


## References
### Relevant Group Equivariant Models
- A. Bogatskiy et al., “Lorentz group equivariant neural network for particle physics”, [arXiv:2006.04780](https://arxiv.org/abs/2006.04780). Repository: [Lorentz Group Network](https://github.com/fizisist/LorentzGroupNetwork).
- F. Marc et al., "A Practical Method for Constructing Equivariant Multilayer Perceptrons for Arbitrary Matrix Groups", [arXiv:2104.09459](https://arxiv.org/abs/2104.09459). Repository: [A Practical Method for Constructing Equivariant Multilayer Perceptrons for Arbitrary Matrix Groups](https://github.com/mfinzi/equivariant-MLP).

### Backgrounds
#### Group theory, the Lorentz group, and group representations
- A. Zee, [“Group Theory in a Nutshell for Physicists”](https://press.princeton.edu/books/hardcover/9780691162690/group-theory-in-a-nutshell-for-physicists). Princeton University Press, 2016. ISBN 9781400881185.
- H. Georgi, [“Lie Algebras In Particle Physics: from Isospin To Unified Theories”](https://www.amazon.com/Lie-Algebras-Particle-Physics-Frontiers/dp/0738202339). CRC Press, 2018. ISBN 9780429978845.
- H. Muller-kirsten and A. Wiedemann, [“Introduction To Supersymmetry”](https://www.worldscientific.com/worldscibooks/10.1142/7594), Chapter 1. World ScientificLecture Notes In Physics. World Scientific Publishing Company, 2nd edition, 2010. ISBN 9789813100961.

#### The deep connection between group theory and particle physics
- M. Schwartz, ["Quantum Field Theory and the Standard Model"](https://www.cambridge.org/us/academic/subjects/physics/theoretical-physics-and-mathematical-physics/quantum-field-theory-and-standard-model). Cambridge University Press, 2013. ISBN 9781107034730.
- S. Weinberg. ["The Quantum Theory of Fields, Volume 1: Foundations"](https://www.cambridge.org/core/books/quantum-theory-of-fields/22986119910BF6A2EFE42684801A3BDF). Cambridge University Press, 1995. DOI: [10.1017/CBO9781139644167](https://doi.org/10.1017/CBO9781139644167).
