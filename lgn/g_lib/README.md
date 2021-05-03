# The `GTensor` Library
## Descriptions
The files in this folder define how to represent the Lorentz tensor features, mix them, and keep track of the multiplicity using the classes inherited from `GTensor` (namely `GScalar`, `GVec`, `GTau`, and `GWeight`). Besides, `g_wigner_d.py` and `rotations.py` describe the Lorentz transformation rules of each irreps. One thing to keep in mind is that irreps are mixed weight-wise; in other words, irreps with different weights are **not** mixed.

## Main Reference
The content in this folders are cloned from [The Lorentz Group Network](https://github.com/fizisist/LorentzGroupNetwork) introduced by Bogatskiy et al. in [arXiv:2006.04780](https://arxiv.org/abs/2006.04780) (with only *minor* adaptations).
