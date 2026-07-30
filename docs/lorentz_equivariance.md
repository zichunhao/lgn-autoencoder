# How Lorentz Equivariance Is Computed in the LGN Autoencoder

*A lecture note on the mechanism, with emphasis on the Cartesian ↔ canonical basis change.*

This note explains, step by step, **how** the Lorentz Group Equivariant Autoencoder (LGAE) achieves and computes Lorentz equivariance — from the way a 4-momentum is stored, through the change of basis into the representation-theoretic ("canonical") coordinates, the Wigner-$D$ / Lorentz-$D$ matrices, the Clebsch–Gordan (CG) products, and finally the end-to-end encoder/decoder pipeline (including the arbitrary-irrep latent space) and how equivariance is verified numerically.

All statements are grounded in the source; file references use the form `path:line`. Every matrix and identity below was checked numerically (unitarity, round-trip, `repdot = normsq4`) and the equivariance claims were verified with independent test scripts (see §10).

> **Notation.** A Lorentz transform is written $\Lambda$ (or $g$). A representation acting on a tensor is $D(\Lambda)$ or $\rho(\Lambda)$. "Equivariant" means $O(\rho(\Lambda)x) = \rho'(\Lambda)\,O(x)$: transforming the input and then applying $O$ equals applying $O$ and then transforming the output.

---

## 1. The group, its irreps, and how a representation is stored

### 1.1 The group and the $(k,n)$ labelling

The physical symmetry is the proper orthochronous Lorentz group $\mathrm{SO}^+(1,3)$. The network works with its double cover / complexification $\mathrm{SL}(2,\mathbb{C})$, whose Lie algebra factorizes into two commuting $\mathfrak{su}(2)$'s:

$$\mathfrak{so}(1,3)_{\mathbb{C}} \;\cong\; \mathfrak{su}(2)_L \oplus \mathfrak{su}(2)_R .$$

Every finite-dimensional irreducible representation (irrep) is therefore a pair of $SU(2)$ spins $(j_L, j_R)$. In the code an irrep is keyed by an **integer pair `(k, n)`** with

$$(j_L, j_R) = (k/2,\; n/2), \qquad \dim(k,n) = (k+1)(n+1).$$

Two irreps carry all the physics:

| irrep | $(j_L,j_R)$ | dim | object |
|---|---|---|---|
| $(0,0)$ | $(0,0)$ | 1 | Lorentz **scalar** |
| $(1,1)$ | $(\tfrac12,\tfrac12)$ | 4 | **four-vector** (e.g. a 4-momentum) |

Higher irreps — $(2,2)$, $(0,2)$, $(2,0)$, $(1,3)$, … — are genuine Lorentz tensors that appear inside the CG message passing and, with this codebase's extension, may also populate the latent space (§9). Each irrep vector is internally **graded by the rotation-subgroup spin $l$**: it is the concatenation over $l = \tfrac{|k-n|}{2}, \dots, \tfrac{k+n}{2}$ of $SU(2)$ sub-blocks of length $2l+1$. This grading is the whole reason the "canonical basis" below looks the way it does.

### 1.2 The `GVec` / `GTau` data structure

A representation is a Python `dict` keyed by irrep type, `GVec = {(k,n): tensor}`. Each per-irrep tensor has shape (`lgn/g_lib/g_vec.py:46-48`):

$$\big(\underbrace{2}_{\texttt{zdim}=0},\ \underbrace{*\,\text{batch}}_{\texttt{bdim}=1:-2},\ \underbrace{\tau}_{\texttt{cdim}=-2},\ \underbrace{(k{+}1)(n{+}1)}_{\texttt{rdim}=-1}\big).$$

- **`zdim` (axis 0, size 2):** the real/imaginary split of a complex number.
- **`cdim` = $\tau$:** the **multiplicity / channel** axis — how many copies of this irrep the layer carries.
- **`rdim`:** the **irrep-component** axis of length $(k{+}1)(n{+}1)$.

`GTau = {(k,n): int}` is just the "type signature" (multiplicities only).

> **This axis split is the key to everything that follows.** Equivariance will amount to a single structural fact: **the group acts only on `rdim`** (the irrep axis), while **learnable linear maps act only on `cdim`** (the multiplicity axis). Because these are different tensor factors, group action and learning commute.

*(A stale docstring in `g_vec.py:15-22` advertises a different shape `(B, C, 2l+1, 2)` with the complex axis last — trust the code, not that docstring.)*

---

## 2. The two bases and the change-of-basis matrix *(core of this note)*

Everything the user asked about — "how the basis transformation is done" — lives in `lgn/cg_lib/zonal_functions.py`.

### 2.1 What the two bases are

- **Cartesian basis.** A real Minkowski four-vector $p = (t,x,y,z) = (p^0,p^1,p^2,p^3)$, a plain tensor with last dim $=4$. This is how input and output momenta live.
- **Canonical basis.** The same four-vector expressed as the $(1,1)$ irrep: a **complex** tensor whose components are the $SO(3)$-adapted *spherical* components. **All CG products and all Wigner-$D$/Lorentz-$D$ matrices act only in this basis** — that is *why* the change of basis exists.

Concretely the canonical basis for $(1,1)$ is the $(l,m)$ basis with $l\in\{0,1\}$, ordered

$$\big[\,(l{=}0,m{=}0),\ (1,-1),\ (1,0),\ (1,+1)\,\big].$$

### 2.2 The change-of-basis matrix $U = A + iB$

The real tensor `cartesian4` (used in `p_to_rep` at `:266-283` and `p_cplx_to_rep` at `:314-331`) is a pair of $4\times4$ real blocks $A = $ `cartesian4[0]`, $B = $ `cartesian4[1]` (columns ordered $(t,x,y,z)$):

$$
A = \begin{pmatrix} 1&0&0&0\\ 0&\tfrac{1}{\sqrt2}&0&0\\ 0&0&0&1\\ 0&-\tfrac{1}{\sqrt2}&0&0 \end{pmatrix},
\qquad
B = \begin{pmatrix} 0&0&0&0\\ 0&0&-\tfrac{1}{\sqrt2}&0\\ 0&0&0&0\\ 0&0&-\tfrac{1}{\sqrt2}&0 \end{pmatrix}.
$$

They assemble into a single **unitary** complex matrix

$$
\boxed{\;U = A + iB = \begin{pmatrix} 1&0&0&0\\[2pt] 0&\tfrac{1}{\sqrt2}&-\tfrac{i}{\sqrt2}&0\\[2pt] 0&0&0&1\\[2pt] 0&-\tfrac{1}{\sqrt2}&-\tfrac{i}{\sqrt2}&0 \end{pmatrix}\;}
\qquad U U^\dagger = \mathbb{1}.
$$

The $1/\sqrt2$ factors are exactly what make the spherical combinations $(x\mp iy)/\sqrt2$ unit-norm, i.e. what makes $U$ unitary.

### 2.3 Forward map, Cartesian → canonical

`p_to_rep` (real input, `:251-289`) computes $Ap$ and $Bp$ and stacks them as (real, imag) of $Up$. Writing it out reveals the **physical content** — the standard spherical components of a 4-vector:

$$
U\begin{pmatrix} t\\ x\\ y\\ z \end{pmatrix}
= \begin{pmatrix} t\\[2pt] \tfrac{x-iy}{\sqrt2}\\[4pt] z\\[2pt] -\tfrac{x+iy}{\sqrt2} \end{pmatrix}
= \begin{pmatrix} \phi_{0,0}\\ V_{1,-1}\\ V_{1,0}\\ V_{1,+1} \end{pmatrix}.
$$

Component 0 is the $l{=}0$ scalar (the time/energy $t$); components 1–3 are the spherical-harmonic components of the spatial vector. `p_cplx_to_rep` (`:292-341`) is the same $U$ but for an input that already carries a complex axis, computing the full complex product $\mathrm{rep} = (A+iB)(p_0 + i p_1)$.

### 2.4 Inverse map, canonical → Cartesian: sign-flip + transpose = $U^\dagger$

`rep_to_p` (`:344-381`) applies the inverse, which for a unitary $U$ is the conjugate transpose $U^\dagger$. The code builds it from a *different* real tensor `cartesian4H` whose imaginary block has its **sign flipped** (`+1/√2` instead of `-1/√2`) and is then **transposed** by `.permute(0,2,1)`:

$$\texttt{cartesian4H[0]} = A^\top, \qquad \texttt{cartesian4H[1]} = -B^\top \;\Longrightarrow\; p = (A^\top - iB^\top)\,\mathrm{rep} = U^\dagger\,\mathrm{rep}.$$

**This is the crux to remember:** the forward map uses $U = A + iB$; the inverse builds $U^\dagger = A^\top - iB^\top$ by (i) flipping the sign of the imaginary block and (ii) transposing. Those two operations together are exactly Hermitian conjugation. Round-trip `rep_to_p(p_to_rep(p))` recovers $p$ to $\sim 10^{-17}$.

**Where each entry point is used:**
- `p_to_rep` — inside the Cartesian zonal functions (§6);
- `rep_to_p` — encoder latent $(1,1)$ output (`lgn_encoder.py:365`) and decoder final output (`lgn_decoder.py:334`), converting back to Cartesian;
- `p_cplx_to_rep` — the decoder re-canonicalizes the latent 4-vector positions before the CG stack (`lgn_decoder.py:397`).

### 2.5 Lorentz norms in the two bases, and why they agree

- **Cartesian** (`normsq4`, `:201-218`): $\;\lVert p\rVert^2 = 2p_0^2 - \sum_\mu p_\mu^2 = t^2 - x^2 - y^2 - z^2$ (signature $+{-}{-}{-}$).
- **Canonical** (`metric(k,n)`, `:396-407`): the Minkowski metric on an irrep is the matrix

$$g_{(l,m),(l',m')} = (-1)^{\,l+m}\,\delta_{l,l'}\,\delta_{m+m',0},$$

block-diagonal in $l$, anti-diagonal $m \leftrightarrow -m$ within each block. `repdot` (`:410-438`) is the complex **symmetric** bilinear form $v^\top g\,w$ (no conjugation), which is invariant under the canonical-basis $D$-matrices.

The two bases are consistent precisely because $U$ intertwines the metrics:

$$\texttt{repdot}(Up, Up) = (Up)^\top g\,(Up) = p^\top (U^\top g\,U)\,p = \eta(p,p), \qquad U^\top g\,U = \mathrm{diag}(1,-1,-1,-1).$$

Verified numerically: for $p=(1.3,0.5,-0.7,0.2)$, both `normsq4(p)` and `repdot(p_to_rep(p), p_to_rep(p))` give $0.91$ (imaginary part $\sim 0$).

### 2.6 Why the change of basis is required

- CG products and the $D$-matrices are **block-diagonal in the $(k,n)$ index only in the canonical basis**; in Cartesian coordinates that structure is hidden.
- The canonical basis **diagonalizes the $SO(3)$ rotation subgroup**: each irrep is a clean $\bigoplus_l D^l$, rotations act by the standard Wigner-$D^l$, and $m$ is a good quantum number.
- **Because $U$ is unitary, the basis change alters only the coordinatization of the four-vector, never its transformation law.** Momenta enter in Cartesian, are rotated into canonical for all the equivariant work, and are mapped back only when a Cartesian output is wanted.

---

## 3. The Lorentz $D$-matrices

File: `lgn/g_lib/rotations.py`. These are the matrices $D_{(k,n)}(\Lambda)$ that implement $\rho(\Lambda)$ on a canonical-basis irrep.

### 3.1 From $SU(2)$ Wigner-$D$ to the Lorentz-$D$

The $SU(2)$ Wigner-$D$ matrix is built (ZYZ Euler angles) from the small-$d$ matrix, itself obtained by diagonalizing $J_y$:

$$d^{j}(\beta) = e^{\,i\beta J_y}, \qquad D^{j}_{m'm}(\alpha,\beta,\gamma) = e^{\,i\alpha m'}\,d^{j}_{m'm}(\beta)\,e^{\,i\gamma m}.$$

The **Lorentz-$D$** matrix for irrep $(k,n)$ is a tensor product of a left and a (conjugated) right $SU(2)$ factor, conjugated by an intertwiner $C_{(k,n)}$ (`LorentzD`, `:119-156`):

$$
D_{(k,n)}(\alpha,\beta,\gamma) = C_{(k,n)}\Big[\, D^{(k/2)}(\alpha,\beta,\gamma)\;\otimes\;\overline{D^{(n/2)}(-\alpha,\beta,-\gamma)}\,\Big] C_{(k,n)}^{\top},
\qquad C_{(k,n)} = \texttt{cg\_dict}[((k,0),(0,n))][(k,n)].
$$

### 3.2 The intertwiner $C$ — the *second* basis change

Since $(k,0)\otimes(0,n)$ contains exactly one copy of $(k,n)$, the matrix $C_{(k,n)}$ is **square and orthogonal** ($C^\top = C^{-1}$): a **pure change of basis**, not a decomposition. It maps the *product basis* $|m_L\rangle\otimes|m_R\rangle$ (in which $D^{(k/2)}\otimes\overline{D^{(n/2)}}$ is naturally written) into the *$l$-graded storage basis* $\bigoplus_l |l,m\rangle$ in which every `GVec` actually lives. Without $C$, the $D$-matrix would act in the wrong basis relative to the stored data.

### 3.3 Rotations (real angles) vs boosts (imaginary angles)

- **Real** Euler angles ⇒ the phases $e^{i\alpha m'}$ are unimodular ⇒ a **compact spatial rotation**; the left and right factors act identically (the diagonal $SU(2)$). All entries are bounded by $1$.
- **Imaginary** angle $\theta \to i\alpha$: $e^{i(i\alpha)m'} = e^{-\alpha m'}$ and $d^j(i\alpha) = e^{-\alpha J_y}$ become **hyperbolic** ($\cosh/\sinh$) ⇒ a non-compact **boost**, with $\gamma = \cosh\alpha$. The opposite signs of $\alpha,\gamma$ in the right factor $\overline{D^{(n/2)}(-\alpha,\beta,-\gamma)}$ are what turn a rotation into a boost. **The exponentially large entries of a boost $D$ are the origin of the numerical caveats in §11.**

### 3.4 Applying $D$ to a `GVec`

`rotate_part` / `rotate_rep` (`:7-51`) apply $D_{(k,n)}$ **contracting the `rdim` (irrep) axis only**. The complex `side="left"` convention actually right-multiplies by the complex conjugate $\overline{D}$; the naming is loose, but the load-bearing fact for equivariance is simply: **$D$ touches `rdim`, nothing else.**

---

## 4. The three equivariant building blocks

The entire network is assembled from exactly three kinds of operation, each provably equivariant, glued together by the unitary basis change of §2 and the orthogonal intertwiner of §3.2.

### 4.1 Per-irrep linear mixing (`MixReps`, `g_torch.mix`)

A learnable layer applies **one matrix $W_{(k,n)}$ per irrep type**, contracting the multiplicity axis `cdim`, identically for every component $m$:

$$(WX)_{\tau',m} = \sum_\tau W_{\tau'\tau}\,X_{\tau,m}.$$

**Why it is equivariant — the two-axes argument.** The group acts on `rdim` (right-multiply by $\overline{D}$); mixing acts on `cdim` (left-multiply by $W$). Different tensor factors commute:

$$W\,(X\overline{D}) = (WX)\,\overline{D} \quad\Longrightarrow\quad M_W(\rho(\Lambda)X) = \rho(\Lambda)\,M_W(X).$$

This is Schur's lemma in disguise: $W$ is a multiplicity-space endomorphism ($\propto \mathbb{1}$ on the irrep axis), $D$ is $\propto \mathbb{1}$ on the multiplicity axis. Two structural consequences are enforced in code:

- mixing is **block-diagonal in irrep type** (no term ever maps one $(k,n)$ into a different one — Schur forbids it);
- every `GWeight` factory asserts **`tau_out.keys() ⊆ tau_in.keys()`** (`g_weight.py:86-88`): a layer may *drop* irrep types but never *create* them, and it can never *merge* different weights. (This is exactly why, in this codebase's latent extension, higher irreps must be injected as separate node features in the decoder rather than "mixed down" — see §9.)

### 4.2 Clebsch–Gordan products — the unique equivariant bilinear map

`cg_product` (`cg_ops.py:135-218`) is the only operation that combines *different* components of covariant tensors (the network's genuine nonlinearity on non-scalars). It maps two `GVec`s to a new one via the Lorentz CG selection rule (two independent $SU(2)$ triangle rules, one per chiral factor):

$$
(k_1,n_1)\otimes(k_2,n_2) = \bigoplus_{\substack{k=|k_1-k_2| \\ \text{step }2}}^{k_1+k_2}\ \bigoplus_{\substack{n=|n_1-n_2| \\ \text{step }2}}^{n_1+n_2} (k,n).
$$

Let $H = \texttt{cg\_dict}[((k_1,n_1),(k_2,n_2))]$ be the (orthogonal) CG matrix. **Equivariance is the intertwiner identity**

$$
\boxed{\;H\,\big[\,D_{(k_1,n_1)}(\Lambda)\otimes D_{(k_2,n_2)}(\Lambda)\,\big] = \Big[\bigoplus_{(k,n)} D_{(k,n)}(\Lambda)\Big]\,H\;}
$$

Since $(D_1 x)\otimes(D_2 y) = (D_1\otimes D_2)(x\otimes y)$, this gives directly

$$\mathrm{CGProduct}(D(\Lambda)x,\,D(\Lambda)y) = \bigoplus_{(k,n)} D_{(k,n)}(\Lambda)\,\big[\mathrm{CGProduct}(x,y)\big]_{(k,n)} .$$

That is: the CG product of two covariant inputs is again covariant. (Internally the Lorentz CG matrix factorizes into two $SU(2)$ CG matrices, `cg_dict.py:253-281`, which is *why* the selection rule factorizes into a left and a right triangle rule.)

### 4.3 Free-form nonlinearity, confined to invariants

Arbitrary (non-equivariant) functions are allowed **only on Lorentz scalars**, because a function of an invariant is automatically invariant. Two places use this:

- **`CGMLP`** (`lgn_levels.py:124-227`) pops the $(0,0)$ scalars, runs a standard `Linear`+activation MLP on them, and re-inserts — it never touches a covariant component.
- **Radial filters** (`RadPolyTrig`, `position_levels.py`) are learnable functions of the **invariant** edge norms $\lVert p_{ij}\rVert^2$ only.

---

## 5. Zonal functions — a tower of Lorentz spherical harmonics

The edge and node geometric features are built (`zonal_functions.py`) by iterated CG products projected onto the "stretched" $(l,l)$ component:

$$
\mathrm{zf}_{(0,0)} = \mathbf{1}, \quad \mathrm{zf}_{(1,1)} = p, \quad
\boxed{\;\mathrm{zf}_{(l,l)} = \sqrt{\tfrac{2l}{l+1}}\,\big[\,\mathrm{zf}_{(l-1,l-1)} \otimes_{\mathrm{CG}} p\,\big]_{(l,l)}\;}
$$

so $\mathrm{zf}_{(l,l)}(p) \propto [\,p^{\otimes l}\,]_{(l,l)}$ — the top-irrep component of the $l$-fold tensor power, the Lorentz analogue of the zonal spherical harmonic $Y_l(\hat p)$. Because `cg_product` is an intertwiner (§4.2), each is covariant: $\mathrm{zf}_{(l,l)}(\Lambda p) = D_{(l,l)}(\Lambda)\,\mathrm{zf}_{(l,l)}(p)$. The edge kernel is then

$$\texttt{edge} = \underbrace{\texttt{rad\_funcs}}_{\text{invariant (radial)}} \times \underbrace{\texttt{zonal\_functions}}_{\text{covariant (angular)}},$$

an invariant-radial $\times$ covariant-angular product — the direct Lorentz analogue of Tensor-Field-Network / SE(3)-Transformer message passing. Relative momenta $p_{ij} = p_i - p_j$ feed the same tower, and their invariant norms $\lVert p_{ij}\rVert^2$ feed the radial filters.

---

## 6. The end-to-end pipeline

### 6.1 Encoder (`lgn_encoder.py`)

1. **Input.** Cartesian momenta $p_i$; the only genuinely new scalars are the per-particle invariant masses $s_i = \sqrt{|p_i^2|}$ (`normsq4`), manifestly Lorentz-invariant.
2. **Zonal features.** Single-particle zonal harmonics (Cartesian→canonical internally via `p_to_rep`), with the $(0,0)$ slot overwritten by $s_i$; plus relative zonal functions on all $p_{ij}$ and their invariant norms.
3. **Radial filters** on the invariant norms → equivariant scalar edge weights.
4. **Input `MixReps`** → CG **message passing** (§6.3).
5. **Final `MixReps`** to the requested latent multiplicities.
6. **Basis split before pooling (`:362-366`):** *only* the $(1,1)$ latent is converted to Cartesian via `rep_to_p`; **every other latent irrep stays in the canonical basis** (it has no Cartesian counterpart).
7. **Aggregate over particles** → the latent (§6.4).

### 6.2 Decoder (`lgn_decoder.py`)

The latent is expanded (`latent_to_graph`) into $N$ per-particle seeds — the multiplicity axis is reinterpreted as the particle axis. The $(1,1)$ latent (Cartesian from step 6) is converted **back to canonical** via `p_cplx_to_rep` to seed particle positions; the $(0,0)$ latent gives scalars; and **each extra latent irrep is injected as an additional covariant CG-seed node feature** (in the canonical basis) — the decoder-side counterpart of the arbitrary-irrep latent. The same LGN-CG stack runs, then `MixReps` → `rep_to_p` back to Cartesian, one four-vector per output particle.

### 6.3 CG message passing (`lgn_cg.py`, `lgn_levels.py`)

Each `LGNNodeLevel` combines three equivariant operations:

1. **`cg_power` $=$ `CGProduct(node, node)`** — self-interaction;
2. **`cg_aggregate` $=$ `CGProduct(node, edge, aggregate=True)`** — mutual interaction, summed over neighbours (an equivariant sum);
3. **`CatMixReps`** of the results → new channels.

The CG product is the sole nonlinearity on non-scalar reps and is equivariant by construction; the optional `CGMLP` acts on $(0,0)$ scalars only.

### 6.4 Aggregation / pooling — including the arbitrary-irrep latent

The pooling over particles must itself be equivariant:

- **`sum` / `mean`** are linear and identical on every component ⇒ manifestly equivariant. **`mix`** folds pooling into the multiplicity axis and mixes ⇒ equivariant. **`+` / `&`** average / concatenate the above.
- **`min` / `max`** select a single particle per channel by ranking on a **Lorentz invariant**, then gather the covariant feature:
  - for $(0,0),(1,1)$: rank by $\texttt{get\_msq} = E^2 - |\vec p|^2$;
  - for any higher irrep: rank by $\mathrm{rank}_i = \Re\big[\texttt{repdot}(f_i, f_i)\big]$, broadcast across the complex axis so the *same* particle is chosen for real and imaginary parts.

  **Why this is equivariant:** under $\Lambda$, $f_i \to D(\Lambda)f_i$ but $\mathrm{rank}_i(\Lambda f) = \mathrm{rank}_i(f)$ *exactly*, so $\arg\max_i$ is frame-independent — the **same index** is selected in every frame, and the gathered feature transforms as $D(\Lambda)f_{i^\star}$. (Ranking on covariant components directly would **not** be equivariant.) The numerical fragility of this argument at extreme boosts is §11.2.

**The arbitrary-irrep latent (this codebase's extension).** `tau_latent = {(k,n): mult}` lets the bottleneck carry any irreps the last CG layer produces (`resolve_tau_latent`, `models/utils.py`); a guard rejects any irrep the layer cannot produce (with `maxdim=2` only $(0,0),(1,1)$ exist — $(2,2),(0,2),(2,0)$ require `maxdim ≥ 3`). Higher latent irreps are stored canonically in the encoder and injected as covariant CG seeds in the decoder, so they genuinely participate in reconstruction while preserving equivariance.

### 6.5 The one-sentence summary

Every operation is one of three equivariance-preserving kinds — **Schur-constrained per-irrep mixing**, **CG intertwiner products**, and **free-form nonlinearity confined to invariants** — glued together by the **unitary** Cartesian↔canonical change of basis $U$ (and the **orthogonal** intertwiner $C$). The only quantities that ever feed a nonlinearity or a selection are the invariants (masses $\sqrt{|p_i^2|}$, edge norms $\sqrt{|p_{ij}^2|}$, and the CG contraction structure). Because nothing nonlinear or order-dependent touches a covariant component directly, the map $p^{(4)} \mapsto \text{latent} \mapsto \hat p^{(4)}$ is Lorentz-equivariant.

---

## 7. How equivariance is verified

File: `lgn/models/autotest/`. The test compares the two ways to combine "transform" and "run the network," which must agree for an equivariant $f = \text{decoder}\circ\text{encoder}$:

$$f(\Lambda\cdot x) \;\overset{?}{=}\; \Lambda\cdot f(x).$$

- **LHS ("transform the input").** Apply $\Lambda$ to the input momenta with a **Cartesian $4\times4$ matrix $R$**, then run the network. $R$ is produced by `_gen_rot`, which sandwiches the $(1,1)$ `LorentzD` between `cartesian4` and `cartesian4H` — i.e. between the very $U$ and $U^\dagger$ of §2 — to yield a **real Cartesian Lorentz matrix**, applied as $p'_a = p_b R_{ba}$.
- **RHS ("transform the output").** Run the untouched input, then act on the output via `rotate_rep` (canonical $D$-matrices).

This is exactly why §2 must be nailed down: **the test transforms momenta in Cartesian but the network's irreps in canonical, and the two agree only because $U^\top g\,U = \eta$.** The relative deviation

$$\delta^{(w)} = \frac{\lVert f(\Lambda x) - \Lambda f(x)\rVert}{\lVert \Lambda f(x)\rVert + \epsilon}$$

is measured per irrep, swept over rotation angles $\theta \in [0,2\pi]$ and boost rapidities $\alpha \in [0,10]$ ($\gamma = \cosh\alpha$ up to $\sim 10^4$). In exact arithmetic $\delta = 0$; any nonzero value is pure floating-point error.

---

## 8. What was actually measured (this analysis)

For the arbitrary-irrep latent, I verified not only the reconstruction output but that **each latent irrep transforms under its own $D$-matrix** — the $(1,1)$ latent as a Cartesian 4-vector under $R$, every other latent irrep as a canonical irrep under `rotate_rep`. The harness includes a self-test proving it *detects* non-equivariance (feeding the wrong, identity, expectation yields $\delta \approx 1.1$, versus $\approx 6\times10^{-15}$ for the correct one). Findings were independently reproduced by separate scripts (fresh seeds, independent code paths), which also confirmed $R$ is a genuine Lorentz transform: $\lVert R\,\eta\,R^\top - \eta\rVert_{\max} = 1.2\times10^{-15}$ (rotation), $1.6\times10^{-15}$ (boost).

| Quantity | Rotations $\theta\le 2\pi$ | Boost, physical $\gamma\lesssim 10$ | Boost, extreme $\gamma\approx 1490$ |
|---|---|---|---|
| Per-latent-irrep covariance — incl. new $(2,2),(0,2),(2,0)$ | $\lesssim 3\times10^{-14}$ | $\sim 10^{-14}$ | $\sim 10^{-9}$ |
| End-to-end reconstruction | $\sim 10^{-15}$ | $\sim 10^{-14}$ | $\sim 10^{-9}$ |
| $(0,0)$ scalar invariance | — | $3.4\times10^{-16}$ | — |
| Composition (rot∘boost), permutation-invariance | $\le 1.9\times10^{-15}$ | — | — |

**Conclusion.** The autoencoder — including the new arbitrary-irrep latent — is Lorentz-equivariant to machine precision under rotations, and under boosts throughout the physically relevant regime, with the expected float-precision degradation at extreme boosts (§11.1).

---

## 9. Numerical caveats

### 9.1 Boost float-precision degradation (all aggregations)

The network is *exactly* equivariant in exact arithmetic, so the growth of $\delta$ with $\gamma$ is purely floating-point:

1. A boost $D$ has entries scaling like $\gamma = \cosh\alpha$ and $e^{\pm\alpha}\approx 2\gamma$ (§3.3); at $\alpha=10$, $\gamma\sim 10^4$.
2. In fixed float64 (~15–16 digits), mixing values across many magnitudes causes catastrophic cancellation, so the *relative* error grows roughly exponentially in $\alpha$.
3. The code mitigates by rescaling GeV→TeV before boosting, and plots the $(1,1)$ deviation on a log-$y$ scale.

Rotations use *unitary* Wigner-$D$ (entries $\le 1$), so there is no blow-up — rotation deviation stays at machine epsilon for all $\theta$. Measured `mean`/`sum`/`mix` boost curve: $\delta$ grows smoothly and monotonically from $\sim 10^{-15}$ at $\gamma\approx 1$ to $\sim 6\times10^{-9}$ at $\gamma\approx 1490$.

### 9.2 `min`/`max` ranking fragility for high-weight irreps at extreme boost

The pooling invariant of §6.4 is exact in exact arithmetic, but for a **high-weight** irrep, `repdot` contracts the sign-alternating metric $(-1)^{l+m}$ over a $(k{+}1)(n{+}1)$-vector whose entries grow rapidly under boost — the invariant is a **difference of large, oppositely-signed terms** (catastrophic cancellation). Under an extreme boost, two particles with nearly-equal true invariants can have their *computed* invariants swap order, so `argmax`/`argmin` **reorders** and a *different* particle is gathered — a **discrete, non-smooth** equivariance violation.

Measured for a $(2,2)$ latent irrep with `min&max` aggregation:

- **Rotations and physical boosts ($\gamma \lesssim 10$): never breaks** ($\delta \le 4\times10^{-13}$).
- **Extreme boost ($\gamma \approx 1490$): input-dependent breakage.** Over 80 random inputs, **8 broke** ($\delta$ up to $3.8\times10^{-1}$; an independent run reached $1.09$), while the other 72 stayed $\sim 10^{-9}$. On the breaking inputs, the selected particle index literally flips (e.g. `argmax [7,3] → [7,6]`) because the top-two ranking invariants were near-tied (gap $\sim 2.6\times10^{-3}$) and the $\gamma^2\!\sim\!10^6$ boost round-off reordered them.
- **`mean`/`sum`/`mix` on the identical breaking inputs stay $\sim 10^{-10}$** (they never select a single particle), confirming this is a property of discrete ranking, not of the model or the invariant itself.

**Practical takeaway.** This is a genuine numerical fragility of `argmax`/`argmin` over a near-degenerate high-weight invariant amplified by extreme-boost round-off — **not** a code bug (the ranking is mathematically exactly invariant, which is what makes `min`/`max` equivariant when there is no near-tie). Real jets never reach $\gamma\sim 10^3$, so it does not affect physical use. If exact equivariance at extreme boosts matters for a high-weight latent irrep, prefer `mean`/`sum`/`mix` over `min`/`max`.

---

## 10. Appendix — implementation notes observed during this analysis

These are orthogonal to equivariance but were noticed while tracing the code; each was verified before being stated here.

- **Decoder radial filters are distance-independent.** The decoder builds `edge_mask` as `torch.zeros` (`lgn_decoder.py:_prepare_input`), so inside `RadPolyTrig.forward` the `torch.where(edge_mask.bool(), bell, 0)` zeros the norm-dependent part of every radial filter. The subsequent `nn.Linear`'s **bias** survives, so the radial filters are **nonzero constants** (measured $\approx 0.11\text{–}0.21$), *not* zero. Net effect: the decoder's `cg_aggregate` mutual-interaction branch still runs, but its radial weighting no longer depends on the actual inter-particle distances — only the covariant zonal angular part carries geometry. This is a pre-existing behavior (the encoder, by contrast, builds a genuine `edge_mask` from $p^0\neq 0$). Whether the constant radial profile is intended is a design question for the authors; it does not affect equivariance (radial filters are invariant scalars either way).
- **Stale `GVec` docstring** (`g_vec.py:15-22`) describes a different tensor layout than the code implements — trust the `_get_shape` / `zdim`/`cdim`/`rdim` properties.
- **`sum` aggregation** previously carried a stray extra `.unsqueeze(-3)` that broke its output shape; it has been fixed to match `mean`, and now passes the equivariance checks above.

---

### Source map

| Topic | Files |
|---|---|
| Bases, norms, zonal functions | `lgn/cg_lib/zonal_functions.py` |
| Generators, Wigner-$D$, Lorentz-$D$ | `lgn/g_lib/rotations.py` |
| CG product, intertwiner, selection rule | `lgn/cg_lib/cg_ops.py`, `lgn/cg_lib/cg_dict.py` |
| `GVec`/`GTau`, linear mixing | `lgn/g_lib/g_vec.py`, `g_tau.py`, `g_torch.py`, `cplx_lib.py`, `lgn/nn/g_nn.py` |
| Pipeline | `lgn/models/lgn_encoder.py`, `lgn_decoder.py`, `lgn_cg.py`, `lgn_levels.py`, `models/utils.py`, `lgn/nn/position_levels.py` |
| Verification | `lgn/models/autotest/lgn_tests.py`, `autotest/utils.py` |
