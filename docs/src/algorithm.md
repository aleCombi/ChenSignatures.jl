# How it works

This page explains the two kernels behind [`sig`](@ref) and its gradient: the Horner
update used in the forward pass, and the reversible backward pass.

**Setting.** A path ``x_1, \dots, x_N \in \mathbb{R}^D`` is treated as piecewise linear,
with increments ``z_j = x_{j+1} - x_j``, ``j = 1, \dots, n``, ``n = N - 1``. Signatures live
in the truncated tensor algebra ``T^{(M)}(\mathbb{R}^D) = \bigoplus_{k=0}^M (\mathbb{R}^D)^{\otimes k}``.
``S_k`` denotes the level-``k`` component of ``S``. Level-``k`` coefficients are indexed by
words ``w = w_1 \dots w_k`` over ``\{1, \dots, D\}``, and in memory the last letter varies
fastest.

## The Horner update

The signature of one linear segment with increment ``z`` is the truncated exponential
``\exp(z) = \sum_{l=0}^M z^{\otimes l} / l!``. Chen's identity glues segments together:
``S_j = S_{j-1} \otimes \exp(z_j)``, with ``S_0 = 1``. Level ``k`` of the product is

```math
(S \otimes \exp z)_k = \sum_{i=0}^k S_i \otimes \frac{z^{\otimes (k-i)}}{(k-i)!}
  = S_k + S_{k-1} \otimes z + S_{k-2} \otimes \frac{z \otimes z}{2!} + \dots + \frac{z^{\otimes k}}{k!}.
```

Read this as a polynomial in ``z``, where "multiply by ``z``" always means appending ``z``
on the right. Horner's rule factors out one ``\otimes z`` at a time:

```math
(S \otimes \exp z)_k = S_k + \Big( S_{k-1} + \Big( S_{k-2} + \Big( \cdots
  \Big( S_1 + \frac{z}{k} \Big) \otimes \frac{z}{k-1} \cdots \Big) \otimes \frac{z}{2} \Big) \Big) \otimes z .
```

Following ``S_i`` outwards, it collects the factors
``\frac{z}{k-i} \cdot \frac{z}{k-i-1} \cdots \frac{z}{1} = z^{\otimes (k-i)} / (k-i)!``, as required.
As a recursion, for each ``k = M, \dots, 2``:

```math
v_1 = \frac{z}{k}, \qquad u_i = v_i + S_i, \qquad v_{i+1} = c_i \, u_i \otimes z, \qquad
c_i = \frac{1}{k-i}, \qquad i = 1, \dots, k-1, \qquad S_k \mathrel{+}= v_k ,
```

followed by ``S_1 \mathrel{+}= z``.

- **Cost.** ``v_{i+1}`` has ``D^{i+1}`` entries, so chain ``k`` costs about ``D^k``
  multiply-adds, and a whole step about ``D^M (D/(D-1))^2``: ``O(N D^M)`` for the path.
  Forming each term ``S_i \otimes z^{\otimes(k-i)}`` separately costs about ``k D^k`` per
  level, so Horner saves a factor of about ``M`` and never forms ``\exp(z)``.
- **In place.** Level ``k`` only reads ``S_1, \dots, S_{k-1}``, which must still hold their
  old values. Updating levels from ``k = M`` down to ``1`` guarantees this, so no copy of
  ``S`` is needed.

## The reversible backward pass

The forward pass is a chain of maps ``S_j = \Phi_{z_j}(S_{j-1})`` with
``\Phi_z(S) = S \otimes \exp(z)``, and the loss is ``L(S_n)``. With
``G_j = \partial L / \partial S_j``, the chain rule gives

```math
G_{j-1} = \Big(\frac{\partial \Phi}{\partial S}\Big)^{\!T} G_j, \qquad
\bar z_j = \Big(\frac{\partial \Phi}{\partial z}\Big)^{\!T} G_j ,
```

both evaluated at ``(S_{j-1}, z_j)``. Generic reverse-mode AD stores every intermediate
signature, at a cost of ``O(N D^M)`` memory.

### Rebuild instead of storing

Signatures form a group: ``\exp(z) \otimes \exp(-z) = 1``, also after truncation. Hence

```math
S_{j-1} = S_j \otimes \exp(-z_j) = \Phi_{-z_j}(S_j),
```

the same Horner update with ``-z``. Walking backwards from ``S_n``, each earlier signature
is rebuilt just in time. Only one signature and one cotangent are alive at once, so memory
is ``O(D^M)``, independent of ``N``.

### Transposing the Horner chain

View a level-``(i+1)`` tensor as a ``D^i \times D`` matrix: rows are indexed by the first
``i`` letters of the word, columns by the last letter. A chain step is then an outer
product, ``V_{i+1} = c_i \, u_i \, z^T``, and its transposes are matrix–vector products:

```math
\bar u_i = c_i \, \bar V_{i+1} \, z \quad \text{(to the prefix)}, \qquad
\bar z \mathrel{+}= c_i \, \bar V_{i+1}^T \, u_i \quad \text{(to the increment)}.
```

Since ``u_i = v_i + S_i``, the same ``\bar u_i`` is added to ``G_i`` and passed down as
``\bar v_i``. Chain ``k`` starts from ``\bar V_k = G_k``; its innermost step ``v_1 = z/k``
contributes ``\bar v_1 / k`` to ``\bar z``, and the level-1 update contributes ``G_1``.

For one path step, with ``S = S_{j-1}`` already rebuilt and ``G = G_j``:

1. ``\bar z \leftarrow G_1``.
2. For ``k = 2, \dots, M`` (increasing):
   1. recompute ``u_1, \dots, u_{k-1}`` of chain ``k`` from ``S`` and ``z``;
   2. for ``i = k-1, \dots, 1``: ``\bar u_i = c_i \bar V_{i+1} z``,
      ``\bar z \mathrel{+}= c_i \bar V_{i+1}^T u_i``, ``G_i \mathrel{+}= \bar u_i``,
      ``\bar v_i = \bar u_i``;
   3. ``\bar z \mathrel{+}= \bar v_1 / k``.
3. On exit ``G = G_{j-1}``. Finally ``\bar x_{j+1} \mathrel{+}= \bar z`` and
   ``\bar x_j \mathrel{-}= \bar z``, because ``z_j = x_{j+1} - x_j``.

Chain ``k`` reads ``G_k`` and only writes to levels below ``k``, and level ``k`` receives
contributions only from chains above ``k``. Processing ``k`` in increasing order (the
mirror of the forward pass) therefore lets ``G`` be updated in place.

### Cost and accuracy

Per path step: about one Horner update to rebuild ``S_{j-1}``, one to recompute the
``u_i``, and about two for the transposed chain, so roughly four forward steps in total.
The hot loop does not allocate.

The price of rebuilding is rounding drift: each ``\otimes \exp(-z)`` adds a little error,
growing with ``N`` and the size of the increments. In practice it is small: rewinding a
10 000-step path with unit-scale increments (``D = 4``, ``M = 5``) returns to the unit
tensor within ``1.3 \cdot 10^{-10}``.
