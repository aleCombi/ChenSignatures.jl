# ChenSignatures.jl

[![CI](https://github.com/aleCombi/ChenSignatures.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/aleCombi/ChenSignatures.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://aleCombi.github.io/ChenSignatures.jl)

Fast truncated path signatures and log-signatures in Julia.

The signature of a path is the sequence of its iterated integrals, a graded
feature map that characterises the path up to reparametrisation. It is
computed incrementally via Chen's identity, one segment at a time.

## Install

```julia
] add ChenSignatures
```

## Usage

```julia
using ChenSignatures

path = randn(100, 3)              # N points × d dims

sig(path, 4)                      # 120-vector: 3 + 9 + 27 + 81

basis = prepare(3, 4)             # Lyndon basis, d = 3, level 4
logsig(path, basis)               # 32-vector

paths = randn(50, 3, 32)          # N × d × batch
sig(paths, 4)                     # 120 × 32, threaded
logsig(paths, basis)              # 32 × 32

sig_kernel(path, randn(80, 3))    # untruncated signature kernel (PDE)
sig_gram(paths; dyadic_order=1)   # 32 × 32 Gram matrix, threaded
sig_gram(paths; static_kernel=RBFKernel(0.5))
sig_gram(paths, 4)                # truncated at level 4
```

Also available: `time_augment`, `lead_lag` (and the `sig_time`, `sig_leadlag`
shortcuts), `rolling_sig` for sliding windows, and word labels (`sig_words`,
`sig_index`, `logsig_words`). Everything above is differentiable with Zygote, Enzyme
and Mooncake, all running the same handwritten adjoints. See the
[docs](https://aleCombi.github.io/ChenSignatures.jl).

`sig_gram` also runs on GPU arrays (CUDA.jl, AMDGPU.jl, ...) through a
KernelAbstractions extension, with a handwritten adjoint kernel for the gradient:

```julia
using CUDA
K = sig_gram(CuArray(paths))      # 32 × 32 CuArray
```

Python bindings: [`chen-signatures`](./python).

## Future work

- More static kernels (Matérn, rational quadratic, polynomial)
- Neural / rough CDEs
- GPU: static kernels other than linear, `sig`/`logsig` on GPU

## Citation

```bibtex
@software{chen_signatures,
  author = {Combi, Alessandro},
  title  = {ChenSignatures.jl: High-performance signatures and log-signatures},
  year   = {2025},
  url    = {https://github.com/aleCombi/ChenSignatures.jl}
}
```

## License

MIT, see [LICENSE](./LICENSE).
