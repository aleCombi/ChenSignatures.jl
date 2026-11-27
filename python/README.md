# chen-signatures

Fast rough path signatures for Python, powered by a high-performance Julia backend.

[![PyPI](https://img.shields.io/pypi/v/chen-signatures)](https://pypi.org/project/chen-signatures/)
[![Python](https://img.shields.io/pypi/pyversions/chen-signatures)](https://pypi.org/project/chen-signatures/)

`chen-signatures` brings the speed and numerical stability of Julia’s **ChenSignatures.jl** to Python (through a thin ctypes layer, no juliacall), offering a modern, actively maintained alternative to existing signature libraries.

Against pySigLib 4.1 (`benchmarks/vs_pysiglib.py`, RTX 2060 laptop, 8 threads):

- **Signature kernels:** the Gram matrix and its gradient are 3–5× faster on CPU and
  1.4–20× faster on GPU (largest for long paths in float32).
- **Signatures and log-signatures:** comparable speed (between 0.5× and 1.6× depending on
  the case); gradients 0.9–1.2×.
- **Startup:** `import chen` takes 0.02 s and the first result arrives in under a
  second (pySigLib: about 1.1 s). The first GPU call compiles kernels (about 15–20 s).


---

## Installation

```bash
pip install chen-signatures
```

The first computation in a new environment installs a Julia runtime and precompiles
ChenSignatures.jl (a few minutes, **once per environment**). After that, `import chen`
is instant and Julia starts on the first computation of a process: under a second from
`import chen` to the first result. chen starts libjulia itself and calls precompiled C
entry points, so there is no JIT warm-up and no juliacall/PythonCall dependency (if
juliacall is already running in the process, chen uses it; import it before chen's
first computation).

---

## Quick Start

```python
import chen
import numpy as np

path = np.random.randn(1000, 10)

# Compute signature
signature = chen.sig(path, m=5)

# Compute log-signature (requires prepare step)
basis = chen.prepare_logsig(d=10, m=5)  # d must match path dimension
logsignature = chen.logsig(path, basis)

# Path augmentations (time/lead–lag) with convenience wrappers
augmented = chen.time_augment(path, Tspan=1.0)
leadlag = chen.lead_lag(path)
sig_time = chen.sig_time(path, m=5)
sig_leadlag = chen.sig_leadlag(path, m=5)
```

### Signature kernels

```python
X = np.random.randn(64, 3, 32)                   # (N, d, B): 32 paths
Y = np.random.randn(80, 3, 16)
K = chen.sig_gram(X, Y)                          # (32, 16), untruncated (PDE) kernel
K = chen.sig_gram(X, dyadic_order=1, static_kernel=chen.RBFKernel(0.5))
k = chen.sig_kernel(X[:, :, 0], Y[:, :, 0])      # one pair
K4 = chen.sig_gram(X, Y, m=4)                    # kernel of signatures truncated at level 4
```

### PyTorch (autograd, CUDA)

`chen.torch` takes batch-first tensors, `(B, N, d)`, and differentiates through
ChenSignatures' handwritten adjoints:

```python
import torch, chen.torch as ct

x = torch.randn(32, 100, 3, requires_grad=True)
ct.sig(x, 4).sum().backward()
ct.logsig(x, chen.prepare_logsig(3, 4)).sum().backward()

X = torch.randn(64, 128, 3, device="cuda", requires_grad=True)
K = ct.sig_gram(X)                               # runs on the GPU, (64, 64)
K.sum().backward()
```

CUDA tensors run `sig_gram` / `sig_kernel` on the GPU without copying to the host
(linear or RBF static kernel). This needs CUDA.jl, installed once with `chen.install_gpu()`.
The first GPU call in a process loads CUDA.jl and compiles the kernels (10–30 s).
Loading GPU support also switches torch's autograd to run backward passes on the
calling thread (`torch.autograd.set_multithreading_enabled(False)`): Julia cannot be
called from torch's per-device worker threads.

---

## API

### `sig(path, m)`

Compute a truncated signature up to level `m`.

```python
signature = chen.sig(path, m=3)
```

**Args:**
- `path`: (N, d) array-like where N is path length, d is dimension
- `m`: truncation level (positive integer)

**Returns:** 1D numpy array of signature coefficients

---

### `prepare_logsig(d, m)`

Precompute and cache the Lyndon basis for log-signature computation.

```python
basis = chen.prepare_logsig(d=10, m=5)
```

**Args:**
- `d`: path dimension (positive integer)
- `m`: truncation level (positive integer)

**Returns:** Basis cache object (reusable across multiple paths)

---

### `logsig(path, basis)`

Compute log-signatures using a precomputed Lyndon basis.

```python
basis = chen.prepare_logsig(d=10, m=5)
logsig = chen.logsig(path, basis)
```

**Args:**
- `path`: (N, d) array-like where N is path length, d is dimension
- `basis`: Basis object from `prepare_logsig(d, m)`

**Returns:** 1D numpy array of log-signature coefficients (Lyndon basis projection)

**Note:** The `basis` object should match the dimension of your path. Prepare it once and reuse for multiple paths of the same dimension.

### Path Augmentations

```python
augmented = chen.time_augment(path, Tspan=1.0)   # add time as first coordinate
leadlag = chen.lead_lag(path)                    # standard lead–lag transform

sig_time = chen.sig_time(path, m=3)              # = chen.sig(time_augment(path), 3)
sig_leadlag = chen.sig_leadlag(path, m=3)        # = chen.sig(lead_lag(path), 3)
```

---

## Supported Types

- `float32`, `float64`
- Any NumPy array-like input  
- Contiguous arrays recommended (handled automatically)

---

## Use Cases

- Financial time series and derivatives pricing
- Neural CDEs and machine learning
- Feature engineering and representation learning  

---

## Threads

Julia uses all cores by default; set `CHEN_JULIA_THREADS` (e.g. `4`) before the first
computation to change it (`PYTHON_JULIACALL_THREADS` and `JULIA_NUM_THREADS` are read
too, in that order). Ctrl-C works as usual in Python; a running Julia call
finishes before the `KeyboardInterrupt` is raised.

## Limitations

- First use in a new environment is slow (Julia installation and precompilation)
- Julia starts on the first computation of a process (under a second)
- On GPU: only `sig_gram` / `sig_kernel`, linear or RBF static kernel; the first GPU call compiles kernels

---

## Requirements

- Python ≥ 3.9  
- NumPy ≥ 1.20  
- ~500MB disk space for Julia runtime

---

## Citation

```bibtex
@software{chen_signatures,
  author = {Combi, Alessandro},
  title = {chen-signatures: Fast rough path signatures for Python},
  year = {2025},
  url = {https://github.com/aleCombi/ChenSignatures.jl}
}
```

---

## Contributing

Issues and feedback are welcome at https://github.com/aleCombi/ChenSignatures.jl/issues

However, pull requests are not being accepted at this time.
