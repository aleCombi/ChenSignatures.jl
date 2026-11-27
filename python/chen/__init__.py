# python/chen/__init__.py
"""
chen-signatures: fast path signatures and signature kernels, powered by Julia.

NumPy layout (as in the Julia package): a path is (N, d), a batch is (N, d, B).
`chen.torch` has the PyTorch versions (batch-first, autograd, CUDA).

`import chen` is instant: Julia starts on the first computation.
"""
from dataclasses import dataclass

import numpy as np

from chen._version import __version__
import ctypes as _ct

from chen._julia import api as _api, install_gpu, new_handle as _new_handle, JuliaError

# ============================================================================
# Helpers
# ============================================================================


def _as_path(path, ndims=(2, 3), name="path"):
    """Float32/float64, column-major (Julia layout), and the precision code."""
    arr = np.asarray(path)
    if arr.dtype not in (np.float32, np.float64):
        arr = arr.astype(np.float64)
    if arr.ndim not in ndims:
        want = " or ".join({2: "2D (N, d)", 3: "3D (N, d, B)"}[k] for k in ndims)
        raise ValueError(f"`{name}` must be {want}; got shape {arr.shape}")
    if arr.shape[0] < 2:
        raise ValueError(f"`{name}` needs at least 2 points, got N={arr.shape[0]}")
    arr = np.asfortranarray(arr)
    return arr, (32 if arr.dtype == np.float32 else 64)


def _check_level(m):
    m = int(m)
    if m < 1:
        raise ValueError(f"truncation level m must be at least 1, got {m}")
    return m


def sig_length(d: int, m: int) -> int:
    """Length of the truncated signature: d + d^2 + ... + d^m."""
    return sum(d**k for k in range(1, m + 1))


# ============================================================================
# Signatures
# ============================================================================


def sig(path, m: int, threaded: bool = True) -> np.ndarray:
    """
    Truncated signature of a path or a batch of paths, levels 1..m.

    Args:
        path: (N, d) path, or (N, d, B) batch of B paths. float32 is kept, other
            types are converted to float64.
        m: truncation level (>= 1).
        threaded: use Julia threads across the batch.

    Returns:
        (L,) for one path, (L, B) for a batch, L = d + d^2 + ... + d^m.

    Example:
        >>> chen.sig(np.random.randn(100, 3), 4).shape
        (120,)
    """
    arr, prec = _as_path(path)
    m = _check_level(m)
    N, d = arr.shape[:2]
    B = arr.shape[2] if arr.ndim == 3 else 0
    L = sig_length(d, m)
    out = np.empty((L, B) if B else (L,), dtype=arr.dtype, order="F")
    _api().sig(prec, out.ctypes.data, arr.ctypes.data, N, d, B, m, int(bool(threaded)))
    return out


class Basis:
    """Lyndon basis for `logsig`, from `prepare_logsig(d, m)` (a Julia object)."""

    def __init__(self, d, m):
        self.d, self.m = int(d), int(m)
        a = _api()
        self.handle = _new_handle(a.prepare, self.d, self.m)

    def __repr__(self):
        return f"chen.Basis(d={self.d}, m={self.m})"


def prepare_logsig(d: int, m: int) -> Basis:
    """
    Precompute the Lyndon basis for dimension d and level m (like
    `iisignature.prepare`). Reuse the result across `logsig` calls.
    """
    return Basis(d, m)


def _logsig_length(basis, coords_code):
    n = _ct.c_int64(0)
    _api().logsig_length(basis.handle.h, coords_code, _ct.byref(n))
    return n.value


_COORDS = {"lyndon": 1, "words": 2, "expanded": 3}


def logsig(path, basis, threaded: bool = True, coords: str = "lyndon") -> np.ndarray:
    """
    Log-signature of a path or batch, with a basis from `prepare_logsig(d, m)`.

    Args:
        path: (N, d) path or (N, d, B) batch.
        basis: result of `prepare_logsig(d, m)`.
        threaded: use Julia threads across the batch.
        coords: "lyndon" (Lyndon basis, iisignature's default), "words"
            (coefficients of the tensor logarithm at the Lyndon words, Signatory's
            default) or "expanded" (the full tensor logarithm, length of `sig`).

    Returns:
        (L,) for one path, (L, B) for a batch.
    """
    if coords not in _COORDS:
        raise ValueError(f"coords must be 'lyndon', 'words' or 'expanded', got {coords!r}")
    arr, prec = _as_path(path)
    N, d = arr.shape[:2]
    if d != basis.d:
        raise ValueError(
            f"Dimension mismatch between path (d={d}) and basis (d={basis.d}). "
            "Did you call prepare_logsig with the correct dimension?"
        )
    B = arr.shape[2] if arr.ndim == 3 else 0
    c = _COORDS[coords]
    L = _logsig_length(basis, c)
    out = np.empty((L, B) if B else (L,), dtype=arr.dtype, order="F")
    _api().logsig(prec, out.ctypes.data, arr.ctypes.data, N, d, B, basis.handle.h, c, int(bool(threaded)))
    return out


def rolling_sig(path, m: int, window_size: int, *, stride: int = 1, threaded: bool = True) -> np.ndarray:
    """
    Signatures of the rolling windows path[i*stride : i*stride + window_size].

    Returns:
        (L, num_windows), num_windows = (N - window_size) // stride + 1.
    """
    arr = np.asarray(path)
    if arr.ndim != 2:
        raise ValueError(f"`path` must be 2D (N, d); got shape {arr.shape}")
    N = arr.shape[0]
    if not 2 <= window_size <= N:
        raise ValueError(f"window_size must be in [2, N={N}], got {window_size}")
    if stride < 1:
        raise ValueError(f"stride must be at least 1, got {stride}")
    win = np.lib.stride_tricks.sliding_window_view(arr, window_size, axis=0)[::stride]  # (W, d, w)
    return sig(np.transpose(win, (2, 1, 0)), m, threaded=threaded)


# ============================================================================
# Signature kernels
# ============================================================================


@dataclass(frozen=True)
class LinearKernel:
    """κ(a, b) = <a, b>: the plain signature kernel. The default."""


@dataclass(frozen=True)
class RBFKernel:
    """κ(a, b) = exp(-|a - b|² / 2σ²): the signature kernel of the RBF-lifted paths."""
    sigma: float

    def __post_init__(self):
        if not self.sigma > 0:
            raise ValueError(f"RBFKernel: sigma must be positive, got {self.sigma}")


def _kernel_code(static_kernel):
    if static_kernel is None or isinstance(static_kernel, LinearKernel):
        return 0, 1.0
    if isinstance(static_kernel, RBFKernel):
        return 1, float(static_kernel.sigma)
    raise TypeError(f"static_kernel must be LinearKernel() or RBFKernel(sigma), got {static_kernel!r}")


def sig_gram(X, Y=None, *, dyadic_order: int = 0, static_kernel=None, m: int = None,
             threaded: bool = True) -> np.ndarray:
    """
    Gram matrix of signature kernels between two batches of paths.

    By default the untruncated kernel, from the Goursat PDE (Salvi et al. 2021), with
    each increment split into 2^dyadic_order pieces and an optional static kernel
    (`LinearKernel()`, `RBFKernel(sigma)`). With `m`, the kernel of the signatures
    truncated at level m.

    Args:
        X: (N, d, Bx) batch; Y: (M, d, By) batch, or None for the symmetric Gram of X.

    Returns:
        (Bx, By) array, K[i, j] = k(X[:, :, i], Y[:, :, j]).
    """
    Xa, prec = _as_path(X, (3,), "X")
    sym = Y is None
    if sym:
        Ya = Xa
    else:
        Ya, precY = _as_path(Y, (3,), "Y")
        if precY != prec:
            Ya, precY = _as_path(Ya.astype(Xa.dtype), (3,), "Y")
        if Ya.shape[1] != Xa.shape[1]:
            raise ValueError(f"X and Y must have the same dimension d, got {Xa.shape[1]} and {Ya.shape[1]}")
    if dyadic_order < 0:
        raise ValueError(f"dyadic_order must be non-negative, got {dyadic_order}")
    kcode, sigma = _kernel_code(static_kernel)
    if m is not None and kcode != 0:
        raise ValueError("static_kernel applies to the untruncated kernel only, not with m")
    mm = 0 if m is None else _check_level(m)
    Nx, d, Bx = Xa.shape
    Ny, _, By = Ya.shape
    out = np.empty((Bx, By), dtype=Xa.dtype, order="F")
    _api().sig_gram(prec, out.ctypes.data, Xa.ctypes.data, Nx, Bx, Ya.ctypes.data, Ny, By, d,
                    int(sym), int(dyadic_order), kcode, sigma, mm, int(bool(threaded)))
    return out


def sig_kernel(x, y, *, dyadic_order: int = 0, static_kernel=None, m: int = None) -> float:
    """
    Signature kernel k(x, y) of two paths (N, d) and (M, d); keywords as `sig_gram`.
    """
    xa, _ = _as_path(x, (2,), "x")
    ya, _ = _as_path(y, (2,), "y")
    K = sig_gram(xa[:, :, None], ya[:, :, None].astype(xa.dtype), dyadic_order=dyadic_order,
                 static_kernel=static_kernel, m=m, threaded=False)
    return float(K[0, 0])


# ============================================================================
# Augmentations (NumPy; same definitions as the Julia package)
# ============================================================================


def time_augment(path, *, Tspan=1.0, times=None) -> np.ndarray:
    """
    Prepend a time coordinate: a uniform grid on [0, Tspan], or `times` (length N).

    Returns:
        (N, d+1) or (N, d+1, B).
    """
    arr = np.asarray(path)
    if arr.ndim not in (2, 3):
        raise ValueError(f"`path` must be 2D (N, d) or 3D (N, d, B); got shape {arr.shape}")
    if arr.dtype not in (np.float32, np.float64):
        arr = arr.astype(np.float64)
    N = arr.shape[0]
    if times is None:
        t = np.linspace(0.0, Tspan, N, dtype=arr.dtype)
    else:
        t = np.asarray(times, dtype=arr.dtype)
        if t.shape != (N,):
            raise ValueError(f"`times` length ({t.shape[0]}) must match N={N}")
    tcol = np.broadcast_to(t.reshape((N, 1) + (1,) * (arr.ndim - 2)), (N, 1) + arr.shape[2:])
    return np.concatenate([tcol, arr], axis=1)


def lead_lag(path) -> np.ndarray:
    """
    Lead–lag transform: (N, d) -> (2N-1, 2d), points (x_i, x_i), (x_i, x_{i+1}), ...
    (and (N, d, B) -> (2N-1, 2d, B)).
    """
    arr = np.asarray(path)
    if arr.ndim not in (2, 3):
        raise ValueError(f"`path` must be 2D (N, d) or 3D (N, d, B); got shape {arr.shape}")
    if arr.dtype not in (np.float32, np.float64):
        arr = arr.astype(np.float64)
    rep = np.repeat(arr, 2, axis=0)
    return np.concatenate([rep[:-1], rep[1:]], axis=1)


def sig_time(path, m: int, *, Tspan=1.0, times=None, threaded: bool = True) -> np.ndarray:
    """Signature of the time-augmented path."""
    return sig(time_augment(path, Tspan=Tspan, times=times), m, threaded=threaded)


def sig_leadlag(path, m: int, *, threaded: bool = True) -> np.ndarray:
    """Signature of the lead–lag transformed path."""
    return sig(lead_lag(path), m, threaded=threaded)


def logsig_time(path, basis, *, Tspan=1.0, times=None, threaded: bool = True) -> np.ndarray:
    """Log-signature of the time-augmented path."""
    return logsig(time_augment(path, Tspan=Tspan, times=times), basis, threaded=threaded)


def logsig_leadlag(path, basis, *, threaded: bool = True) -> np.ndarray:
    """Log-signature of the lead–lag transformed path."""
    return logsig(lead_lag(path), basis, threaded=threaded)


__all__ = [
    '__version__',
    'sig',
    'sig_length',
    'logsig',
    'prepare_logsig',
    'rolling_sig',
    'sig_kernel',
    'sig_gram',
    'LinearKernel',
    'RBFKernel',
    'time_augment',
    'lead_lag',
    'sig_time',
    'sig_leadlag',
    'logsig_time',
    'logsig_leadlag',
    'install_gpu',
    'Basis',
    'JuliaError',
]
