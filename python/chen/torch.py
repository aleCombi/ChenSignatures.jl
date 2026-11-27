"""PyTorch API for chen: autograd through ChenSignatures' handwritten adjoints.

Layout is batch-first, as usual in PyTorch: a path is (N, d), a batch is (B, N, d).
float32 and float64 tensors keep their dtype.

- `sig`, `logsig`: computed on the CPU (CUDA tensors are moved there and back).
- `sig_gram`, `sig_kernel`: CUDA tensors run on the GPU kernels with no copy to the
  host (needs `chen.install_gpu()` once; linear static kernel), CPU tensors on the
  threaded CPU code.
"""
import torch

from chen import _COORDS, _kernel_code, _check_level, _logsig_length, sig_length
from chen._julia import api, gpu_api, new_handle

__all__ = ["sig", "logsig", "sig_gram", "sig_kernel", "sig_torch"]


def _prec(t):
    if t.dtype == torch.float32:
        return 32
    if t.dtype == torch.float64:
        return 64
    raise TypeError(f"expected a float32 or float64 tensor, got {t.dtype}")


def _julia_layout(x):
    """Contiguous buffer whose memory is the Julia array (N, d) or (N, d, B)."""
    x = x.detach()
    if x.dim() == 2:
        return x.t().contiguous()            # (d, N) row-major == (N, d) column-major
    return x.transpose(1, 2).contiguous()    # (B, d, N) row-major == (N, d, B) column-major


def _from_julia_layout(buf):
    return buf.t() if buf.dim() == 2 else buf.transpose(1, 2)


def _check_path(x, name="path", dims=(2, 3)):
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"`{name}` must be a torch.Tensor")
    if x.dim() not in dims:
        raise ValueError(f"`{name}` must be (N, d) or (B, N, d); got shape {tuple(x.shape)}")
    n = x.shape[-2]
    if n < 2:
        raise ValueError(f"`{name}` needs at least 2 points, got N={n}")


# ----------------------------------------------------------------------------
# sig / logsig (CPU)
# ----------------------------------------------------------------------------

class _SigFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, m, threaded, basis, coords):
        dev = x.device
        xc = _julia_layout(x.cpu())
        ctx.device_out, ctx.dtype = dev, xc.dtype
        prec = _prec(xc)
        if x.dim() == 2:
            N, d, B = x.shape[0], x.shape[1], 0
        else:
            B, N, d = x.shape
        a = api()
        if basis is None:
            L = sig_length(d, m)
        else:
            if basis.d != d:
                raise ValueError(f"basis is for dimension {basis.d}, path has d={d}")
            L = _logsig_length(basis, coords)
        out = torch.empty((B, L) if B else (L,), dtype=xc.dtype)   # == Julia (L, B)
        th = int(threaded)
        args = (prec, out.data_ptr(), xc.data_ptr(), N, d, B)
        if ctx.needs_input_grad[0]:
            if basis is None:
                ctx.pb = new_handle(a.sig_fwd, *args, m, th)
            else:
                ctx.pb = new_handle(a.logsig_fwd, *args, basis.handle.h, coords, th)
            ctx.shape = xc.shape
        elif basis is None:
            a.sig(*args, m, th)
        else:
            a.logsig(*args, basis.handle.h, coords, th)
        return out.to(dev)

    @staticmethod
    def backward(ctx, g):
        g = g.detach().cpu().to(ctx.dtype).contiguous()      # == Julia (L,) or (L, B)
        xbar = torch.empty(ctx.shape, dtype=g.dtype)
        api().pull(ctx.pb.h, g.data_ptr(), xbar.data_ptr(), None)
        ctx.pb = None
        return _from_julia_layout(xbar).to(ctx.device_out), None, None, None, None


def sig(path, m: int, threaded: bool = True):
    """
    Truncated signature, levels 1..m, differentiable.

    Args:
        path: (N, d) path or (B, N, d) batch.

    Returns:
        (L,) or (B, L), L = d + d^2 + ... + d^m.
    """
    _check_path(path)
    return _apply_sig(path, _check_level(m), bool(threaded), None, 0)


def logsig(path, basis, threaded: bool = True, coords: str = "lyndon"):
    """
    Log-signature with a basis from `chen.prepare_logsig(d, m)`, differentiable.
    `coords` as in `chen.logsig`.

    Returns:
        (L,) or (B, L).
    """
    _check_path(path)
    if coords not in _COORDS:
        raise ValueError(f"coords must be 'lyndon', 'words' or 'expanded', got {coords!r}")
    return _apply_sig(path, 0, bool(threaded), basis, _COORDS[coords])


def _apply_sig(path, m, threaded, basis, coords):
    return _SigFn.apply(path, m, threaded, basis, coords)


def sig_torch(path, m):
    """Alias of `chen.torch.sig` (kept for compatibility)."""
    return sig(path, m)


# ----------------------------------------------------------------------------
# sig_gram / sig_kernel (CPU or CUDA)
# ----------------------------------------------------------------------------

class _GramFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, Y, sym, dyadic_order, kcode, sigma, threaded):
        cuda = X.is_cuda
        prec = _prec(X)
        Xc = _julia_layout(X)                 # Julia (Nx, d, Bx)
        Bx, Nx, d = X.shape
        if sym:
            Yc, By, Ny = Xc, Bx, Nx
        else:
            Yc = _julia_layout(Y)
            By, Ny, _ = Y.shape
        out = torch.empty((By, Bx), dtype=X.dtype, device=X.device)   # == Julia (Bx, By)
        need = ctx.needs_input_grad[0] or (not sym and ctx.needs_input_grad[1])
        args = (prec, out.data_ptr(), Xc.data_ptr(), Nx, Bx, Yc.data_ptr(), Ny, By, d,
                int(sym), dyadic_order, kcode, sigma)
        if cuda:
            torch.cuda.current_stream(X.device).synchronize()
            a = gpu_api()
            if need:
                ctx.pb = new_handle(a.sig_gram_fwd, *args)
            else:
                a.sig_gram(*args)
        else:
            a = api()
            if need:
                ctx.pb = new_handle(a.sig_gram_fwd, *args, int(threaded))
            else:
                a.sig_gram(*args, 0, int(threaded))
        if need:
            ctx.sym, ctx.cuda = sym, cuda
            ctx.xshape = Xc.shape
            ctx.yshape = None if sym else Yc.shape
        return out.t()

    @staticmethod
    def backward(ctx, g):
        gK = g.detach().to(g.dtype).t().contiguous()          # Julia (Bx, By)
        Xbar = torch.empty(ctx.xshape, dtype=g.dtype, device=g.device)
        if ctx.cuda:
            torch.cuda.current_stream(g.device).synchronize()
            a = gpu_api()
        else:
            a = api()
        if ctx.sym:
            a.pull(ctx.pb.h, gK.data_ptr(), Xbar.data_ptr(), None)
            Ybar = None
        else:
            Ybar = torch.empty(ctx.yshape, dtype=g.dtype, device=g.device)
            a.pull(ctx.pb.h, gK.data_ptr(), Xbar.data_ptr(), Ybar.data_ptr())
            Ybar = _from_julia_layout(Ybar)
        ctx.pb = None
        return _from_julia_layout(Xbar), Ybar, None, None, None, None, None


def sig_gram(X, Y=None, *, dyadic_order: int = 0, static_kernel=None, threaded: bool = True):
    """
    Gram matrix of untruncated signature kernels (Goursat PDE), differentiable.

    Args:
        X: (Bx, N, d) batch; Y: (By, M, d) batch, or None for the symmetric Gram of X.
            CUDA tensors run on the GPU (linear static kernel only).
        dyadic_order: each increment is split into 2^dyadic_order pieces.
        static_kernel: `chen.LinearKernel()` (default) or `chen.RBFKernel(sigma)`.

    Returns:
        (Bx, By) tensor, K[i, j] = k(X[i], Y[j]).
    """
    _check_path(X, "X", (3,))
    if Y is not None:
        _check_path(Y, "Y", (3,))
        if Y.shape[2] != X.shape[2]:
            raise ValueError(f"X and Y must have the same dimension d, got {X.shape[2]} and {Y.shape[2]}")
        if Y.dtype != X.dtype or Y.device != X.device:
            raise ValueError("X and Y must have the same dtype and device")
    if dyadic_order < 0:
        raise ValueError(f"dyadic_order must be non-negative, got {dyadic_order}")
    kcode, sigma = _kernel_code(static_kernel)
    return _GramFn.apply(X, Y, Y is None, int(dyadic_order), kcode, sigma, bool(threaded))


def sig_kernel(x, y, *, dyadic_order: int = 0, static_kernel=None):
    """Signature kernel k(x, y) of two paths (N, d), (M, d): a 0-dim tensor, differentiable."""
    _check_path(x, "x", (2,))
    _check_path(y, "y", (2,))
    return sig_gram(x[None], y[None], dyadic_order=dyadic_order, static_kernel=static_kernel,
                    threaded=False)[0, 0]
