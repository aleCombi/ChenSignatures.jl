"""Signature kernels, the NumPy augmentations, and the torch autograd functions."""
import numpy as np
import pytest

import chen

rng = np.random.default_rng(3)


def paths(N, d, B=None):
    shape = (N, d) if B is None else (N, d, B)
    return 0.1 * np.cumsum(rng.standard_normal(shape), axis=0)


def test_sig_gram_matches_sig_kernel():
    X, Y = paths(12, 3, 4), paths(9, 3, 5)
    K = chen.sig_gram(X, Y, dyadic_order=1)
    assert K.shape == (4, 5)
    for i, j in ((0, 0), (3, 4), (2, 1)):
        assert K[i, j] == pytest.approx(chen.sig_kernel(X[:, :, i], Y[:, :, j], dyadic_order=1), rel=1e-12)
    np.testing.assert_allclose(chen.sig_gram(X), chen.sig_gram(X, X), rtol=1e-12)


def test_truncated_kernel_is_inner_product_of_signatures():
    x, y = paths(10, 2), paths(8, 2)
    k = chen.sig_kernel(x, y, m=4)
    assert k == pytest.approx(1 + chen.sig(x, 4) @ chen.sig(y, 4), rel=1e-12)


def test_pde_kernel_approaches_truncated():
    x, y = paths(10, 2), paths(8, 2)
    assert chen.sig_kernel(x, y, dyadic_order=3) == pytest.approx(chen.sig_kernel(x, y, m=8), rel=1e-5)


def test_rbf_kernel_and_float32():
    X = paths(10, 2, 3)
    K = chen.sig_gram(X, static_kernel=chen.RBFKernel(0.5))
    assert np.allclose(K, K.T)
    K32 = chen.sig_gram(X.astype(np.float32))
    assert K32.dtype == np.float32
    np.testing.assert_allclose(K32, chen.sig_gram(X), rtol=1e-5)
    with pytest.raises(ValueError):
        chen.RBFKernel(0.0)


def test_augmentations_match_julia_definitions():
    x = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    # lead–lag: (x_i, x_i), (x_i, x_{i+1}), ..., (x_N, x_N)
    expect = np.array([[0, 1, 0, 1], [0, 1, 2, 3], [2, 3, 2, 3], [2, 3, 4, 5], [4, 5, 4, 5]], dtype=float)
    np.testing.assert_array_equal(chen.lead_lag(x), expect)
    np.testing.assert_array_equal(chen.time_augment(x, Tspan=2.0), np.column_stack([[0.0, 1.0, 2.0], x]))
    np.testing.assert_array_equal(chen.time_augment(x, times=[0.0, 0.5, 3.0])[:, 0], [0.0, 0.5, 3.0])
    X = paths(6, 2, 3)
    np.testing.assert_array_equal(chen.lead_lag(X)[:, :, 1], chen.lead_lag(X[:, :, 1]))
    np.testing.assert_array_equal(chen.time_augment(X)[:, :, 2], chen.time_augment(X[:, :, 2]))


def test_julia_errors_become_exceptions():
    b = chen.prepare_logsig(2, 3)
    with pytest.raises(ValueError):
        chen.logsig(paths(5, 3), b)                       # caught in Python
    with pytest.raises(ValueError):
        chen.sig_gram(paths(5, 2, 2), static_kernel=chen.RBFKernel(1.0), m=2)
    import ctypes
    from chen._julia import api
    with pytest.raises(chen.JuliaError, match="KeyError"):
        api().logsig_length(-1, 1, ctypes.byref(ctypes.c_int64(0)))   # raised in Julia


torch = pytest.importorskip("torch")
import chen.torch as ct  # noqa: E402


def _fd_check(f, x, eps=1e-6, n=5):
    """Directional finite differences of a scalar f against autograd."""
    x = x.clone().requires_grad_(True)
    f(x).backward()
    for _ in range(n):
        v = torch.randn_like(x)
        fd = (f(x.detach() + eps * v) - f(x.detach() - eps * v)) / (2 * eps)
        # tolerance relative to |grad||v|: a small directional derivative is a difference
        # of large terms, and finite differences lose digits there
        scale = (x.grad.norm() * v.norm()).item()
        assert (x.grad * v).sum().item() == pytest.approx(fd.item(), abs=1e-7 * scale)


def test_torch_sig_batch_and_grad():
    X = torch.tensor(np.transpose(paths(9, 3, 4), (2, 0, 1)))          # (B, N, d)
    S = ct.sig(X, 3)
    np.testing.assert_allclose(S.numpy(), chen.sig(np.transpose(X.numpy(), (1, 2, 0)), 3).T, rtol=1e-13)
    W = torch.randn_like(S)
    _fd_check(lambda x: (ct.sig(x, 3) * W).sum(), X)


def test_torch_logsig_grad():
    b = chen.prepare_logsig(2, 4)
    X = torch.tensor(np.transpose(paths(8, 2, 3), (2, 0, 1)))
    W = torch.randn(3, int(chen.logsig(paths(3, 2), b).shape[0]), dtype=torch.float64)
    _fd_check(lambda x: (ct.logsig(x, b) * W).sum(), X)


def test_torch_sig_gram_grad():
    X = torch.tensor(np.transpose(paths(9, 2, 3), (2, 0, 1)))
    Y = torch.tensor(np.transpose(paths(7, 2, 4), (2, 0, 1)))
    W = torch.randn(3, 4, dtype=torch.float64)
    K = ct.sig_gram(X, Y)
    np.testing.assert_allclose(K.numpy(), chen.sig_gram(np.transpose(X.numpy(), (1, 2, 0)),
                                                        np.transpose(Y.numpy(), (1, 2, 0))), rtol=1e-13)
    _fd_check(lambda x: (ct.sig_gram(x, Y) * W).sum(), X)
    _fd_check(lambda y: (ct.sig_gram(X, y) * W).sum(), Y)
    Ws = torch.randn(3, 3, dtype=torch.float64)
    _fd_check(lambda x: (ct.sig_gram(x) * Ws).sum(), X)
    _fd_check(lambda x: ct.sig_kernel(x[0], Y[1]), X)


def test_torch_sig_gram_cuda_matches_cpu():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from chen._julia import gpu_api
    try:
        gpu_api()
    except RuntimeError:
        pytest.skip("chen.install_gpu() not run")
    X = torch.tensor(np.transpose(paths(20, 3, 5), (2, 0, 1)))
    Y = torch.tensor(np.transpose(paths(15, 3, 4), (2, 0, 1)))
    W = torch.randn(5, 4, dtype=torch.float64)
    for kw in ({}, {"static_kernel": chen.RBFKernel(0.3), "dyadic_order": 1}):
        xc, yc = X.clone().requires_grad_(), Y.clone().requires_grad_()
        (ct.sig_gram(xc, yc, **kw) * W).sum().backward()
        xg, yg = X.cuda().requires_grad_(), Y.cuda().requires_grad_()
        K = ct.sig_gram(xg, yg, **kw)
        (K * W.cuda()).sum().backward()
        assert K.is_cuda
        np.testing.assert_allclose(K.detach().cpu().numpy(), ct.sig_gram(X, Y, **kw).numpy(), rtol=1e-12)
        np.testing.assert_allclose(xg.grad.cpu().numpy(), xc.grad.numpy(), rtol=1e-10, atol=1e-14)
        np.testing.assert_allclose(yg.grad.cpu().numpy(), yc.grad.numpy(), rtol=1e-10, atol=1e-14)
