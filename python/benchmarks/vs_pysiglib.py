"""chen vs pySigLib, the whole lifecycle: import, first call, steady-state forward and
gradient, CPU and GPU.

    cd python && .venv/bin/python benchmarks/vs_pysiglib.py [--threads 8] [--no-gpu]

Needs `pysiglib` (and `pysiglib-cuda`, torch with CUDA, and `chen.install_gpu()` for the
GPU rows). Cold-start rows run in fresh processes; times there include everything
after the interpreter starts (for chen: Julia startup, and on GPU, loading CUDA.jl and
compiling the GPU kernels).
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--threads", type=int, default=os.cpu_count())
ap.add_argument("--no-gpu", action="store_true")
ap.add_argument("--cold-reps", type=int, default=3)
args = ap.parse_args()
TH = args.threads
os.environ["PYTHON_JULIACALL_THREADS"] = str(TH)

# ---------------------------------------------------------------------------
# Cold start: fresh process per measurement
# ---------------------------------------------------------------------------

SETUP = """
import time, numpy as np
T0 = time.perf_counter()
{body}
print(f"{{(time.perf_counter() - T0):.3f}}")
"""
DATA = "rng = np.random.default_rng(0); X = 0.1*np.cumsum(rng.standard_normal((32, 512, 8)), axis=1)"
GDATA = "rng = np.random.default_rng(0); X = 0.1*np.cumsum(rng.standard_normal((64, 64, 3)), axis=1)"
COLD = [
    ("import", "import chen", "import pysiglib"),
    ("import + first sig (32×512×8, m=4)",
     f"import chen\n{DATA}\nchen.sig(np.transpose(X, (1, 2, 0)), 4)",
     f"import pysiglib\n{DATA}\npysiglib.sig(X, 4, n_jobs={TH})"),
    ("import + first logsig (lyndon)",
     f"import chen\n{DATA}\nchen.logsig(np.transpose(X, (1, 2, 0)), chen.prepare_logsig(8, 4))",
     f"import pysiglib\n{DATA}\npysiglib.prepare_log_sig(8, 4, 2)\npysiglib.log_sig(X, 4, method=2, n_jobs={TH})"),
    ("import + first sig_gram CPU (64 paths × 64, d=3)",
     f"import chen\n{GDATA}\nchen.sig_gram(np.transpose(X, (1, 2, 0)))",
     f"import pysiglib\n{GDATA}\npysiglib.sig_kernel_gram(X, X, dyadic_order=0, n_jobs={TH})"),
    ("import torch + first sig gradient",
     f"import torch, chen.torch as ct\n{DATA}\nx = torch.tensor(X, requires_grad=True)\nct.sig(x, 4).sum().backward()",
     f"import torch, pysiglib.torch_api as ta\n{DATA}\nx = torch.tensor(X, requires_grad=True)\nta.sig(x, 4, n_jobs={TH}).sum().backward()"),
]
if not args.no_gpu:
    COLD.append(("import torch + first GPU sig_gram fwd+grad (F32)",
                 f"import torch, chen.torch as ct\n{GDATA}\nx = torch.tensor(X, dtype=torch.float32, device='cuda', requires_grad=True)\n"
                 "ct.sig_gram(x, x.detach()).sum().backward(); torch.cuda.synchronize()",
                 f"import torch, pysiglib.torch_api as ta\n{GDATA}\nx = torch.tensor(X, dtype=torch.float32, device='cuda', requires_grad=True)\n"
                 "ta.sig_kernel_gram(x, x.detach(), dyadic_order=0).sum().backward(); torch.cuda.synchronize()"))


def cold(body):
    ts = []
    for _ in range(args.cold_reps):
        r = subprocess.run([sys.executable, "-c", SETUP.format(body=body)], capture_output=True, text=True)
        if r.returncode != 0:
            return float("nan")
        ts.append(float(r.stdout.strip().splitlines()[-1]))
    return sorted(ts)[len(ts) // 2]


print(f"threads: {TH}   (cold start: median of {args.cold_reps} fresh processes, seconds)")
print(f"{'':52s} {'chen':>8s} {'pySigLib':>9s}")
for name, c, p in COLD:
    print(f"{name:52s} {cold(c):8.2f} {cold(p):9.2f}", flush=True)

# ---------------------------------------------------------------------------
# Steady state, one process
# ---------------------------------------------------------------------------

import torch
import chen
import chen.torch as ct
import pysiglib
import pysiglib.torch_api as ta


def med(f, k=20, sync=False):
    f()
    if sync:
        torch.cuda.synchronize()
    ts = []
    for _ in range(k):
        t = time.perf_counter(); f()
        if sync:
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return sorted(ts)[k // 2] * 1e3


def relerr(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return np.abs(a - b).max() / np.abs(b).max()


rows = []


def row(name, tc, tp, err):
    print(f"{name:52s} {tc:8.2f} {tp:9.2f} {tp / tc:7.2f}×   (diff {err:.0e})", flush=True)


print(f"\nsteady state, ms (median)                            {'chen':>8s} {'pySigLib':>9s}  pySigLib/chen")
rng = np.random.default_rng(1)
for dt, tdt in ((np.float64, torch.float64), (np.float32, torch.float32)):
    tag = np.dtype(dt).name
    Xb = (0.1 * np.cumsum(rng.standard_normal((32, 512, 8)), axis=1)).astype(dt)   # (B, N, d)
    Xc = np.asfortranarray(np.transpose(Xb, (1, 2, 0)))                           # (N, d, B)
    row(f"sig fwd 32×512×8 m=4 {tag}", med(lambda: chen.sig(Xc, 4)),
        med(lambda: pysiglib.sig(Xb, 4, n_jobs=TH)), relerr(chen.sig(Xc, 4).T, pysiglib.sig(Xb, 4, n_jobs=TH)))
    b = chen.prepare_logsig(8, 4)
    pysiglib.prepare_log_sig(8, 4, 1); pysiglib.prepare_log_sig(8, 4, 2)
    for coords, meth in (("words", 1), ("lyndon", 2)):
        row(f"logsig fwd {coords} 32×512×8 m=4 {tag}", med(lambda: chen.logsig(Xc, b, coords=coords)),
            med(lambda: pysiglib.log_sig(Xb, 4, method=meth, n_jobs=TH)),
            relerr(chen.logsig(Xc, b, coords=coords).T, pysiglib.log_sig(Xb, 4, method=meth, n_jobs=TH)))
    xt = torch.tensor(Xb)

    def grad(f):
        x = xt.clone().requires_grad_(True)
        f(x).sum().backward()
        return x.grad

    row(f"sig fwd+grad 32×512×8 m=4 {tag}", med(lambda: grad(lambda x: ct.sig(x, 4)), 10),
        med(lambda: grad(lambda x: ta.sig(x, 4, n_jobs=TH)), 10),
        relerr(grad(lambda x: ct.sig(x, 4)), grad(lambda x: ta.sig(x, 4, n_jobs=TH))))
    row(f"logsig fwd+grad lyndon 32×512×8 m=4 {tag}", med(lambda: grad(lambda x: ct.logsig(x, b)), 10),
        med(lambda: grad(lambda x: ta.log_sig(x, 4, method=2, n_jobs=TH)), 10),
        relerr(grad(lambda x: ct.logsig(x, b)), grad(lambda x: ta.log_sig(x, 4, method=2, n_jobs=TH))))

    G = (0.1 * np.cumsum(rng.standard_normal((64, 64, 3)), axis=1)).astype(dt)
    H = (0.1 * np.cumsum(rng.standard_normal((64, 64, 3)), axis=1)).astype(dt)
    Gc, Hc = np.asfortranarray(np.transpose(G, (1, 2, 0))), np.asfortranarray(np.transpose(H, (1, 2, 0)))
    row(f"sig_gram CPU fwd 64×64 paths, N=64 d=3 {tag}", med(lambda: chen.sig_gram(Gc, Hc), 10),
        med(lambda: pysiglib.sig_kernel_gram(G, H, dyadic_order=0, n_jobs=TH), 10),
        relerr(chen.sig_gram(Gc, Hc), pysiglib.sig_kernel_gram(G, H, dyadic_order=0, n_jobs=TH)))
    gt, ht = torch.tensor(G), torch.tensor(H)
    W = torch.randn(64, 64, dtype=tdt)

    def ggrad(f, dev="cpu"):
        x = gt.to(dev).clone().requires_grad_(True)
        (f(x, ht.to(dev)) * W.to(dev)).sum().backward()
        return x.grad

    row(f"sig_gram CPU fwd+grad 64×64 paths {tag}", med(lambda: ggrad(lambda x, y: ct.sig_gram(x, y)), 5),
        med(lambda: ggrad(lambda x, y: ta.sig_kernel_gram(x, y, dyadic_order=0, n_jobs=TH)), 5),
        relerr(ggrad(lambda x, y: ct.sig_gram(x, y)), ggrad(lambda x, y: ta.sig_kernel_gram(x, y, dyadic_order=0, n_jobs=TH))))
    if not args.no_gpu:
        for (N, B) in ((64, 64), (128, 64), (256, 32)):
            Gg = torch.tensor(0.1 * np.cumsum(rng.standard_normal((B, N, 3)), axis=1), dtype=tdt, device="cuda")
            Hg = torch.tensor(0.1 * np.cumsum(rng.standard_normal((B, N, 3)), axis=1), dtype=tdt, device="cuda")
            Wg = torch.randn(B, B, dtype=tdt, device="cuda")

            def gg(f):
                x = Gg.clone().requires_grad_(True)
                (f(x, Hg) * Wg).sum().backward()
                return x.grad

            with torch.no_grad():
                row(f"sig_gram GPU fwd {B*B} pairs N={N} {tag}", med(lambda: ct.sig_gram(Gg, Hg), 10, True),
                    med(lambda: ta.sig_kernel_gram(Gg, Hg, dyadic_order=0), 10, True),
                    relerr(ct.sig_gram(Gg, Hg).cpu(), ta.sig_kernel_gram(Gg, Hg, dyadic_order=0).cpu()))
            row(f"sig_gram GPU fwd+grad {B*B} pairs N={N} {tag}", med(lambda: gg(ct.sig_gram), 5, True),
                med(lambda: gg(lambda x, y: ta.sig_kernel_gram(x, y, dyadic_order=0)), 5, True),
                relerr(gg(ct.sig_gram).cpu(), gg(lambda x, y: ta.sig_kernel_gram(x, y, dyadic_order=0)).cpu()))
