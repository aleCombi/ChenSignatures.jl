"""Signature-kernel Gram on GPU: pySigLib and sigkernel, forward and gradient.

Writes each configuration's paths and weights to data/ (float64, C order, shapes
(B, N, d) and (B, B)) so the Julia side (compare_rivals.jl) runs on identical data,
and each library's Gram and gradient to out/ for the accuracy comparison.
Linear static kernel, dyadic order 0, loss = sum(W * K), gradient w.r.t. X.

Setup (Python 3.12, CUDA 12):
    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu126
    uv pip install --python .venv/bin/python pysiglib pysiglib-cuda "numba-cuda[cu12]" \
        "cuda-bindings>=12.9.8,<13" cython numpy setuptools
    uv pip install --python .venv/bin/python --no-build-isolation \
        "git+https://github.com/crispitagorico/sigkernel"
    .venv/bin/python bench_rivals.py
"""
import os, time, json
import numpy as np
if not hasattr(np, 'row_stack'):
    np.row_stack = np.vstack   # removed in NumPy 2; sigkernel still calls it
import torch

CONFIGS = [(64, 3, 64), (64, 3, 128), (128, 3, 64), (256, 3, 32)]
os.makedirs("data", exist_ok=True)
os.makedirs("out", exist_ok=True)
dev = "cuda"


def med(f, k):
    f(); torch.cuda.synchronize()
    ts = []
    for _ in range(k):
        t = time.perf_counter(); f(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    return sorted(ts)[len(ts) // 2] * 1e3


def peak_mem(f):
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    f(); torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**20


def make_libs():
    libs = {}
    try:
        import pysiglib.torch_api as ta
        libs["pySigLib"] = lambda X, Y: ta.sig_kernel_gram(X, Y, dyadic_order=0)
    except Exception as e:
        print("pySigLib unavailable:", e)
    try:
        import sigkernel
        sk = sigkernel.SigKernel(sigkernel.LinearKernel(), dyadic_order=0)
        libs["sigkernel"] = lambda X, Y: sk.compute_Gram(X, Y, sym=False)   # default max_batch=100 (chunked)
    except Exception as e:
        print("sigkernel unavailable:", e)
    return libs


def main():
    libs = make_libs()
    results = []
    for (N, d, B) in CONFIGS:
        rng = np.random.default_rng(1)
        X = 0.1 * np.cumsum(rng.standard_normal((B, N, d)), axis=1)
        Y = 0.1 * np.cumsum(rng.standard_normal((B, N, d)), axis=1)
        W = rng.standard_normal((B, B))
        tag = f"N{N}_d{d}_B{B}"
        X.tofile(f"data/{tag}_X.bin"); Y.tofile(f"data/{tag}_Y.bin"); W.tofile(f"data/{tag}_W.bin")
        print(f"--- N={N} d={d} B={B} ({B*B} pairs)", flush=True)
        for name, gram in libs.items():
            for dt in (torch.float32, torch.float64):
                r = dict(lib=name, N=N, d=d, B=B, dtype=str(dt).split(".")[-1])
                try:
                    Xt = torch.tensor(X, dtype=dt, device=dev)
                    Yt = torch.tensor(Y, dtype=dt, device=dev)
                    Wt = torch.tensor(W, dtype=dt, device=dev)

                    def fwd():
                        with torch.no_grad():
                            return gram(Xt, Yt)

                    def grad():
                        Xg = Xt.clone().requires_grad_(True)
                        (gram(Xg, Yt) * Wt).sum().backward()
                        return Xg.grad

                    K = fwd(); G = grad()
                    r["fwd_ms"] = med(fwd, 10)
                    r["grad_ms"] = med(grad, 5)
                    r["fwd_mib"] = peak_mem(fwd)
                    r["grad_mib"] = peak_mem(grad)
                    K.double().cpu().numpy().tofile(f"out/{tag}_{name}_{r['dtype']}_K.bin")
                    G.double().cpu().numpy().tofile(f"out/{tag}_{name}_{r['dtype']}_G.bin")
                    print(f"  {name:9s} {r['dtype']:7s} forward {r['fwd_ms']:8.2f} ms ({r['fwd_mib']:7.1f} MiB)"
                          f"   gradient {r['grad_ms']:8.2f} ms ({r['grad_mib']:7.1f} MiB)", flush=True)
                except Exception as e:
                    r["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
                    print(f"  {name:9s} {r['dtype']:7s} FAIL {r['error']}", flush=True)
                results.append(r)
        torch.cuda.empty_cache()
    json.dump(results, open("out/rivals.json", "w"), indent=1)


if __name__ == "__main__":
    main()
