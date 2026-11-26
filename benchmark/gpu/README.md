# GPU benchmarks

`sig_gram` on GPU arrays (`ext/ChenSignaturesKernelAbstractionsExt.jl`), run on CUDA.

- `bench.jl` checks the results against the CPU implementation, then times the forward
  pass and the gradient through Zygote, Enzyme and Mooncake.

      julia --project=benchmark/gpu -t 8 benchmark/gpu/bench.jl

- `rivals/` compares against pySigLib and sigkernel on identical data. Set up the
  Python environment as described in `rivals/bench_rivals.py`, then:

      cd benchmark/gpu/rivals
      .venv/bin/python bench_rivals.py
      julia --project=.. -t 8 compare_rivals.jl

## Results

RTX 2060 Max-Q (6 GB), 8 CPU threads, 2026-11-26. Linear static kernel, dyadic
order 0, d = 3. Loss `sum(W .* sig_gram(X, Y))`, gradient with respect to X. Times
in ms, Float32:

| Case | ChenSignatures GPU fwd / grad | pySigLib 4.1 CUDA fwd / grad | sigkernel fwd / grad | ChenSignatures CPU fwd / grad |
|---|---|---|---|---|
| N=64, 4096 pairs | 1.7 / 5.9 | 10.2 / 34.9 | 19 / 94 | 7.3 / 64 |
| N=64, 16384 pairs | 6.0 / 21 | 34.8 / 138 | 78 / 382 | 17 / 171 |
| N=128, 4096 pairs | 5.8 / 19 | 53.0 / 203 | out of memory | 32 / 185 |
| N=256, 1024 pairs | 6.1 / 21 | 70.7 / 288 | out of memory | 51 / 266 |

Memory: the forward pass needs under 1 MiB of GPU memory; the gradient stores the PDE
grid, 70–280 MiB in Float32 for the cases above. pySigLib uses 0.4–3.9 GB.

Accuracy against the CPU in Float64: ChenSignatures GPU 1e-14; pySigLib 1e-14.
In Float32 pySigLib is more accurate (2e-7 against our 5e-6). sigkernel's forward is
right, but its gradient of the Gram matrix with `sym=False` is not (relative error
0.2–0.4 in Float64).
