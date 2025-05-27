using Revise, PathSignatures, PythonCall
@py import iisignature
@py import numpy as np
using BenchmarkTools
using StaticArrays

# Define path
f(t) = @SVector [t, 2t]

ts = range(0, stop=1, length=1000)
path = [f(t) for t in ts]
m = 7
d = length(path[1])

# Julia: N-point signature via Chen
sig_julia_func = signature_path(f, ts, m)
sig_julia_vec = signature_path(path, m)

# Python: full path signature
path_mat = reduce(vcat, [f(t)' for t in ts])
path_np = np.asarray(path_mat; order="C")
sig_py = iisignature.sig(path_np, m)
sig_py_julia = pyconvert(Vector{Float64}, sig_py)

# ✅ Validate match
@assert isapprox(sig_julia_func, sig_py_julia; atol=1e-12)
@assert isapprox(sig_julia_vec, sig_py_julia; atol=1e-12)

# 🕒 Benchmark
println("Benchmarking:")
println("Julia (signature_path(f, ts, m))")
@btime signature_path($f, $ts, $m)
println("Julia (signature_path(path, m))")
@btime signature_path($path, $m)
println("Python (iisignature.sig)")
@btime $iisignature.sig($path_np, $m)
