# sigcheck.jl
#
# Usage:
#   julia --project=. sigcheck.jl N d m path_kind
#
# Prints the signature coefficients (in the iisignature-compatible layout)
# as a single space-separated line on stdout.

using StaticArrays
using PathSignatures

if length(ARGS) < 4
    error("Usage: julia sigcheck.jl N d m path_kind")
end

N = parse(Int, ARGS[1])
d = parse(Int, ARGS[2])
m = parse(Int, ARGS[3])
path_kind = Symbol(ARGS[4])  # "linear" or "sin"

# -------- path generators (same as your old benchmark) --------

function make_path_linear(d::Int, N::Int)
    ts = range(0.0, stop=1.0, length=N)
    [SVector{d,Float64}(ntuple(i -> (i == 1 ? t : 2t), d)) for t in ts]
end

function make_path_sin(d::Int, N::Int)
    ts = range(0.0, stop=1.0, length=N)
    ω = 2π
    [SVector{d,Float64}(ntuple(i -> sin(ω * i * t), d)) for t in ts]
end

function make_path(d::Int, N::Int, kind::Symbol)
    if kind === :linear
        return make_path_linear(d, N)
    elseif kind === :sin
        return make_path_sin(d, N)
    else
        error("Unknown path_kind: $kind (expected :linear or :sin)")
    end
end

# -------- compute signature (same as old benchmark) --------

path = make_path(d, N, path_kind)
tensor_type = PathSignatures.Tensor{eltype(path[1])}
sig = signature_path(tensor_type, path, m)

# DEBUG: offsets, to stderr
println(stderr, "=== sigcheck debug ===")
println(stderr, "N=$N, d=$d, m=$m, kind=$path_kind")
println(stderr, "length(sig.coeffs) = ", length(sig.coeffs))
println(stderr, "sig.offsets        = ", sig.offsets)
println(stderr, "======================")

# IMPORTANT: use the same slice you used before when comparing to iisignature:
# sig_julia.coeffs[sig_julia.offsets[2]+1:end]
sig_vec = sig.coeffs[sig.offsets[2]+1:end]

# ---------- NUMERIC OUTPUT TO STDOUT ONLY ----------
for (i, v) in enumerate(sig_vec)
    print(v)
    if i < length(sig_vec)
        print(' ')
    end
end
println()
