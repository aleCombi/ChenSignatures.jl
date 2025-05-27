using Revise, PathSignatures, PythonCall
@py import iisignature
@py import numpy as np
using BenchmarkTools

PathSignatures.signature_words(2,2) |> collect
PathSignatures.all_signature_words(2,2) |> collect

function segment_signature(f, a, b, m)
    displacement = f(b) - f(a)
    d = length(displacement)
    T = eltype(displacement)

    # Total number of signature terms up to level m:
    total_terms = div(d^(m + 1) - 1, d - 1)

    sig = Vector{T}(undef, total_terms)
    idx = 1

    # Zeroth level is always 1 (the empty word)
    sig[idx] = one(T)
    idx += 1

    previous = sig[1:1]

    for level in 1:m
        current = _segment_level(displacement, level, previous)
        sig[idx:idx+length(current)-1] = current
        previous = view(sig, idx:idx+length(current)-1)
        idx += length(current)
    end

    return sig[2:end]
end

function _segment_level(displacement::AbstractVector{T}, m::Int, previous::AbstractVector{T}) where T
    return vec((displacement / m) * previous')
end

f(t) = [t, 2t]
a, b = 0.0, 1.0
m = 10

sig = segment_signature(f, a, b, m)
# @show sig
@show length(sig)  # should be 1 + 2 + 4 + 8 = 15 for d = 2, m = 3

x0 = f(a)
x1 = f(b)

path = vcat(x0', x1')  # shape (2, d), one row per point
path_np = np.asarray(path; order="C")

sig_py = iisignature.sig(path_np, m)

@btime segment_signature($f, $a, $b, $m)
@btime $iisignature.sig($path_np, $m)