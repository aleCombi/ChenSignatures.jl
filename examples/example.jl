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

    sig[idx] = one(T)  # Zeroth level
    idx += 1

    prevlen = 1
    for level in 1:m
        curlen = d^level
        current = view(sig, idx:idx+curlen-1)
        _segment_level!(current, displacement, level, view(sig, idx - prevlen:idx - 1))
        idx += curlen
        prevlen = curlen
    end

    return sig[2:end]
end

function _segment_level!(out::AbstractVector{T}, Δ::AbstractVector{T}, m::Int, previous::AbstractVector{T}) where T
    d, n = length(Δ), length(previous)
    scale = inv(T(m))
    @inbounds for i in 1:d
        for j in 1:n
            out[(i - 1) * n + j] = scale * Δ[i] * previous[j]
        end
    end
end


f(t) = [t, 2t]
a, b = 0.0, 1.0
m = 20

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