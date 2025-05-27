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

function chen_product(x1::Vector{T}, x2::Vector{T}, d::Int, m::Int) where T
    level_sizes = [d^k for k in 1:m]
    offsets = cumsum([0; level_sizes])

    out = Vector{T}(undef, sum(level_sizes))

    for k in 1:m
        out_k = view(out, offsets[k]+1 : offsets[k+1])
        fill!(out_k, 0)

        for i in 0:k
            # Handle level-0 as scalar 1.0 identity
            a = (i == 0)      ? [one(T)] :
                (i <= m)      ? view(x1, offsets[i]+1:offsets[i+1]) :
                                nothing

            b = (k - i == 0)  ? [one(T)] :
                (k - i <= m)  ? view(x2, offsets[k - i]+1:offsets[k - i + 1]) :
                                nothing

            if a === nothing || b === nothing
                continue
            end

            na, nb = length(a), length(b)
            @inbounds for ai in 1:na
                for bi in 1:nb
                    out_k[(ai - 1) * nb + bi] += a[ai] * b[bi]
                end
            end
        end
    end

    return out
end


function segment_signature_three_points(f, a, b, c, m)
    d = length(f(0.0))

    x_ab = segment_signature(f, a, b, m)
    x_bc = segment_signature(f, b, c, m)
    x_ac = chen_product(x_ab, x_bc, d, m)

    return x_ac
end


# f(t) = [t, 2t]
# a, b = 0.0, 1.0
# m = 20

# sig = segment_signature(f, a, b, m)
# # @show sig
# @show length(sig)  # should be 1 + 2 + 4 + 8 = 15 for d = 2, m = 3

# x0 = f(a)
# x1 = f(b)

# path = vcat(x0', x1')  # shape (2, d), one row per point
# path_np = np.asarray(path; order="C")

# sig_py = iisignature.sig(path_np, m)

# @btime segment_signature($f, $a, $b, $m)
# @btime $iisignature.sig($path_np, $m)


####

f(t) = [t, 2t]
a, b, c = 0.0, 0.4, 1.0
m = 20
d = length(f(0.0))

# Julia: via Chen identity
sig_julia = segment_signature_three_points(f, a, b, c, m)

# Python: full path
x0 = f(a)
x1 = f(b)
x2 = f(c)
path = vcat(x0', x1', x2')   # (3, d)
path_np = np.asarray(path; order="C")
sig_py = iisignature.sig(path_np, m)
sig_py_julia = pyconvert(Vector{Float64}, sig_py)

# ✅ Validate
@assert isapprox(sig_julia, sig_py_julia; atol=1e-12)

# 🕒 Benchmark
println("Benchmarking:")
@btime segment_signature_three_points($f, $a, $b, $c, $m)
@btime $iisignature.sig($path_np, $m)