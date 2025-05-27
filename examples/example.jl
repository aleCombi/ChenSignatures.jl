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

    # Total number of signature terms from level 1 to m:
    total_terms = div(d^(m + 1) - d, d - 1)

    sig = Vector{T}(undef, total_terms)
    idx = 1

    # First level
    curlen = d
    current = view(sig, idx:idx+curlen-1)
    current .= displacement
    idx += curlen
    prevlen = curlen

    for level in 2:m
        curlen = d^level
        current = view(sig, idx:idx+curlen-1)
        _segment_level!(current, displacement, level, view(sig, idx - prevlen:idx - 1))
        idx += curlen
        prevlen = curlen
    end

    return sig
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

function chen_product!(out::Vector{T}, x1::Vector{T}, x2::Vector{T}, d::Int, m::Int) where T
    level_sizes = [d^k for k in 1:m]
    offsets = cumsum([0; level_sizes])

    for k in 1:m
        out_k = view(out, offsets[k]+1 : offsets[k+1])
        fill!(out_k, 0)

        for i in 0:k
            a = i == 0     ? nothing : view(x1, offsets[i]+1:offsets[i+1])
            b = (k - i) == 0 ? nothing : view(x2, offsets[k - i]+1:offsets[k - i + 1])
            na = i == 0     ? 1 : length(a)
            nb = (k - i) == 0 ? 1 : length(b)

            @inbounds @simd for ai in 1:na
                a_val = i == 0 ? one(T) : a[ai]
                @simd for bi in 1:nb
                    b_val = (k - i) == 0 ? one(T) : b[bi]
                    out_k[(ai - 1) * nb + bi] += a_val * b_val
                end
            end
        end
    end
    return out
end

function full_signature(f, ts::Vector{Float64}, m::Int)
    d = length(f(0.0))
    segs = map(1:length(ts)-1) do i
        segment_signature(f, ts[i], ts[i+1], m)
    end

    total_terms = div(d^(m+1) - 1, d - 1)
    result = copy(segs[1])  # will be overwritten in-place

    for i in 2:length(segs)
        buf = similar(result)
        chen_product!(buf, result, segs[i], d, m)
        result .= buf
    end
    return result
end


function segment_signature_three_points(f, a, b, c, m)
    d = length(f(0.0))

    x_ab = segment_signature(f, a, b, m)
    x_bc = segment_signature(f, b, c, m)

    total_terms = div(d^(m + 1) - d, d - 1)
    x_ac = Vector{eltype(x_ab)}(undef, total_terms)

    chen_product!(x_ac, x_ab, x_bc, d, m)

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