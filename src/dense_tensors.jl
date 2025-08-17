abstract type AbstractTensor end

struct Tensor{T} <: AbstractTensor
    coeffs::StridedVector{T}
    dim::Int
    level::Int
    offsets::Vector{Int}
    
    function Tensor(coeffs::StridedVector{T}, dim::Int, level::Int) where {T}
        offsets = level_starts0(dim, level)
        new{T}(coeffs, dim, level, offsets)
    end

    function Tensor{T}(dim::Int, level::Int) where {T}
        total_terms = div(dim^(level + 1) - dim, dim - 1)
        coeffs = Vector{T}(undef, total_terms)
        offsets = level_starts0(dim, level)
        new{T}(coeffs, dim, level, offsets)
    end
end

## Tensor interface

Base.length(ts::Tensor{T}) where T = length(ts.coeffs)
Base.getindex(ts::Tensor{T}, i::Int) where T = ts.coeffs[i]
Base.show(io::IO, ts::Tensor{T}) where T =
    print(io, "TensorSeries(dim=$(ts.dim), level=$(ts.level), length=$(length(ts)))")

    # Element type of a TensorSeries
Base.eltype(::Tensor{T}) where {T} = T

# Allocate a new TensorSeries with the same "shape" (dim, level)
Base.similar(ts::Tensor{T}) where {T} = Tensor{T}(ts.dim, ts.level)

# Same, but change element type (handy for promoting to BigFloat, Dual, etc.)
Base.similar(ts::Tensor{T}, ::Type{S}) where {T,S} = Tensor{S}(ts.dim, ts.level)

# Allocate-and-copy convenience
Base.copy(ts::Tensor{T}) where {T} = Tensor(copy(ts.coeffs), ts.dim, ts.level)

# In-place copy with shape check (future-proof for pipelines)
function Base.copy!(dest::Tensor, src::Tensor)
    @assert dest.dim == src.dim && dest.level == src.level "TensorSeries shape mismatch"
    copyto!(dest.coeffs, src.coeffs)
    return dest
end

dim(ts::Tensor)   = ts.dim
level(ts::Tensor) = ts.level
coeffs(ts::Tensor) = ts.coeffs

# --- core kernel: tensor_exponential!(out, x, m) ---
# Computes: out = Σ_{k=1}^m (x^{⊗k} / k!)
@inline function exp!(
    out::Tensor{T}, x::StridedVector{T}
) where {T}

# level 1
    idx    = 1
    d = out.dim
    curlen = d
    m = out.level
    copyto!(out.coeffs, idx, x, 1, d)
    prev_start = idx
    idx += curlen

    # quick return
    if m == 1
        @assert idx - 1 == length(out)
        return nothing
    end

    # levels 2..m
    @inbounds for level in 2:m
        prev_len  = curlen
        curlen   *= d
        cur_start = idx
        scale = inv(T(level))
        _segment_level_offsets!(out.coeffs, x, scale,
                                prev_start, prev_len, cur_start)
        prev_start = cur_start
        idx += curlen
    end 

    # cheaper postcondition: avoids pow/div
    @assert idx - 1 == length(out)
    return nothing
end


function mul(a::AbstractTensor, b::AbstractTensor)
    # allocate same "shape" as a, using eltype promotion
    dest = similar(a, promote_type(eltype(a), eltype(b)))
    return mul!(dest, a, b)
end

⊗(a::AbstractTensor, b::AbstractTensor) = mul(a, b)

@inline function mul!(
    out_tensor::Tensor{T}, x1_tensor::Tensor{T}, x2_tensor::Tensor{T}
) where {T}
    out, x1, x2, m = out_tensor.coeffs, x1_tensor.coeffs, x2_tensor.coeffs, out_tensor.level
    offsets = out_tensor.offsets
    @inbounds for k in 1:m
        out_start = offsets[k] + 1
        out_len   = offsets[k+1] - offsets[k]

        # ---- init with i = 0 term: out_k = 1 ⊗ x2_k = x2_k (full-block copy)
        b_start = offsets[k]                  # 0-based in offsets
        copyto!(out, out_start, x2, b_start + 1, out_len)

        # ---- middle terms: i = 1 .. k-1  (outer products, +=)
        for i in 1:(k-1)
            a_start = offsets[i]
            a_len   = offsets[i+1] - offsets[i]
            b_start = offsets[k - i]
            b_len   = offsets[k - i + 1] - offsets[k - i]

            @avx for ai in 1:a_len, bi in 1:b_len
                row0 = out_start + (ai - 1) * b_len - 1
                out[row0 + bi] = muladd(x1[a_start + ai], x2[b_start + bi], out[row0 + bi])
            end
        end

        # ---- endpoint term: i = k  (x1_k ⊗ 1) → contiguous add
        # This is a_len = out_len, b_len = 1 → flatten to one loop.
        a_start = offsets[k]
        @avx for j in 1:out_len
            out[out_start + j - 1] += x1[a_start + j]
        end
    end
    return out
end

@inline function _segment_level_offsets!(
    out::StridedVector{T}, Δ::StridedVector{T}, scale::T,
    prev_start::Int, prev_len::Int, cur_start::Int
) where {T}
    d = length(Δ)
    @inbounds @avx for i in 1:d, j in 1:prev_len
        s = scale * Δ[i]
        base = cur_start + (i - 1) * prev_len - 1
        out[base + j] = s * out[prev_start + j - 1]
    end
    return nothing
end

"""
    level_starts0(dim::Int, level::Int) -> Vector{Int}

Return a vector `s` of length `level + 1` with **0-based** start indices for the
flattened tensor-series blocks of each level `k = 1..level` in column-major layout.

Invariant:
- `s[1] == 0`
- `s[k+1] - s[k] == dim^k` (length of level-`k` block)
- `s[level+1] == (dim^(level+1) - dim) ÷ (dim - 1)` (total length)

Example:
```julia
julia> level_starts0(2, 3)
4-element Vector{Int}: [0, 2, 6, 14]   # sizes 2,4,8 → cumulative 0,2,6,14
"""
function level_starts0(d, m)
    offsets = Vector{Int}(undef, m + 1)
    offsets[1] = 0
    len = d
    @inbounds for k in 1:m
        offsets[k+1] = offsets[k] + len
        len *= d
    end

    return offsets
end
