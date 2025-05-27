module PathSignatures

export signature_path

function signature_words(level::Int, dim::Int)
    Iterators.product(ntuple(_ -> 1:dim, level)...)
end

function all_signature_words(max_level::Int, dim::Int)
    Iterators.flatten(signature_words(ℓ, dim) for ℓ in 1:max_level)
end

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

function signature_path(f, ts::AbstractVector{<:Real}, m::Int)
    d = length(f(ts[1]))
    T = eltype(f(ts[1]))

    # Number of signature terms (excluding level 0)
    total_terms = div(d^(m + 1) - d, d - 1)

    # Compute signature of first segment
    sig = segment_signature(f, ts[1], ts[2], m)

    # Use a scratch buffer for chen_product!
    buf = Vector{T}(undef, total_terms)

    for i in 2:length(ts)-1
        sig_next = segment_signature(f, ts[i], ts[i+1], m)
        chen_product!(buf, sig, sig_next, d, m)
        sig, buf = buf, sig  # swap buffers
    end

    return sig
end

end # module PathSignatures
