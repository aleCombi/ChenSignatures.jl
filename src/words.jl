# ============================================================================
# Word labels and indices for `sig` and `logsig` outputs
# ============================================================================
#
# `sig(path, m)` returns levels 1..m one after another; within level k the
# word w_1 ... w_k (letters 1..d) sits at 0-based offset Σ_i (w_i - 1) d^(k-i),
# i.e. lexicographic order with the last letter varying fastest.

"""
    sig_words(d::Int, m::Int) -> Vector{Vector{Int}}

The word labelling each coordinate of `sig(path, m)` for a `d`-dimensional path:
all words over the letters `1:d` of length `1` to `m`, by length and then
lexicographically. `sig_words(d, m)[i]` is the word of `sig(path, m)[i]`.

# Example
```julia
sig_words(2, 2)        # [[1], [2], [1, 1], [1, 2], [2, 1], [2, 2]]

# coordinates of words ending in letter 2
idx = findall(w -> last(w) == 2, sig_words(2, 3))
sig(path, 3)[idx]
```

See also: [`sig_index`](@ref), [`logsig_words`](@ref)
"""
function sig_words(d::Int, m::Int)
    d >= 1 || throw(ArgumentError("Path dimension must be at least 1, got d=$d"))
    m >= 1 || throw(ArgumentError("Signature level must be at least 1, got m=$m"))
    words = Vector{Vector{Int}}(undef, sum(d^k for k in 1:m))
    i = 0
    for k in 1:m, x in 0:d^k-1
        w = Vector{Int}(undef, k)
        r = x
        for j in k:-1:1
            w[j] = r % d + 1
            r ÷= d
        end
        words[i += 1] = w
    end
    return words
end

"""
    sig_index(word, d::Int) -> Int

The position of `word` (letters in `1:d`, length ≥ 1) in the output of `sig` for a
`d`-dimensional path, at any truncation level `m ≥ length(word)`. The inverse of
indexing [`sig_words`](@ref): `sig_words(d, m)[sig_index(w, d)] == w`.

# Example
```julia
sig_index([1, 2], 2)   # 4: after [1], [2], [1, 1]
```
"""
function sig_index(word::AbstractVector{<:Integer}, d::Int)
    k = length(word)
    k >= 1 || throw(ArgumentError("word must have at least one letter"))
    idx = 0
    for a in word
        1 <= a <= d || throw(ArgumentError("letter $a is outside 1:$d"))
        idx = idx * d + (a - 1)
    end
    return (d == 1 ? k - 1 : (d^k - d) ÷ (d - 1)) + idx + 1
end

"""
    logsig_words(basis::BasisCache) -> Vector{Vector{Int}}

The Lyndon word labelling each coordinate of `logsig(path, basis)` with
`coords = :lyndon` or `:words`, in output order (by length, then
lexicographically). With `coords = :expanded` the labels are
`sig_words(basis.d, basis.m)`.

See also: [`sig_words`](@ref), [`prepare`](@ref)
"""
logsig_words(basis::BasisCache) = [copy(w.indices) for w in basis.lynds]
