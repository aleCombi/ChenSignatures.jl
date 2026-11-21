# ============================================================================
# Log-signatures: Lyndon basis, projection, tensor logarithm
# ============================================================================
#
# logsig(path) = coords(log(sig(path))), with three coordinate systems:
#   :expanded  the full tensor logarithm, flattened like `sig` (levels 1..m)
#   :words     the coefficients of log(sig) at the Lyndon words
#   :lyndon    coordinates in the Lyndon (bracket) basis of the free Lie algebra
#
# For a Lyndon word w with standard factorisation w = uv (v the longest proper
# Lyndon suffix), the basis element is the bracket P_w = [P_u, P_v] = P_u P_v - P_v P_u,
# with P_a = a for letters. P_w = w + (lexicographically larger words of the same
# length), so the matrix L[u, w] = coefficient of the Lyndon word u in P_w is unit
# lower triangular, block diagonal by level, and very sparse (P_w has at most
# 2^(|w|-1) terms). If ℓ are the Lyndon-basis coordinates of log(sig) and u its
# coefficients at the Lyndon words, then u = L ℓ, so ℓ = L \ u.

using SparseArrays

# All Lyndon words over the letters 1..d with length ≤ n, ordered by length and
# lexicographically within a length (Duval's algorithm generates them
# lexicographically across lengths; a stable sort by length regroups them).
function _lyndon_words_sorted(d::Int, n::Int)
    out = Vector{Vector{Int}}()
    w = [-1]                       # 0-based letters
    while !isempty(w)
        w[end] += 1
        push!(out, w .+ 1)
        m = length(w)
        while length(w) < n
            push!(w, w[length(w) - m + 1])
        end
        while !isempty(w) && w[end] == d - 1
            pop!(w)
        end
    end
    sort!(out; by=length, alg=MergeSort)
    return out
end

# 0-based index of a word within its level (last letter fastest).
@inline function _word_index(w::AbstractVector{Int}, d::Int)
    idx = 0
    @inbounds for a in w
        idx = idx * d + (a - 1)
    end
    return idx
end

# Build the Lyndon words and the strictly-lower part of L (the diagonal is 1).
# Returns (words, flat_idx, Lstrict) where flat_idx[j] is the position of word j
# in a flattened signature (levels 1..m, as returned by `sig`).
function _build_lyndon_projection(d::Int, m::Int)
    words = _lyndon_words_sorted(d, m)
    nL = length(words)

    # position of each Lyndon word, looked up by (level, index within level)
    pos = Dict{Vector{Int},Int}()
    for (j, w) in enumerate(words)
        pos[w] = j
    end
    levoff = zeros(Int, m + 1)                 # flat offset of level k = levoff[k]
    for k in 2:m+1
        levoff[k] = levoff[k-1] + d^(k-1)
    end
    lyn_at = [zeros(Int, d^k) for k in 1:m]    # lyn_at[k][idx+1] = Lyndon position or 0
    flat_idx = Vector{Int}(undef, nL)
    for (j, w) in enumerate(words)
        k = length(w); i0 = _word_index(w, d)
        lyn_at[k][i0 + 1] = j
        flat_idx[j] = levoff[k] + i0 + 1
    end

    # Expansions of the brackets P_w as (indices within level, coefficients).
    P_idx = Vector{Vector{Int}}(undef, nL)
    P_val = Vector{Vector{Int}}(undef, nL)     # coefficients are integers
    rows = Int[]; cols = Int[]; vals = Int[]
    acc = Dict{Int,Int}()
    for (j, w) in enumerate(words)
        k = length(w)
        if k == 1
            P_idx[j] = [w[1] - 1]; P_val[j] = [1]
            continue
        end
        # standard factorisation: v = longest proper suffix that is Lyndon
        s = 2
        while !haskey(pos, w[s:end])
            s += 1
        end
        ju = pos[w[1:s-1]]; jv = pos[w[s:end]]
        lu = s - 1; lv = k - lu
        du = d^lu; dv = d^lv
        empty!(acc)
        for (a, ca) in zip(P_idx[ju], P_val[ju]), (b, cb) in zip(P_idx[jv], P_val[jv])
            x = a * dv + b                      # word u·v
            acc[x] = get(acc, x, 0) + ca * cb
            y = b * du + a                      # word v·u
            acc[y] = get(acc, y, 0) - ca * cb
        end
        ks = sort!([key for (key, c) in acc if c != 0])
        P_idx[j] = ks
        P_val[j] = [acc[key] for key in ks]
        for (key, c) in zip(P_idx[j], P_val[j])
            i = lyn_at[k][key + 1]
            i == 0 && continue
            if i == j
                c == 1 || error("Lyndon bracket for $w has leading coefficient $c ≠ 1")
            else
                i > j || error("Lyndon projection not lower triangular at $w")
                push!(rows, i); push!(cols, j); push!(vals, c)
            end
        end
    end
    Lstrict = sparse(rows, cols, Float64.(vals), nL, nL)
    return words, flat_idx, Lstrict
end

# Solve (I + Lstrict) x = u in place (forward substitution, column oriented).
function _lyndon_solve!(x::AbstractVector{T}, Ls::SparseMatrixCSC) where {T}
    rv = rowvals(Ls); nz = nonzeros(Ls)
    @inbounds for j in 1:size(Ls, 2)
        xj = x[j]
        iszero(xj) && continue
        for p in nzrange(Ls, j)
            x[rv[p]] -= T(nz[p]) * xj
        end
    end
    return x
end

# Solve (I + Lstrict)ᵀ y = b in place (back substitution).
function _lyndon_solve_transpose!(y::AbstractVector{T}, Ls::SparseMatrixCSC) where {T}
    rv = rowvals(Ls); nz = nonzeros(Ls)
    @inbounds for j in size(Ls, 2):-1:1
        s = y[j]
        for p in nzrange(Ls, j)
            s -= T(nz[p]) * y[rv[p]]
        end
        y[j] = s
    end
    return y
end

# ============================================================================
# Public API
# ============================================================================

"""
    BasisCache

Precomputed Lyndon basis for [`logsig`](@ref), created by [`prepare`](@ref).
Its fields are an implementation detail.
"""
struct BasisCache{T}
    d::Int
    m::Int
    lynds::Vector{Word}          # Lyndon words, by length then lexicographically
    idx::Vector{Int}             # position of each Lyndon word in a flattened signature
    L::SparseMatrixCSC{T,Int}    # strictly lower part of the unit lower-triangular projection
end

"""
    prepare(d::Int, m::Int) -> BasisCache

Precompute the Lyndon basis for log-signatures of `d`-dimensional paths up to level
`m`. Pass the result to [`logsig`](@ref); reuse it for all paths with the same `(d, m)`.
This is cheap (milliseconds for typical sizes) and the cache is small.

# Example
```julia
basis = prepare(3, 4)
logsig(randn(100, 3), basis)          # 32 coordinates
```

See also: [`logsig`](@ref)
"""
function prepare(d::Int, m::Int)
    d >= 1 || throw(ArgumentError("Path dimension must be at least 1, got d=$d"))
    m >= 1 || throw(ArgumentError("Signature level must be at least 1, got m=$m"))
    words, idx, Ls = _build_lyndon_projection(d, m)
    return BasisCache{Float64}(d, m, [Word(w) for w in words], idx, Ls)
end

const _LOGSIG_COORDS = (:lyndon, :words, :expanded)

_check_coords(coords::Symbol) = coords in _LOGSIG_COORDS ||
    throw(ArgumentError("coords must be one of $(_LOGSIG_COORDS), got :$coords"))

_logsig_length(basis::BasisCache, coords::Symbol) =
    coords === :expanded ? sum(basis.d^k for k in 1:basis.m) : length(basis.idx)

# Scratch for one log-signature: signature, two power buffers, log, flat buffer.
struct LogSigWorkspace{T,D,M}
    sws::SignatureWorkspace{T,D,M}
    S::Tensor{T,D,M}
    P::Tensor{T,D,M}
    Q::Tensor{T,D,M}
    Lg::Tensor{T,D,M}
end
LogSigWorkspace{T,D,M}() where {T,D,M} = LogSigWorkspace{T,D,M}(
    SignatureWorkspace{T,D,M}(), Tensor{T,D,M}(), Tensor{T,D,M}(), Tensor{T,D,M}(), Tensor{T,D,M}())

# Q = P ⊗ X restricted to levels > kmin, where P is zero below level kmin and
# X has no level-0 term (X = S - 1). Levels of Q below kmin+1 are zeroed.
function _mul_from_level!(Q::Tensor{T,D,M}, P::Tensor{T,D,M}, X::Tensor{T,D,M},
                          kmin::Int) where {T,D,M}
    q, p, x = Q.coeffs, P.coeffs, X.coeffs
    off = Q.offsets
    fill!(q, zero(T))
    @inbounds for j in kmin+1:M
        qs = off[j+1]
        for i in kmin:j-1                      # P_i ⊗ X_{j-i}
            ps = off[i+1]; xs = off[j-i+1]
            blen = D^(j-i)
            for a in 1:D^i
                pa = p[ps + a]
                iszero(pa) && continue
                row = qs + (a - 1) * blen
                @simd for b in 1:blen
                    q[row + b] = muladd(pa, x[xs + b], q[row + b])
                end
            end
        end
    end
    return Q
end

# out = log(S) = Σ_{k=1}^M (-1)^{k+1} X^k / k with X = S - 1, using that X^k
# vanishes below level k. Reads S only at levels ≥ 1.
function _log_levels!(out::Tensor{T,D,M}, S::Tensor{T,D,M},
                      P::Tensor{T,D,M}, Q::Tensor{T,D,M}) where {T,D,M}
    o, s = out.coeffs, S.coeffs
    off = out.offsets
    fill!(o, zero(T))
    @inbounds for i in off[2]+1:off[M+2]
        o[i] = s[i]                            # k = 1 term
    end
    M == 1 && return out
    copyto!(P.coeffs, S.coeffs)
    P.coeffs[off[1]+1] = zero(T)               # P = X
    for k in 2:M
        _mul_from_level!(Q, P, S, k - 1)       # Q = X^k (S's level 0 is never read)
        c = T(isodd(k) ? 1 : -1) / T(k)
        q = Q.coeffs
        @inbounds for i in off[k+1]+1:off[M+2]
            o[i] = muladd(c, q[i], o[i])
        end
        P, Q = Q, P
    end
    return out
end

# Write the coordinates of log(sig(path)) into `y`.
# If `sflat` is given, the flattened signature (levels 1..M) is also stored there,
# so a later reverse pass does not need to recompute it.
function _logsig!(y::AbstractVector, path::AbstractMatrix{T}, basis::BasisCache,
                  coords::Symbol, ws::LogSigWorkspace{T,D,M}, sflat=nothing) where {T,D,M}
    signature_path!(ws.S, path, ws.sws)
    if sflat !== nothing
        copyto!(sflat, 1, ws.S.coeffs, ws.S.offsets[2] + 1, length(sflat))
    end
    _log_levels!(ws.Lg, ws.S, ws.P, ws.Q)
    _write_coords!(y, ws.Lg, basis, coords)
    return y
end

function _write_coords!(y::AbstractVector{T}, Lg::Tensor{T,D,M}, basis::BasisCache,
                        coords::Symbol) where {T,D,M}
    c = Lg.coeffs; s1 = Lg.offsets[2]
    # flat position f (levels 1..M, as in `sig`) lives at c[s1 + f]: the levels
    # are contiguous after the level-1 offset.
    if coords === :expanded
        copyto!(y, 1, c, s1 + 1, length(y))
    else
        @inbounds for (j, f) in enumerate(basis.idx)
            y[j] = c[s1 + f]
        end
        coords === :lyndon && _lyndon_solve!(y, basis.L)
    end
    return y
end

function _check_logsig_input(N::Int, D::Int, basis::BasisCache)
    N >= 2 || throw(ArgumentError("Path must have at least 2 points, got N=$N"))
    D == basis.d || throw(ArgumentError(
        "Dimension mismatch: path has dimension $D but basis expects $(basis.d)"))
end

"""
    logsig(path::AbstractMatrix, basis::BasisCache; coords=:lyndon) -> Vector
    logsig(paths::AbstractArray{T,3}, basis::BasisCache; coords=:lyndon, threaded=true) -> Matrix

Log-signature of an `N×d` path (or of each path in an `N×d×B` batch, returning one
column per path), truncated at level `m`, where `basis = prepare(d, m)`.

`coords` selects the coordinates:
- `:lyndon` (default): coordinates in the Lyndon basis of the free Lie algebra, the
  canonical compressed log-signature (iisignature's default; pySigLib `method=2`).
- `:words`: the coefficients of the tensor logarithm at the Lyndon words. Same length
  and same information as `:lyndon` (an invertible linear change of coordinates), and
  slightly cheaper (Signatory's and pySigLib's default).
- `:expanded`: the full tensor logarithm, flattened like [`sig`](@ref).

All three are differentiable (see the "How it works" page).

# Example
```julia
basis = prepare(3, 4)
path = randn(100, 3)
logsig(path, basis)                      # 32 Lyndon-basis coordinates
logsig(path, basis; coords=:words)       # 32 coordinates
logsig(path, basis; coords=:expanded)    # 120 = 3 + 9 + 27 + 81
```

See also: [`prepare`](@ref), [`sig`](@ref)
"""
function logsig(path::AbstractMatrix{T}, basis::BasisCache; coords::Symbol=:lyndon) where {T}
    N, D = size(path)
    _check_logsig_input(N, D, basis)
    _check_coords(coords)
    return _logsig_single(path, basis, coords, Val(D), Val(basis.m))
end

function _logsig_single(path::AbstractMatrix{T}, basis, coords, ::Val{D}, ::Val{M};
                        keep_sig::Bool=false) where {T,D,M}
    y = Vector{T}(undef, _logsig_length(basis, coords))
    sflat = keep_sig ? Vector{T}(undef, sum(D^k for k in 1:M)) : nothing
    _logsig!(y, path, basis, coords, LogSigWorkspace{T,D,M}(), sflat)
    return keep_sig ? (y, sflat) : y
end

function logsig(paths::AbstractArray{T,3}, basis::BasisCache;
                coords::Symbol=:lyndon, threaded::Bool=true) where {T}
    N, D, B = size(paths)
    _check_logsig_input(N, D, basis)
    B >= 1 || throw(ArgumentError("Batch size must be at least 1, got B=$B"))
    _check_coords(coords)
    return _logsig_batch(paths, basis, coords, threaded, Val(D), Val(basis.m))
end

function _logsig_batch(paths::AbstractArray{T,3}, basis, coords, threaded,
                       ::Val{D}, ::Val{M}; keep_sig::Bool=false) where {T,D,M}
    B = size(paths, 3)
    Y = Matrix{T}(undef, _logsig_length(basis, coords), B)
    Sg = keep_sig ? Matrix{T}(undef, sum(D^k for k in 1:M), B) : nothing
    sv(b) = Sg === nothing ? nothing : view(Sg, :, b)
    if threaded
        Threads.@threads for b in 1:B
            local wsb = LogSigWorkspace{T,D,M}()
            _logsig!(@view(Y[:, b]), @view(paths[:, :, b]), basis, coords, wsb, sv(b))
        end
    else
        ws = LogSigWorkspace{T,D,M}()
        for b in 1:B
            _logsig!(@view(Y[:, b]), @view(paths[:, :, b]), basis, coords, ws, sv(b))
        end
    end
    return keep_sig ? (Y, Sg) : Y
end

# ============================================================================
# Reverse pass
# ============================================================================
#
# log(S) = Σ_{k=1}^M c_k X^k, c_k = (-1)^{k+1}/k, X = S - 1, with powers
# P_1 = X, P_k = P_{k-1} ⊗ X. Given Ḡ = ∂L/∂log(S):
#   P̄_M = c_M Ḡ
#   for k = M..2:   P̄_{k-1} = c_{k-1} Ḡ + P̄_k ◁ X      (P̄_k ◁ X)(u) = Σ_v P̄_k(uv) X(v)
#                   X̄      += P_{k-1} ▷ P̄_k            (P ▷ P̄)(v)  = Σ_u P(u) P̄(uv)
#   X̄ += P̄_1,  and ∂L/∂S = X̄ (levels ≥ 1).
# P_k vanishes below level k, so only levels ≥ k of P̄_k matter. Both
# contractions read the same rows of P̄_k, so they share one pass.

struct LogSigBackWorkspace{T,D,M}
    fwd::LogSigWorkspace{T,D,M}
    pows::Vector{Tensor{T,D,M}}      # pows[k] = X^k for k = 2..M-1 (pows[1] unused: X is S)
    Gl::Tensor{T,D,M}                # ∂L/∂log(S)
    Pb::Tensor{T,D,M}
    Pb2::Tensor{T,D,M}
    Xb::Tensor{T,D,M}                # ∂L/∂S
    bws::SigBackwardWorkspace{T,D,M}
end
LogSigBackWorkspace{T,D,M}() where {T,D,M} = LogSigBackWorkspace{T,D,M}(
    LogSigWorkspace{T,D,M}(), [Tensor{T,D,M}() for _ in 1:max(M - 1, 1)],
    Tensor{T,D,M}(), Tensor{T,D,M}(), Tensor{T,D,M}(), Tensor{T,D,M}(),
    SigBackwardWorkspace{T,D,M}())

# Pdst_i += Σ_j P̄_{i+j} X_j  and  X̄_j += Σ_i P_i P̄_{i+j}, over i ≥ kmin, j ≥ 1, i+j ≤ M.
function _log_step_adjoint!(Pdst::Tensor{T,D,M}, Xb::Tensor{T,D,M}, Pb::Tensor{T,D,M},
                            X::Tensor{T,D,M}, P::Tensor{T,D,M}, kmin::Int) where {T,D,M}
    pd, xb, pb, x, p = Pdst.coeffs, Xb.coeffs, Pb.coeffs, X.coeffs, P.coeffs
    off = X.offsets
    @inbounds for i in kmin:M-1, j in 1:M-i
        ps = off[i+1]; xs = off[j+1]; cs = off[i+j+1]
        blen = D^j
        for a in 1:D^i
            row = cs + (a - 1) * blen
            pa = p[ps + a]
            acc = zero(T)
            @simd for b in 1:blen
                cb = pb[row + b]
                acc = muladd(cb, x[xs + b], acc)
                xb[xs + b] = muladd(pa, cb, xb[xs + b])
            end
            pd[ps + a] += acc
        end
    end
    return nothing
end

# Given ws.fwd.S = S, compute ws.Xb = ∂L/∂S from ws.Gl = ∂L/∂log(S).
function _log_pullback!(ws::LogSigBackWorkspace{T,D,M}) where {T,D,M}
    S = ws.fwd.S
    Gl, Xb = ws.Gl, ws.Xb
    fill!(Xb.coeffs, zero(T))
    c(k) = T(isodd(k) ? 1 : -1) / T(k)
    if M == 1
        copyto!(Xb.coeffs, Gl.coeffs)
        Xb.coeffs[Xb.offsets[1] + 1] = zero(T)
        return Xb
    end
    # powers X^k, k = 2..M-1 (X^1 is S itself: level 0 is never read)
    prev = S
    for k in 2:M-1
        _mul_from_level!(ws.pows[k], prev, S, k - 1)
        prev = ws.pows[k]
    end
    Pb, Pb2 = ws.Pb, ws.Pb2
    Pb.coeffs .= c(M) .* Gl.coeffs
    for k in M:-1:2
        Pb2.coeffs .= c(k - 1) .* Gl.coeffs
        Pk1 = k == 2 ? S : ws.pows[k-1]
        _log_step_adjoint!(Pb2, Xb, Pb, S, Pk1, k - 1)
        Pb, Pb2 = Pb2, Pb
    end
    Xb.coeffs .+= Pb.coeffs                    # X̄ += P̄_1
    Xb.coeffs[Xb.offsets[1] + 1] = zero(T)
    return Xb
end

# ∂L/∂path for one path, given ȳ = ∂L/∂logsig. Uses the stored flattened
# signature `sflat` if given, otherwise recomputes it.
function _logsig_vjp!(xbar::AbstractMatrix{T}, path::AbstractMatrix{T}, ybar::AbstractVector,
                      basis::BasisCache, coords::Symbol,
                      ws::LogSigBackWorkspace{T,D,M}, sflat=nothing) where {T,D,M}
    S = ws.fwd.S
    if sflat === nothing
        signature_path!(S, path, ws.fwd.sws)
    else
        _unflatten!(S, sflat)
    end

    # ȳ -> ∂L/∂log(S) as a dense tensor
    g = ws.Gl.coeffs; s1 = ws.Gl.offsets[2]
    fill!(g, zero(T))
    if coords === :expanded
        @inbounds for f in eachindex(ybar)
            g[s1 + f] = T(ybar[f])
        end
    else
        v = ws.fwd.Lg.coeffs                   # reuse as a length-#Lyndon buffer
        nL = length(basis.idx)
        @inbounds for j in 1:nL
            v[j] = T(ybar[j])
        end
        coords === :lyndon && _lyndon_solve_transpose!(view(v, 1:nL), basis.L)
        @inbounds for (j, f) in enumerate(basis.idx)
            g[s1 + f] = v[j]
        end
    end

    _log_pullback!(ws)
    return sig_pullback!(xbar, path, S, ws.Xb, ws.bws)   # rewinds S
end

function _logsig_vjp(path::AbstractMatrix{T}, ybar, basis, coords,
                     ::Val{D}, ::Val{M}, sflat=nothing) where {T,D,M}
    xbar = similar(path, T, size(path))
    return _logsig_vjp!(xbar, path, ybar, basis, coords, LogSigBackWorkspace{T,D,M}(), sflat)
end

function _logsig_vjp_batch(paths::AbstractArray{T,3}, Ybar::AbstractMatrix, basis, coords,
                           threaded::Bool, ::Val{D}, ::Val{M}, Sg=nothing) where {T,D,M}
    sv(b) = Sg === nothing ? nothing : view(Sg, :, b)
    N, _, B = size(paths)
    X̄ = zeros(T, N, D, B)
    if threaded
        Threads.@threads for b in 1:B
            local wsb = LogSigBackWorkspace{T,D,M}()
            _logsig_vjp!(@view(X̄[:, :, b]), @view(paths[:, :, b]), @view(Ybar[:, b]),
                         basis, coords, wsb, sv(b))
        end
    else
        ws = LogSigBackWorkspace{T,D,M}()
        for b in 1:B
            _logsig_vjp!(@view(X̄[:, :, b]), @view(paths[:, :, b]), @view(Ybar[:, b]),
                         basis, coords, ws, sv(b))
        end
    end
    return X̄
end
