# The PDE signature kernel Gram on GPU arrays: `sig_gram(X, Y)` and `sig_gram(X)` for
# any `AbstractGPUArray` with a KernelAbstractions backend (CUDA.jl, AMDGPU.jl, ...),
# with a ChainRules rrule that runs a handwritten adjoint kernel. Zygote uses the rrule
# directly, Enzyme through ChenSignaturesEnzymeExt, Mooncake through
# ChenSignaturesGPUMooncakeExt.
#
# Supported: the linear and RBF static kernels, any `dyadic_order`. Others raise.
#
# Grid of one pair: K(r, c), r = 0..n, c = 0..m (n, m increments of x, y after the
# dyadic refinement, which splits each increment into 2^λ equal pieces),
#   K(r, c) = (K(r, c-1) + K(r-1, c)) A − K(r-1, c-1) B,   r, c ≥ 1,  K = 1 on the boundary,
#   A = 1 + κ/2 + κ²/12,  B = 1 − κ²/12,  κ(r, c) = ⟨Δx_r, Δy_c⟩.
# One workgroup per pair, one thread per grid row (thread tid owns row r = tid - 1); the
# anti-diagonals t = r + c are swept in order, the three live ones kept in shared memory.
# The rows are the shorter path, so n + 1 must fit in one workgroup (≤ 1024 threads).
#
# For the gradient the forward pass stores the grid, packed diagonal by diagonal (one
# pair's grid is exactly (n+1)(m+1) values, consecutive threads write consecutive
# addresses). The adjoint kernel sweeps the diagonals backwards: K̄(r, c) is gathered
# from the successors (r, c+1), (r+1, c), (r+1, c+1), then
#   κ̄ = K̄ (∂A/∂κ (K(r,c-1) + K(r-1,c)) − ∂B/∂κ K(r-1,c-1)),
# x̄_r += κ̄ Δy_c in registers and ȳ_c += κ̄ Δx_r in shared memory (on one diagonal
# every thread has its own c, so there are no write conflicts).
module ChenSignaturesKernelAbstractionsExt

using ChenSignatures
using ChenSignatures: sig_gram, StaticKernel, LinearKernel, RBFKernel
using ChainRulesCore
using KernelAbstractions
using KernelAbstractions: @kernel, @index, @localmem, @private, @synchronize, @Const, get_backend
using GPUArraysCore: AbstractGPUArray

const KA = KernelAbstractions

# ---------------------------------------------------------------------------
# Device helpers. Kept outside the kernels: closures inside a @kernel body box the
# variables that KA's rewrite reassigns.
# ---------------------------------------------------------------------------

@inline _dot_row(xr, dY, c, b, ::Val{D}) where {D} =
    sum(ntuple(k -> @inbounds(xr[k] * dY[c, k, b]), Val(D)))

@inline _slot(t) = mod(t, 3) + 1
@inline _coefA(κ) = 1 + κ / 2 + κ * κ / 12
@inline _coefB(κ) = 1 - κ * κ / 12

# Packed grid: diagonal t holds the cells r = rlo(t)..min(n, t), starting at off[t+1].
@inline _rlo(t, m) = max(0, t - m)
@inline _valid(r, t, n, m) = (r <= n) & (0 <= t - r <= m)
@inline _gidx(off, r, t, m) = @inbounds(off[t + 1]) + (r - _rlo(t, m)) + 1

# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

# On KA's CPU backend a per-thread value survives a @synchronize only if it is an
# @index result (recomputed) or lives in a @private array (registers on a GPU): the
# row r is recomputed after each barrier, the pair indices and row vectors are @private.
@inline _pair_a(p, Bx) = (p - 1) % Bx + 1
@inline _pair_b(p, Bx) = (p - 1) ÷ Bx + 1
@inline _pair_base(p, n, m) = (p - 1) * (n + 1) * (m + 1)

@kernel function _gram_fwd!(Kout::AbstractMatrix{T}, Kd, @Const(off), @Const(dX), @Const(dY),
                            n, m, Bx, ::Val{W}, ::Val{D}, ::Val{STORE}) where {T, W, D, STORE}
    tid = @index(Local, Linear)
    p = @index(Group, Linear)
    S = @localmem T (W, 3)
    xr = @private T (D,)
    ib = @private Int (2,)                  # pair column b, grid offset of the pair
    r = tid - 1
    a = _pair_a(p, Bx)
    @inbounds ib[1] = _pair_b(p, Bx)
    @inbounds ib[2] = _pair_base(p, n, m)
    for k in 1:D
        @inbounds xr[k] = (1 <= r <= n) ? dX[r, k, a] : zero(T)
    end
    @inbounds S[tid, 1] = one(T)            # diagonals 0 and 1: all boundary
    @inbounds S[tid, 2] = one(T)
    if STORE
        base = _pair_base(p, n, m)
        if _valid(r, 0, n, m)
            @inbounds Kd[base + _gidx(off, r, 0, m)] = one(T)
        end
        if _valid(r, 1, n, m)
            @inbounds Kd[base + _gidx(off, r, 1, m)] = one(T)
        end
    end
    @synchronize
    for t in 2:(n + m)
        r = tid - 1
        c = t - r
        val = one(T)
        if 1 <= r <= n && 1 <= c <= m
            κ = _dot_row(xr, dY, c, @inbounds(ib[1]), Val(D))
            @inbounds left = S[tid, _slot(t - 1)]        # K(r, c-1)
            @inbounds up = S[tid - 1, _slot(t - 1)]      # K(r-1, c)
            @inbounds dg = S[tid - 1, _slot(t - 2)]      # K(r-1, c-1)
            val = (left + up) * _coefA(κ) - dg * _coefB(κ)
        end
        @inbounds S[tid, _slot(t)] = val
        if STORE && _valid(r, t, n, m)
            @inbounds Kd[ib[2] + _gidx(off, r, t, m)] = val
        end
        @synchronize
    end
    if tid - 1 == n
        @inbounds Kout[_pair_a(p, Bx), _pair_b(p, Bx)] = S[tid, _slot(n + m)]
    end
end

@kernel function _gram_bwd!(Xbar::AbstractArray{T,3}, Ybar, @Const(Kd), @Const(off), @Const(Kbar),
                            @Const(dX), @Const(dY), n, m, Bx,
                            ::Val{W}, ::Val{MW}, ::Val{D}) where {T, W, MW, D}
    tid = @index(Local, Linear)
    p = @index(Group, Linear)
    S = @localmem T (W, 3)
    Yb = @localmem T (MW, D)
    xr = @private T (D,)
    xr1 = @private T (D,)
    xb = @private T (D,)
    ib = @private Int (3,)                  # pair row a, column b, grid offset of the pair
    r = tid - 1
    a = _pair_a(p, Bx)
    @inbounds ib[1] = a
    @inbounds ib[2] = _pair_b(p, Bx)
    @inbounds ib[3] = _pair_base(p, n, m)
    for k in 1:D
        @inbounds xr[k] = (1 <= r <= n) ? dX[r, k, a] : zero(T)
        @inbounds xr1[k] = (1 <= r + 1 <= n) ? dX[r + 1, k, a] : zero(T)
        @inbounds xb[k] = zero(T)
    end
    for s in 1:3
        @inbounds S[tid, s] = zero(T)
    end
    for c in tid:W:MW, k in 1:D
        @inbounds Yb[c, k] = zero(T)
    end
    @synchronize
    for t in (n + m):-1:2
        r = tid - 1
        c = t - r
        g = zero(T)
        if 1 <= r <= n && 1 <= c <= m
            b = @inbounds ib[2]
            if r == n && c == m
                g += @inbounds Kbar[ib[1], b]
            end
            if c + 1 <= m
                g += _coefA(_dot_row(xr, dY, c + 1, b, Val(D))) * @inbounds(S[tid, _slot(t + 1)])
            end
            if r + 1 <= n
                g += _coefA(_dot_row(xr1, dY, c, b, Val(D))) * @inbounds(S[tid + 1, _slot(t + 1)])
                if c + 1 <= m
                    g -= _coefB(_dot_row(xr1, dY, c + 1, b, Val(D))) * @inbounds(S[tid + 1, _slot(t + 2)])
                end
            end
            κ = _dot_row(xr, dY, c, b, Val(D))
            base = @inbounds ib[3]
            @inbounds left = Kd[base + _gidx(off, r, t - 1, m)]       # K(r, c-1)
            @inbounds up = Kd[base + _gidx(off, r - 1, t - 1, m)]     # K(r-1, c)
            @inbounds dg = Kd[base + _gidx(off, r - 1, t - 2, m)]     # K(r-1, c-1)
            κ̄ = g * ((left + up) * (T(1) / 2 + κ / 6) + dg * κ / 6)
            for k in 1:D
                @inbounds xb[k] += κ̄ * dY[c, k, b]
                @inbounds Yb[c, k] += κ̄ * xr[k]
            end
        end
        @inbounds S[tid, _slot(t)] = g
        @synchronize
    end
    r = tid - 1
    if 1 <= r <= n
        for k in 1:D
            @inbounds Xbar[r, k, p] = xb[k]
        end
    end
    for c in tid:W:m, k in 1:D
        @inbounds Ybar[c, k, p] = Yb[c, k]
    end
end

# ---------------------------------------------------------------------------
# RBF static kernel, κ(u, v) = exp(-γ|u - v|²), γ = 1/2σ².
#
# The cell coefficient is the second difference of κ over the corners of the original
# cell (p, q), divided by 4^λ for each of its refined sub-cells:
#   G(p, q) = κ(x_{p+1}, y_{q+1}) − κ(x_{p+1}, y_q) − κ(x_p, y_{q+1}) + κ(x_p, y_q).
# A thread's refined row r lies in the original row p = (r-1)>>λ + 1 for the whole
# sweep, and its column moves by one per diagonal, so the thread keeps the four corner
# values in registers ("kc": a0 = κ(x_p, y_q), a1 = κ(x_{p+1}, y_q), b0, b1 the same at
# y_{q+1}) and evaluates two new ones when q changes.
#
# The adjoint scatters instead of gathering: each cell stores A·K̄ and B·K̄ in shared
# memory, so K̄(r, c) = SA(r, c+1) + SA(r+1, c) − SB(r+1, c+1) needs no coefficients of
# the neighbouring cells. A cell's point cotangents go to its two corner points on each
# side: x_p, x_{p+1} in registers, y_q, y_{q+1} in shared memory indexed by the refined
# column (distinct across one diagonal), summed into points on the host.
# ---------------------------------------------------------------------------

@inline _rbf(xv, Y, q, b, γ, ::Val{D}) where {D} =
    exp(-γ * sum(ntuple(k -> @inbounds((xv[k] - Y[q, k, b])^2), Val(D))))

# corner values for original column q into kc[1:2] (a) or kc[3:4] (b)
@inline function _rbf_col!(kc, i, xp, xp1, Y, q, b, γ, ::Val{D}) where {D}
    @inbounds kc[i] = _rbf(xp, Y, q, b, γ, Val(D))
    @inbounds kc[i + 1] = _rbf(xp1, Y, q, b, γ, Val(D))
    return nothing
end
@inline _rbf_cell(kc, sc) = @inbounds((kc[4] - kc[2] - kc[3] + kc[1]) * sc)

@kernel function _gram_fwd_rbf!(Kout::AbstractMatrix{T}, Kd, @Const(off), @Const(X), @Const(Y),
                                n, m, Bx, λ, γ, sc, ::Val{W}, ::Val{D}, ::Val{STORE}) where {T, W, D, STORE}
    tid = @index(Local, Linear)
    p = @index(Group, Linear)
    S = @localmem T (W, 3)
    xp = @private T (D,)
    xp1 = @private T (D,)
    kc = @private T (4,)
    ib = @private Int (3,)                  # pair column b, grid offset, current column q
    r = tid - 1
    a = _pair_a(p, Bx)
    @inbounds ib[1] = _pair_b(p, Bx)
    @inbounds ib[2] = _pair_base(p, n, m)
    @inbounds ib[3] = 1
    if 1 <= r <= n
        po = ((r - 1) >> λ) + 1
        for k in 1:D
            @inbounds xp[k] = X[po, k, a]
            @inbounds xp1[k] = X[po + 1, k, a]
        end
        _rbf_col!(kc, 1, xp, xp1, Y, 1, @inbounds(ib[1]), γ, Val(D))
        _rbf_col!(kc, 3, xp, xp1, Y, 2, @inbounds(ib[1]), γ, Val(D))
    end
    @inbounds S[tid, 1] = one(T)
    @inbounds S[tid, 2] = one(T)
    if STORE
        base = _pair_base(p, n, m)
        if _valid(r, 0, n, m)
            @inbounds Kd[base + _gidx(off, r, 0, m)] = one(T)
        end
        if _valid(r, 1, n, m)
            @inbounds Kd[base + _gidx(off, r, 1, m)] = one(T)
        end
    end
    @synchronize
    for t in 2:(n + m)
        r = tid - 1
        c = t - r
        val = one(T)
        if 1 <= r <= n && 1 <= c <= m
            q = ((c - 1) >> λ) + 1
            if q != @inbounds(ib[3])             # next original column: shift b -> a
                @inbounds kc[1] = kc[3]
                @inbounds kc[2] = kc[4]
                _rbf_col!(kc, 3, xp, xp1, Y, q + 1, @inbounds(ib[1]), γ, Val(D))
                @inbounds ib[3] = q
            end
            κ = _rbf_cell(kc, sc)
            @inbounds left = S[tid, _slot(t - 1)]
            @inbounds up = S[tid - 1, _slot(t - 1)]
            @inbounds dg = S[tid - 1, _slot(t - 2)]
            val = (left + up) * _coefA(κ) - dg * _coefB(κ)
        end
        @inbounds S[tid, _slot(t)] = val
        if STORE && _valid(r, t, n, m)
            @inbounds Kd[ib[2] + _gidx(off, r, t, m)] = val
        end
        @synchronize
    end
    if tid - 1 == n
        @inbounds Kout[_pair_a(p, Bx), _pair_b(p, Bx)] = S[tid, _slot(n + m)]
    end
end

# Xbar[r, :, pair, 1|2]: cotangent of x_p | x_{p+1} from refined row r;
# Ybar[c, :, pair, 1|2]: of y_q | y_{q+1} from refined column c.
@kernel function _gram_bwd_rbf!(Xbar::AbstractArray{T,4}, Ybar, @Const(Kd), @Const(off), @Const(Kbar),
                                @Const(X), @Const(Y), n, m, Mq, Bx, λ, γ, sc,
                                ::Val{W}, ::Val{MW}, ::Val{D}) where {T, W, MW, D}
    tid = @index(Local, Linear)
    p = @index(Group, Linear)
    SA = @localmem T (W, 3)
    SB = @localmem T (W, 3)
    Y0 = @localmem T (MW, D)
    Y1 = @localmem T (MW, D)
    xp = @private T (D,)
    xp1 = @private T (D,)
    xb0 = @private T (D,)
    xb1 = @private T (D,)
    kc = @private T (4,)
    ib = @private Int (4,)                  # pair row a, column b, grid offset, column q
    r = tid - 1
    a = _pair_a(p, Bx)
    @inbounds ib[1] = a
    @inbounds ib[2] = _pair_b(p, Bx)
    @inbounds ib[3] = _pair_base(p, n, m)
    @inbounds ib[4] = Mq                    # sweep starts at the last original column
    for k in 1:D
        @inbounds xb0[k] = zero(T)
        @inbounds xb1[k] = zero(T)
    end
    if 1 <= r <= n
        po = ((r - 1) >> λ) + 1
        for k in 1:D
            @inbounds xp[k] = X[po, k, a]
            @inbounds xp1[k] = X[po + 1, k, a]
        end
        _rbf_col!(kc, 1, xp, xp1, Y, Mq, @inbounds(ib[2]), γ, Val(D))
        _rbf_col!(kc, 3, xp, xp1, Y, Mq + 1, @inbounds(ib[2]), γ, Val(D))
    end
    for s in 1:3
        @inbounds SA[tid, s] = zero(T)
        @inbounds SB[tid, s] = zero(T)
    end
    for c in tid:W:MW, k in 1:D
        @inbounds Y0[c, k] = zero(T)
        @inbounds Y1[c, k] = zero(T)
    end
    @synchronize
    for t in (n + m):-1:2
        r = tid - 1
        c = t - r
        g = zero(T)
        κ = zero(T)
        if 1 <= r <= n && 1 <= c <= m
            b = @inbounds ib[2]
            q = ((c - 1) >> λ) + 1
            if q != @inbounds(ib[4])             # previous original column: shift a -> b
                @inbounds kc[3] = kc[1]
                @inbounds kc[4] = kc[2]
                _rbf_col!(kc, 1, xp, xp1, Y, q, b, γ, Val(D))
                @inbounds ib[4] = q
            end
            if r == n && c == m
                g += @inbounds Kbar[ib[1], b]
            end
            g += @inbounds SA[tid, _slot(t + 1)]                  # (r, c+1)
            if r + 1 <= n
                g += @inbounds SA[tid + 1, _slot(t + 1)]          # (r+1, c)
                g -= @inbounds SB[tid + 1, _slot(t + 2)]          # (r+1, c+1)
            end
            κ = _rbf_cell(kc, sc)
            base = @inbounds ib[3]
            @inbounds left = Kd[base + _gidx(off, r, t - 1, m)]
            @inbounds up = Kd[base + _gidx(off, r - 1, t - 1, m)]
            @inbounds dg = Kd[base + _gidx(off, r - 1, t - 2, m)]
            Ḡ = g * ((left + up) * (T(1) / 2 + κ / 6) + dg * κ / 6) * sc
            # ∂κ(u, v)/∂u = −2γ κ(u, v) (u − v); signs + a0, − a1, − b0, + b1
            w0 = 2γ * Ḡ * @inbounds(kc[1])
            w1 = -2γ * Ḡ * @inbounds(kc[2])
            w2 = -2γ * Ḡ * @inbounds(kc[3])
            w3 = 2γ * Ḡ * @inbounds(kc[4])
            for k in 1:D
                yq = @inbounds Y[q, k, b]
                yq1 = @inbounds Y[q + 1, k, b]
                u0 = @inbounds(xp[k]) - yq;  u1 = @inbounds(xp1[k]) - yq
                u2 = @inbounds(xp[k]) - yq1; u3 = @inbounds(xp1[k]) - yq1
                @inbounds xb0[k] -= w0 * u0 + w2 * u2
                @inbounds xb1[k] -= w1 * u1 + w3 * u3
                @inbounds Y0[c, k] += w0 * u0 + w1 * u1
                @inbounds Y1[c, k] += w2 * u2 + w3 * u3
            end
        end
        @inbounds SA[tid, _slot(t)] = g * _coefA(κ)
        @inbounds SB[tid, _slot(t)] = g * _coefB(κ)
        @synchronize
    end
    r = tid - 1
    if 1 <= r <= n
        for k in 1:D
            @inbounds Xbar[r, k, p, 1] = xb0[k]
            @inbounds Xbar[r, k, p, 2] = xb1[k]
        end
    end
    for c in tid:W:m, k in 1:D
        @inbounds Ybar[c, k, p, 1] = Y0[c, k]
        @inbounds Ybar[c, k, p, 2] = Y1[c, k]
    end
end

# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------

const _MAX_THREADS = 1024          # rows per workgroup
const _MAX_SHMEM = 48 * 1024       # bytes of shared memory per workgroup

_rows_wg(n) = cld(n + 1, 32) * 32
_cols_sh(m) = cld(m, 32) * 32

function _check_gpu_args(X, Y, dyadic_order, static_kernel)
    static_kernel isa Union{LinearKernel, RBFKernel} || throw(ArgumentError(
        "sig_gram on GPU arrays supports the linear and RBF static kernels, got $(typeof(static_kernel))"))
    dyadic_order >= 0 || throw(ArgumentError("dyadic_order must be non-negative, got $dyadic_order"))
    size(X, 2) == size(Y, 2) || throw(DimensionMismatch(
        "paths must have the same dimension, got $(size(X, 2)) and $(size(Y, 2))"))
    min(size(X, 1), size(Y, 1)) >= 2 || throw(ArgumentError("paths need at least 2 points"))
    return nothing
end

# Increments, each split into 2^λ equal pieces.
function _increments(X, λ)
    d = X[2:end, :, :] .- X[1:end-1, :, :]
    λ == 0 && return d
    s = 1 << λ
    return repeat(d ./ s, inner = (s, 1, 1))
end

# Cotangent of the refined increments -> cotangent of the path.
function _path_cotangent(ḡ::AbstractArray{T,3}, λ) where {T}
    s = 1 << λ
    nr, D, B = size(ḡ)
    g = λ == 0 ? ḡ : dropdims(sum(reshape(ḡ, s, nr ÷ s, D, B); dims = 1); dims = 1) ./ s
    z = fill!(similar(g, 1, D, B), zero(T))
    return vcat(z, g) .- vcat(g, z)
end

function _diag_offsets(backend, n, m)
    off = zeros(Int, n + m + 2)
    for t in 0:(n + m)
        off[t + 2] = off[t + 1] + (min(n, t) - max(0, t - m) + 1)
    end
    @assert off[end] == (n + 1) * (m + 1)
    d_off = KA.allocate(backend, Int, n + m + 2)
    copyto!(d_off, off)
    return d_off
end

# Forward on increments oriented so that rows (dX) are the shorter path.
function _gram_inc(dX::AbstractArray{T,3}, dY::AbstractArray{T,3}, store::Bool) where {T}
    backend = get_backend(dX)
    n, D, Bx = size(dX)
    m, _, By = size(dY)
    W = _rows_wg(n)
    W <= _MAX_THREADS || throw(ArgumentError(
        "sig_gram on GPU: the shorter path has $n increments after dyadic refinement; at most $(_MAX_THREADS - 1) are supported"))
    K = KA.allocate(backend, T, Bx, By)
    off = _diag_offsets(backend, n, m)
    Kd = KA.allocate(backend, T, store ? (n + 1) * (m + 1) * Bx * By : 1)
    _gram_fwd!(backend, W)(K, Kd, off, dX, dY, n, m, Bx, Val(W), Val(D), Val(store);
                           ndrange = W * Bx * By)
    return K, Kd, off
end

function _gram_inc_adjoint(K̄, Kd, off, dX::AbstractArray{T,3}, dY::AbstractArray{T,3}) where {T}
    backend = get_backend(dX)
    n, D, Bx = size(dX)
    m, _, By = size(dY)
    W, MW = _rows_wg(n), _cols_sh(m)
    shmem = (3W + MW * D) * sizeof(T)
    shmem <= _MAX_SHMEM || throw(ArgumentError(
        "sig_gram gradient on GPU: $m increments × dimension $D need $shmem bytes of shared memory, more than $(_MAX_SHMEM)"))
    Xbar = KA.allocate(backend, T, n, D, Bx * By)
    Ybar = KA.zeros(backend, T, m, D, Bx * By)
    _gram_bwd!(backend, W)(Xbar, Ybar, Kd, off, K̄, dX, dY, n, m, Bx, Val(W), Val(MW), Val(D);
                           ndrange = W * Bx * By)
    x̄ = dropdims(sum(reshape(Xbar, n, D, Bx, By); dims = 4); dims = 4)
    ȳ = dropdims(sum(reshape(Ybar, m, D, Bx, By); dims = 3); dims = 3)
    return x̄, ȳ
end

# RBF: forward and adjoint on the points, rows (X) the shorter path.
_rbf_gamma(::Type{T}, κ::RBFKernel) where {T} = T(1 / (2 * Float64(κ.sigma)^2))

function _gram_pts(X::AbstractArray{T,3}, Y::AbstractArray{T,3}, λ, γ, store::Bool) where {T}
    backend = get_backend(X)
    Nx, D, Bx = size(X)
    My, _, By = size(Y)
    n, m = (Nx - 1) << λ, (My - 1) << λ
    W = _rows_wg(n)
    W <= _MAX_THREADS || throw(ArgumentError(
        "sig_gram on GPU: the shorter path has $n increments after dyadic refinement; at most $(_MAX_THREADS - 1) are supported"))
    K = KA.allocate(backend, T, Bx, By)
    off = _diag_offsets(backend, n, m)
    Kd = KA.allocate(backend, T, store ? (n + 1) * (m + 1) * Bx * By : 1)
    _gram_fwd_rbf!(backend, W)(K, Kd, off, X, Y, n, m, Bx, λ, γ, T(1) / T(4)^λ,
                               Val(W), Val(D), Val(store); ndrange = W * Bx * By)
    return K, Kd, off
end

# corner cotangents (refined rows, D, B, 2) -> point cotangents (N, D, B)
function _corner_points(g::AbstractArray{T,4}, λ) where {T}
    s = 1 << λ
    nr, D, B, _ = size(g)
    h = λ == 0 ? g : dropdims(sum(reshape(g, s, nr ÷ s, D, B, 2); dims = 1); dims = 1)
    z = fill!(similar(h, 1, D, B), zero(T))
    return vcat(h[:, :, :, 1], z) .+ vcat(z, h[:, :, :, 2])
end

function _gram_pts_adjoint(K̄, Kd, off, X::AbstractArray{T,3}, Y::AbstractArray{T,3}, λ, γ) where {T}
    backend = get_backend(X)
    Nx, D, Bx = size(X)
    My, _, By = size(Y)
    n, m = (Nx - 1) << λ, (My - 1) << λ
    W, MW = _rows_wg(n), _cols_sh(m)
    shmem = (6W + 2 * MW * D) * sizeof(T)
    shmem <= _MAX_SHMEM || throw(ArgumentError(
        "sig_gram gradient on GPU (RBF): $m increments × dimension $D need $shmem bytes of shared memory, more than $(_MAX_SHMEM)"))
    Xbar = KA.allocate(backend, T, n, D, Bx * By, 2)
    Ybar = KA.zeros(backend, T, m, D, Bx * By, 2)
    _gram_bwd_rbf!(backend, W)(Xbar, Ybar, Kd, off, K̄, X, Y, n, m, My - 1, Bx, λ, γ, T(1) / T(4)^λ,
                               Val(W), Val(MW), Val(D); ndrange = W * Bx * By)
    x̄ = dropdims(sum(reshape(Xbar, n, D, Bx, By, 2); dims = 4); dims = 4)
    ȳ = dropdims(sum(reshape(Ybar, m, D, Bx, By, 2); dims = 3); dims = 3)
    return _corner_points(x̄, λ), _corner_points(ȳ, λ)
end

# Forward of the Gram, choosing the orientation (k(x, y) = k(y, x): rows = shorter
# path). Returns the Gram and what the pullback needs.
function _gram_gpu(X::AbstractArray{T,3}, Y::AbstractArray{T,3}, λ::Int, store::Bool,
                   ::LinearKernel) where {T}
    dX, dY = _increments(X, λ), _increments(Y, λ)
    swap = size(dX, 1) > size(dY, 1)
    if swap
        Kt, Kd, off = _gram_inc(dY, dX, store)
        return permutedims(Kt), (swap, Kd, off, dY, dX)
    else
        K, Kd, off = _gram_inc(dX, dY, store)
        return K, (swap, Kd, off, dX, dY)
    end
end

function _gram_gpu(X::AbstractArray{T,3}, Y::AbstractArray{T,3}, λ::Int, store::Bool,
                   κ::RBFKernel) where {T}
    γ = _rbf_gamma(T, κ)
    swap = size(X, 1) > size(Y, 1)
    if swap
        Kt, Kd, off = _gram_pts(Y, X, λ, γ, store)
        return permutedims(Kt), (swap, Kd, off, Y, X)
    else
        K, Kd, off = _gram_pts(X, Y, λ, γ, store)
        return K, (swap, Kd, off, X, Y)
    end
end

_adjoint(K̄, Kd, off, R, C, λ, ::LinearKernel) = map(g -> _path_cotangent(g, λ), _gram_inc_adjoint(K̄, Kd, off, R, C))
_adjoint(K̄, Kd, off, R, C::AbstractArray{T}, λ, κ::RBFKernel) where {T} =
    _gram_pts_adjoint(K̄, Kd, off, R, C, λ, _rbf_gamma(T, κ))

function _gram_gpu_pullback(K̄, saved, λ, κ)
    swap, Kd, off, R, C = saved
    if swap
        ȳ, x̄ = _adjoint(permutedims(K̄), Kd, off, R, C, λ, κ)
    else
        x̄, ȳ = _adjoint(K̄, Kd, off, R, C, λ, κ)
    end
    return x̄, ȳ
end

const _GPU3{T} = AbstractGPUArray{T,3}

function ChenSignatures.sig_gram(X::_GPU3{T}, Y::_GPU3{T}; dyadic_order::Int = 0,
                                 static_kernel::StaticKernel = LinearKernel(),
                                 threaded::Bool = true) where {T<:AbstractFloat}
    _check_gpu_args(X, Y, dyadic_order, static_kernel)
    return first(_gram_gpu(X, Y, dyadic_order, false, static_kernel))
end

function ChenSignatures.sig_gram(X::_GPU3{T}; dyadic_order::Int = 0,
                                 static_kernel::StaticKernel = LinearKernel(),
                                 threaded::Bool = true) where {T<:AbstractFloat}
    _check_gpu_args(X, X, dyadic_order, static_kernel)
    return first(_gram_gpu(X, X, dyadic_order, false, static_kernel))
end

_dense(K̄, ::Type{T}, like) where {T} = (k = unthunk(K̄); k isa AbstractGPUArray{T} ? k : copyto!(similar(like, T, size(k)), k))

function ChainRulesCore.rrule(::typeof(sig_gram), X::_GPU3{T}, Y::_GPU3{T};
                              dyadic_order::Int = 0, static_kernel::StaticKernel = LinearKernel(),
                              threaded::Bool = true) where {T<:AbstractFloat}
    _check_gpu_args(X, Y, dyadic_order, static_kernel)
    K, saved = _gram_gpu(X, Y, dyadic_order, true, static_kernel)
    function sig_gram_gpu_pullback(K̄)
        unthunk(K̄) isa AbstractZero && return NoTangent(), ZeroTangent(), ZeroTangent()
        X̄, Ȳ = _gram_gpu_pullback(_dense(K̄, T, K), saved, dyadic_order, static_kernel)
        return NoTangent(), X̄, Ȳ
    end
    return K, sig_gram_gpu_pullback
end

function ChainRulesCore.rrule(::typeof(sig_gram), X::_GPU3{T};
                              dyadic_order::Int = 0, static_kernel::StaticKernel = LinearKernel(),
                              threaded::Bool = true) where {T<:AbstractFloat}
    _check_gpu_args(X, X, dyadic_order, static_kernel)
    K, saved = _gram_gpu(X, X, dyadic_order, true, static_kernel)
    function sig_gram_gpu_sym_pullback(K̄)
        unthunk(K̄) isa AbstractZero && return NoTangent(), ZeroTangent()
        X̄, Ȳ = _gram_gpu_pullback(_dense(K̄, T, K), saved, dyadic_order, static_kernel)
        return NoTangent(), X̄ .+ Ȳ
    end
    return K, sig_gram_gpu_sym_pullback
end

end
