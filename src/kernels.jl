# Signature kernels k(x, y) = ⟨S(x), S(y)⟩.
#
# Truncated: the inner product of truncated signatures (level 0 included).
# Untruncated: the Goursat PDE of Salvi et al. (2021),
#     ∂²K/∂s∂t = ⟨ẋ(s), ẏ(t)⟩ K,   K(0, ·) = K(·, 0) = 1,   k(x, y) = K(1, 1),
# optionally with a static kernel κ on points in place of ⟨·,·⟩,
# solved with their explicit second-order scheme on the grid of path points,
# each increment split into 2^λ equal pieces (λ = dyadic_order):
#     K[i+1,j+1] = (K[i+1,j] + K[i,j+1]) A − K[i,j] B,
#     A = 1 + c/2 + c²/12,   B = 1 − c²/12,   c = G[p, q] / 4^λ,
# where G[p, q] = ⟨Δx_p, Δy_q⟩ for the linear static kernel, and in general the
# second difference of κ over the cell's corners,
#     G[p, q] = κ(x_{p+1}, y_{q+1}) − κ(x_{p+1}, y_q) − κ(x_p, y_{q+1}) + κ(x_p, y_q).
# The gradient is the adjoint of that recurrence (backward over the grid).

# ---------------------------------------------------------------------------
# PDE solver on one pair, given the cell coefficients G
# ---------------------------------------------------------------------------

@inline function _pde_coeffs(c)
    c2 = c * c / 12
    return 1 + c / 2 + c2, 1 - c2
end

# Value only: two columns of the grid.
function _pde_value!(prev::AbstractVector{T}, cur::AbstractVector{T},
                     G::AbstractMatrix, λ::Int) where {T}
    nx = size(G, 1) << λ
    ny = size(G, 2) << λ
    s = one(T) / T(4)^λ
    fill!(view(prev, 1:nx+1), one(T))
    @inbounds for j in 1:ny
        q = ((j - 1) >> λ) + 1
        cur[1] = one(T)
        for i in 1:nx
            a, b = _pde_coeffs(G[((i - 1) >> λ) + 1, q] * s)
            cur[i+1] = (cur[i] + prev[i+1]) * a - prev[i] * b
        end
        prev, cur = cur, prev
    end
    return prev[nx+1]
end

function _pde_value(G::AbstractMatrix{T}, λ::Int) where {T}
    n = (size(G, 1) << λ) + 1
    return _pde_value!(Vector{T}(undef, n), Vector{T}(undef, n), G, λ)
end

# Full grid, kept for the backward pass.
function _pde_grid!(K::AbstractMatrix{T}, G::AbstractMatrix, λ::Int) where {T}
    nx = size(G, 1) << λ
    ny = size(G, 2) << λ
    s = one(T) / T(4)^λ
    @inbounds begin
        for i in 1:nx+1; K[i, 1] = one(T); end
        for j in 1:ny+1; K[1, j] = one(T); end
        for j in 1:ny
            q = ((j - 1) >> λ) + 1
            for i in 1:nx
                a, b = _pde_coeffs(G[((i - 1) >> λ) + 1, q] * s)
                K[i+1, j+1] = (K[i+1, j] + K[i, j+1]) * a - K[i, j] * b
            end
        end
    end
    return K
end

# Ḡ += ∂(k̄ · k)/∂G, with K the forward grid; K̄ is scratch of the same size.
function _pde_adjoint!(Ḡ::AbstractMatrix{T}, K::AbstractMatrix{T}, K̄::AbstractMatrix{T},
                       G::AbstractMatrix, λ::Int, k̄) where {T}
    nx = size(G, 1) << λ
    ny = size(G, 2) << λ
    s = one(T) / T(4)^λ
    fill!(view(K̄, 1:nx+1, 1:ny+1), zero(T))
    K̄[nx+1, ny+1] = k̄
    @inbounds for j in ny:-1:1
        q = ((j - 1) >> λ) + 1
        for i in nx:-1:1
            g = K̄[i+1, j+1]
            iszero(g) && continue
            p = ((i - 1) >> λ) + 1
            c = G[p, q] * s
            a, b = _pde_coeffs(c)
            ga = g * a
            K̄[i+1, j] += ga
            K̄[i, j+1] += ga
            K̄[i, j]   -= g * b
            # ∂A/∂c = 1/2 + c/6,  ∂B/∂c = −c/6
            c̄ = g * ((K[i+1, j] + K[i, j+1]) * (one(T) / 2 + c / 6) + K[i, j] * c / 6)
            Ḡ[p, q] += c̄ * s
        end
    end
    return Ḡ
end

# ---------------------------------------------------------------------------
# Static kernels: the cell coefficients G and their adjoint
# ---------------------------------------------------------------------------

"""
    StaticKernel

A kernel ``κ(a, b)`` on points of ``ℝ^d``, lifted into the signature kernel through
the `static_kernel` keyword of [`sig_kernel`](@ref) and [`sig_gram`](@ref). Provided:
[`LinearKernel`](@ref) (the plain signature kernel) and [`RBFKernel`](@ref).
"""
abstract type StaticKernel end

"""
    LinearKernel()

``κ(a, b) = ⟨a, b⟩``: the signature kernel of the paths themselves. The default.
"""
struct LinearKernel <: StaticKernel end

# Kernels that depend on the points only through d² = |a − b|²: they define
# `_kfun(κ, d²)` and `_kderiv(κ, d², value)` = dκ/d(d²), and share the code below.
abstract type _DistanceKernel <: StaticKernel end

"""
    RBFKernel(σ)

``κ(a, b) = \\exp(-|a - b|^2 / 2σ^2)``: the signature kernel of the paths lifted into
the RBF feature space. `σ` sets the length scale on the scale of the path values.
"""
struct RBFKernel{T<:Real} <: _DistanceKernel
    sigma::T
    function RBFKernel(σ::Real)
        σ > 0 || throw(ArgumentError("RBFKernel: σ must be positive, got $σ"))
        return new{typeof(σ)}(σ)
    end
end

@inline _kfun(κ::RBFKernel, d2) = Base.exp(-d2 / (2 * κ.sigma^2))
@inline _kderiv(κ::RBFKernel, d2, v) = -v / (2 * κ.sigma^2)

# What a path contributes to the cell coefficients: its increments for the linear
# kernel (G = ΔX ΔYᵀ is exact, with no cancellation between large point values),
# its points otherwise.
_cell_source(X, ::LinearKernel) = _increments(X)
_cell_source(X, ::StaticKernel) = X

# One pair: G = cell coefficients, (N-1) × (M-1).
_cells(x, y, ::LinearKernel) = _increments(x) * transpose(_increments(y))

function _kernel_points(x, y, κ::_DistanceKernel)
    T = float(promote_type(eltype(x), eltype(y)))
    P = Matrix{T}(undef, size(x, 1), size(y, 1))
    @inbounds for q in axes(y, 1), p in axes(x, 1)
        d2 = zero(T)
        for k in axes(x, 2)
            d2 += (x[p, k] - y[q, k])^2
        end
        P[p, q] = _kfun(κ, d2)
    end
    return P
end

_second_difference(P) = P[2:end, 2:end] .- P[2:end, 1:end-1] .- P[1:end-1, 2:end] .+ P[1:end-1, 1:end-1]

_cells(x, y, κ::_DistanceKernel) = _second_difference(_kernel_points(x, y, κ))

# One pair: accumulate the point cotangents x̄ += ∂(Ḡ·G)/∂x and ȳ += ∂(Ḡ·G)/∂y.
function _cells_adjoint!(x̄, ȳ, Ḡ, x, y, ::LinearKernel)
    dx = _increments(x); dy = _increments(y)
    x̄ .+= _path_cotangent(Ḡ * dy)
    ȳ .+= _path_cotangent(transpose(Ḡ) * dx)
    return nothing
end

function _cells_adjoint!(x̄, ȳ, Ḡ, x, y, κ::_DistanceKernel)
    T = eltype(Ḡ)
    N, M = size(x, 1), size(y, 1)
    # adjoint of the second difference
    P̄ = zeros(T, N, M)
    P̄[2:end, 2:end]     .+= Ḡ
    P̄[2:end, 1:end-1]   .-= Ḡ
    P̄[1:end-1, 2:end]   .-= Ḡ
    P̄[1:end-1, 1:end-1] .+= Ḡ
    # ∂κ(a, b)/∂a = 2 κ'(d²) (a − b), with W = −2 P̄ κ'(d²)
    W = similar(P̄)
    @inbounds for q in 1:M, p in 1:N
        d2 = zero(T)
        for k in axes(x, 2)
            d2 += (x[p, k] - y[q, k])^2
        end
        W[p, q] = -2 * P̄[p, q] * _kderiv(κ, d2, _kfun(κ, d2))
    end
    x̄ .+= W * y .- sum(W; dims=2) .* x
    ȳ .+= transpose(W) * x .- transpose(sum(W; dims=1)) .* y
    return nothing
end

# A lane group: Gt[l + (q-1)L, p] = G[p, q] of the pair (x_a, y_{b_l}), with ycat
# holding the lanes' y sources lane-major and xa the x source.
function _lane_cells!(Gt, scratch, ycat, xa, L::Int, ::LinearKernel)
    LinearAlgebra.mul!(Gt, ycat, transpose(xa))
    return Gt
end

function _lane_cells!(Gt, P, ycat, xa, L::Int, κ::_DistanceKernel)
    T = eltype(Gt)
    R, D = size(ycat)              # R = L * M
    @inbounds for p in axes(xa, 1)
        for r in 1:R
            d2 = zero(T)
            for k in 1:D
                d2 += (ycat[r, k] - xa[p, k])^2
            end
            P[r, p] = _kfun(κ, d2)
        end
    end
    nq = R ÷ L - 1
    @inbounds for p in 1:size(xa, 1)-1
        @simd for r in 1:L*nq      # r = l + (q-1)L, so r + L is (l, q+1)
            Gt[r, p] = P[r+L, p+1] - P[r, p+1] - P[r+L, p] + P[r, p]
        end
    end
    return Gt
end

_lane_scratch(::Type{T}, ycat, xa, ::LinearKernel) where {T} = nothing
_lane_scratch(::Type{T}, ycat, xa, ::StaticKernel) where {T} =
    Matrix{T}(undef, size(ycat, 1), size(xa, 1))

# ---------------------------------------------------------------------------
# Increments and their cotangents
# ---------------------------------------------------------------------------

_increments(x::AbstractMatrix) = x[2:end, :] .- x[1:end-1, :]

function _increments(X::AbstractArray{T,3}) where {T}
    return X[2:end, :, :] .- X[1:end-1, :, :]
end

# Cotangent of the path from the cotangent of its increments (dims=1).
function _path_cotangent(dz::AbstractArray{T}) where {T}
    sz = (size(dz, 1) + 1, Base.tail(size(dz))...)
    x̄ = zeros(T, sz)
    n = size(dz, 1)
    @inbounds for idx in CartesianIndices(Base.tail(size(dz))), p in 1:n
        v = dz[p, idx]
        x̄[p+1, idx] += v
        x̄[p, idx]   -= v
    end
    return x̄
end

function _check_kernel_args(dx::Int, dy::Int, λ::Int)
    dx == dy || throw(ArgumentError("paths must have the same dimension, got $dx and $dy"))
    λ >= 0 || throw(ArgumentError("dyadic_order must be non-negative, got $λ"))
    return nothing
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    sig_kernel(x, y; dyadic_order=0, static_kernel=LinearKernel()) -> Real
    sig_kernel(x, y, m) -> Real

Signature kernel ``k(x, y) = ⟨S(x), S(y)⟩`` of two piecewise-linear paths, given as
`N × d` and `M × d` matrices (`N`, `M` may differ).

- Without `m`: the untruncated kernel, solved from the Goursat PDE
  ``∂²K/∂s∂t = ⟨ẋ, ẏ⟩ K`` (Salvi et al., 2021) with a second-order explicit scheme.
  Each path increment is split into `2^dyadic_order` pieces; raise it to refine.
  Cost `O(N M d + N M 4^dyadic_order)`, independent of any truncation level.
  `static_kernel` replaces ``⟨·,·⟩`` by a kernel on points, e.g. [`RBFKernel`](@ref);
  each grid cell then carries the second difference of ``κ`` over its corners.
- With `m`: the kernel of signatures truncated at level `m`, `1 + dot(sig(x, m), sig(y, m))`.

Both are differentiable (ChainRules) in `x` and `y`.

# Example
```julia
x, y = randn(50, 3), randn(40, 3)
sig_kernel(x, y; dyadic_order=1)
sig_kernel(x, y; static_kernel=RBFKernel(0.5))
sig_kernel(x, y, 6)
```
"""
function sig_kernel(x::AbstractMatrix, y::AbstractMatrix; dyadic_order::Int=0,
                    static_kernel::StaticKernel=LinearKernel())
    _check_kernel_args(size(x, 2), size(y, 2), dyadic_order)
    return _pde_value(_cells(x, y, static_kernel), dyadic_order)
end

function sig_kernel(x::AbstractMatrix, y::AbstractMatrix, m::Int)
    _check_kernel_args(size(x, 2), size(y, 2), 0)
    return 1 + dot(sig(x, m), sig(y, m))
end

"""
    sig_gram(X, Y; dyadic_order=0, static_kernel=LinearKernel(), threaded=true) -> Matrix
    sig_gram(X; dyadic_order=0, static_kernel=LinearKernel(), threaded=true) -> Matrix
    sig_gram(X, Y, m; threaded=true) -> Matrix
    sig_gram(X, m; threaded=true) -> Matrix

Gram matrix `K[a, b] = sig_kernel(X[:, :, a], Y[:, :, b])` for batches of paths stored
as `N × d × B` arrays (the two batches may have different `N` and `B`). The one-batch
forms compute `K = sig_gram(X, X)`, using its symmetry.

Without `m` the kernel is the untruncated PDE kernel (see [`sig_kernel`](@ref)), with
pairs spread over threads. With `m` it is `1 .+ sig(X, m)' * sig(Y, m)`.

Differentiable (ChainRules) in `X` and `Y`.
"""
function sig_gram(X::AbstractArray{<:Any,3}, Y::AbstractArray{<:Any,3};
                  dyadic_order::Int=0, static_kernel::StaticKernel=LinearKernel(),
                  threaded::Bool=true)
    _check_kernel_args(size(X, 2), size(Y, 2), dyadic_order)
    groups = _lane_groups(axes(X, 3), a -> axes(Y, 3))
    return _gram_forward(X, Y, groups, dyadic_order, static_kernel, threaded, false)
end

function sig_gram(X::AbstractArray{<:Any,3}; dyadic_order::Int=0,
                  static_kernel::StaticKernel=LinearKernel(), threaded::Bool=true)
    _check_kernel_args(size(X, 2), size(X, 2), dyadic_order)
    B = size(X, 3)
    groups = _lane_groups(1:B, a -> a:B)
    return _gram_forward(X, X, groups, dyadic_order, static_kernel, threaded, true)
end

function sig_gram(X::AbstractArray{<:Any,3}, Y::AbstractArray{<:Any,3}, m::Int;
                  threaded::Bool=true)
    _check_kernel_args(size(X, 2), size(Y, 2), 0)
    return 1 .+ transpose(sig(X, m; threaded=threaded)) * sig(Y, m; threaded=threaded)
end

function sig_gram(X::AbstractArray{<:Any,3}, m::Int; threaded::Bool=true)
    S = sig(X, m; threaded=threaded)
    return 1 .+ transpose(S) * S
end

# ---------------------------------------------------------------------------
# Gram internals
# ---------------------------------------------------------------------------

# Split `n` items into contiguous ranges, one per task.
function _chunks(n::Int, threaded::Bool)
    nt = threaded ? min(Threads.nthreads(), n) : 1
    nt = max(nt, 1)
    return collect(Iterators.partition(1:n, cld(max(n, 1), nt)))
end

# Pairs are solved _GRAM_LANES at a time, side by side: each grid row is a serial
# chain, so one pair alone is latency-bound, while independent pairs fill the SIMD
# lanes. A group shares its x path, so the cell coefficients of all its pairs come
# from one pass over x (a matrix product for the linear kernel), landing lane-major.
const _GRAM_LANES = 16

# Groups (a, bs) with length(bs) <= _GRAM_LANES, covering the pairs (a, b ∈ cols(a)).
function _lane_groups(rows, cols)
    groups = Tuple{Int,Vector{Int}}[]
    for a in rows, bs in Iterators.partition(cols(a), _GRAM_LANES)
        push!(groups, (a, collect(bs)))
    end
    return groups
end

function _gram_forward(X, Y, groups, λ::Int, κ::StaticKernel, threaded::Bool,
                       symmetric::Bool)
    T = float(promote_type(eltype(X), eltype(Y)))
    sX = _cell_source(X, κ); sY = symmetric ? sX : _cell_source(Y, κ)
    Kout = Matrix{T}(undef, size(X, 3), size(Y, 3))
    nX, nY = size(X, 1) - 1, size(Y, 1) - 1      # cells along x and y
    mY, D = size(sY, 1), size(sY, 2)
    L = _GRAM_LANES
    tasks = map(_chunks(length(groups), threaded)) do r
        Threads.@spawn begin
            local ycat = Matrix{T}(undef, L * mY, D)
            local Gt = Matrix{T}(undef, L * nY, nX)
            local scratch = _lane_scratch(T, ycat, view(sX, :, :, 1), κ)
            local prev = Matrix{T}(undef, L, (nY << λ) + 1)
            local cur = similar(prev)
            local out = Vector{T}(undef, L)
            for g in r
                a, bs = groups[g]
                # ycat[l + (q-1)L, k] = source row q of lane l; a short group repeats its last path
                @inbounds for k in 1:D, q in 1:mY, l in 1:L
                    ycat[l + (q - 1) * L, k] = sY[q, k, bs[min(l, length(bs))]]
                end
                _lane_cells!(Gt, scratch, ycat, view(sX, :, :, a), L, κ)
                # the scheme is symmetric in (x, y): sweep y along the inner loop
                _pde_value_lanes!(out, prev, cur, reshape(Gt, L, nY, nX), λ)
                for (l, b) in enumerate(bs)
                    Kout[a, b] = out[l]
                    symmetric && (Kout[b, a] = out[l])
                end
            end
        end
    end
    foreach(wait, tasks)
    return Kout
end

function _pde_value_lanes!(out, prev::AbstractMatrix{T}, cur::AbstractMatrix{T},
                           G::AbstractArray{T,3}, λ::Int) where {T}
    L, nX, nY = size(G)
    nx, ny = nX << λ, nY << λ
    s = one(T) / T(4)^λ
    fill!(prev, one(T))
    @inbounds for j in 1:ny
        q = ((j - 1) >> λ) + 1
        for l in 1:L; cur[l, 1] = one(T); end
        for i in 1:nx
            p = ((i - 1) >> λ) + 1
            @simd for l in 1:L
                a, b = _pde_coeffs(G[l, p, q] * s)
                cur[l, i+1] = (cur[l, i] + prev[l, i+1]) * a - prev[l, i] * b
            end
        end
        prev, cur = cur, prev
    end
    @inbounds for l in 1:L; out[l] = prev[l, nx+1]; end
    return out
end

# Ḡ for one pair with output cotangent w.
function _pair_cell_cotangent!(Ḡ, K, K̄, G, λ::Int, w)
    gx, gy = (size(G, 1) << λ) + 1, (size(G, 2) << λ) + 1
    Kv = view(K, 1:gx, 1:gy); K̄v = view(K̄, 1:gx, 1:gy)
    _pde_grid!(Kv, G, λ)
    fill!(Ḡ, zero(eltype(Ḡ)))
    _pde_adjoint!(Ḡ, Kv, K̄v, G, λ, w)
    return Ḡ
end

# Weighted pairs (a, b, w): accumulate w · ∂k(a, b) into the path cotangents.
# With `symmetric`, X and Y are the same batch and both sides land in X̄.
function _gram_vjp(X, Y, wpairs, λ::Int, κ::StaticKernel, threaded::Bool, symmetric::Bool)
    T = float(promote_type(eltype(X), eltype(Y)))
    isempty(wpairs) && return zeros(T, size(X)), zeros(T, size(Y))
    nX, nY = size(X, 1) - 1, size(Y, 1) - 1
    gx, gy = (nX << λ) + 1, (nY << λ) + 1
    chunks = _chunks(length(wpairs), threaded)
    results = Vector{Any}(undef, length(chunks))
    tasks = map(enumerate(chunks)) do (c, r)
        Threads.@spawn begin
            local X̄ = zeros(T, size(X))
            local Ȳ = symmetric ? X̄ : zeros(T, size(Y))
            local Ḡ = Matrix{T}(undef, nX, nY)
            local K = Matrix{T}(undef, gx, gy)
            local K̄ = Matrix{T}(undef, gx, gy)
            for t in r
                a, b, w = wpairs[t]
                xa = view(X, :, :, a); yb = view(Y, :, :, b)
                G = _cells(xa, yb, κ)
                _pair_cell_cotangent!(Ḡ, K, K̄, G, λ, w)
                _cells_adjoint!(view(X̄, :, :, a), view(Ȳ, :, :, b), Ḡ, xa, yb, κ)
            end
            results[c] = (X̄, Ȳ)
        end
    end
    foreach(wait, tasks)
    X̄ = reduce(+, first.(results))
    symmetric && return X̄, X̄
    return X̄, reduce(+, last.(results))
end

function _sig_kernel_vjp(x, y, λ::Int, κ::StaticKernel, k̄)
    T = float(promote_type(eltype(x), eltype(y)))
    G = _cells(x, y, κ)
    gx, gy = (size(G, 1) << λ) + 1, (size(G, 2) << λ) + 1
    Ḡ = _pair_cell_cotangent!(similar(G, T), Matrix{T}(undef, gx, gy),
                              Matrix{T}(undef, gx, gy), G, λ, k̄)
    x̄ = zeros(T, size(x)); ȳ = zeros(T, size(y))
    _cells_adjoint!(x̄, ȳ, Ḡ, x, y, κ)
    return x̄, ȳ
end
