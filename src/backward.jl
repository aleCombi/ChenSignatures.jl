# ============================================================================
# Reverse-mode gradient of the signature via reversibility
# ============================================================================
#
# Forward:  S_j = S_{j-1} ⊗ exp(z_j),  z_j = x_{j+1} - x_j,  j = 1..n.
#
# Backward, for j = n..1, holding S_j and G_j = ∂L/∂S_j:
#   1. Reconstruct S_{j-1} = S_j ⊗ exp(-z_j)   (same Horner kernel, with -z)
#   2. Run the adjoint of the Horner update to get G_{j-1} = ∂L/∂S_{j-1}
#      (in place) and z̄_j = ∂L/∂z_j.
#   3. x̄_{j+1} += z̄_j,  x̄_j -= z̄_j.
#
# Only one signature, one cotangent and O(D^(M-1)) scratch are alive at a time,
# so memory does not grow with the path length.
#
# The forward Horner chain for target level k (see update_signature_horner!) is
#   v_1 = z / k
#   u_i = v_i + S_i,   v_{i+1} = c_i · (u_i ⊗ z),   c_i = 1/(k-i),   i = 1..k-1
#   S_k += v_k
# and the level-1 update is S_1 += z. All S_i on the right are S_{j-1}, because
# the forward pass updates levels from M down to 1.
#
# The adjoint of chain k, given v̄_k = G_k:
#   for i = k-1..1:
#     ū_i[r]  = c_i Σ_d v̄_{i+1}[(r-1)D+d] z[d]      (contract the last letter)
#     z̄[d]   += c_i Σ_r v̄_{i+1}[(r-1)D+d] u_i[r]
#     G_i    += ū_i,   v̄_i = ū_i
#   z̄ += v̄_1 / k
# Chains are processed for k = 2..M in increasing order: chain k reads G_k and
# only writes to levels below k, and contributions to level k come only from
# chains above k, so G_k is still the untouched G_j value when chain k reads it.

"""
    SigBackwardWorkspace{T,D,M}()

Preallocated scratch buffers for the reverse pass of the signature
(see `sig_pullback!`). Reusing one avoids all allocations in the hot loop.
"""
struct SigBackwardWorkspace{T,D,M}
    fwd::SignatureWorkspace{T,D,M}   # scratch for the reconstruction step
    U::Vector{T}                      # u_1..u_{M-1} for one chain, level layout
    uoff::Vector{Int}                 # 0-based start of u_i in U
    A1::Vector{T}                     # ping-pong adjoint buffers, size D^(M-1)
    A2::Vector{T}
end

function SigBackwardWorkspace{T,D,M}() where {T,D,M}
    uoff = Vector{Int}(undef, max(M, 1))
    total = 0
    for i in 1:M-1
        uoff[i] = total
        total += D^i
    end
    max_len = M > 1 ? D^(M-1) : 1
    return SigBackwardWorkspace{T,D,M}(
        SignatureWorkspace{T,D,M}(),
        Vector{T}(undef, max(total, 1)),
        uoff,
        Vector{T}(undef, max_len),
        Vector{T}(undef, max_len),
    )
end

# Adjoint of one update_signature_horner! step.
# S holds S_{j-1} (already reconstructed); G holds G_j on entry and G_{j-1} on
# exit. Returns z̄_j.
function _horner_step_adjoint!(
    G::Tensor{T,D,M},
    S::Tensor{T,D,M},
    z::SVector{D,T},
    ws::SigBackwardWorkspace{T,D,M},
) where {T,D,M}
    g = G.coeffs
    s = S.coeffs
    off = S.offsets
    U, uoff = ws.U, ws.uoff

    # Level 1: S_1 += z
    zbar = MVector{D,T}(undef)
    @inbounds for d in 1:D
        zbar[d] = g[off[2] + d]
    end

    @inbounds for k in 2:M
        # --- recompute u_1..u_{k-1} of chain k -------------------------------
        inv_k = inv(T(k))
        # u_1 = z/k + S_1
        for d in 1:D
            U[uoff[1] + d] = z[d] * inv_k + s[off[2] + d]
        end
        len = D
        for i in 1:k-2
            c = inv(T(k - i))
            src = uoff[i]
            dst = uoff[i+1]
            soff = off[i+2]          # start of level i+1 in S
            for r in 1:len
                cu = c * U[src + r]
                base = (r - 1) * D
                for d in 1:D
                    U[dst + base + d] = cu * z[d] + s[soff + base + d]
                end
            end
            len *= D
        end

        # --- reverse sweep: v̄_k = G_k ----------------------------------------
        # i = k-1 reads v̄ straight from G_k; later steps ping-pong in A1/A2.
        vbar_src_is_G = true
        vbar_buf = ws.A1
        out_buf = ws.A2
        gk = off[k+1]                # start of level k in G
        len = D^(k-1)                # length of u_i, starting with i = k-1
        for i in k-1:-1:1
            c = inv(T(k - i))
            u0 = uoff[i]
            gi = off[i+1]            # start of level i in G
            if vbar_src_is_G
                _adjoint_level!(zbar, g, gi, out_buf, g, gk, U, u0, z, c, len)
            else
                _adjoint_level!(zbar, g, gi, out_buf, vbar_buf, 0, U, u0, z, c, len)
            end
            # ū_i becomes v̄_i for the next (lower) step
            vbar_src_is_G = false
            vbar_buf, out_buf = out_buf, vbar_buf
            len = div(len, D)
        end
        # v_1 = z/k
        for d in 1:D
            zbar[d] = muladd(vbar_buf[d], inv_k, zbar[d])
        end
    end
    return SVector{D,T}(zbar)
end

# One transposed chain step. View V̄_{i+1} = src[so+1 : so+len*D] as a len × D
# matrix (rows = prefixes, columns = last letter; rows are contiguous). Then
#   ū_i = c V̄ z        -> written to out[1:len] and added to g[gi+1 : gi+len]
#   z̄  += c V̄ᵀ u_i
# `src` may be the same array as `g`: it is read at level i+1 and g is written
# at level i, which do not overlap.
#
# Two loop shapes, picked at compile time from D (measured per step):
# - small D: one fused row loop is fastest (D = 4: 3.7 vs 5.2 μs, D = 8: 14 vs 18);
# - large D: two passes with @simd inner loops (D = 16: 28 vs 47 μs).
const _ADJOINT_TWO_PASS_MIN_D = 12

@inline function _adjoint_level!(
    zbar::MVector{D,T}, g::Vector{T}, gi::Int, out::Vector{T},
    src::Vector{T}, so::Int, U::Vector{T}, u0::Int,
    z::SVector{D,T}, c::T, len::Int,
) where {D,T}
    if D < _ADJOINT_TWO_PASS_MIN_D
        @inbounds for r in 1:len
            base = so + (r - 1) * D
            acc = zero(T)
            ur = U[u0 + r]
            for d in 1:D
                vb = src[base + d]
                acc = muladd(vb, z[d], acc)
                zbar[d] = muladd(c * vb, ur, zbar[d])
            end
            ubar = c * acc
            out[r] = ubar
            g[gi + r] += ubar
        end
        return nothing
    end
    @inbounds for r in 1:len
        base = so + (r - 1) * D
        acc = zero(T)
        @simd for d in 1:D
            acc = muladd(src[base + d], z[d], acc)
        end
        ubar = c * acc
        out[r] = ubar
        g[gi + r] += ubar
    end
    @inbounds for r in 1:len
        base = so + (r - 1) * D
        cu = c * U[u0 + r]
        @simd for d in 1:D
            zbar[d] = muladd(cu, src[base + d], zbar[d])
        end
    end
    return nothing
end

"""
    sig_pullback!(x̄, path, S, G, ws) -> x̄

Reverse pass of the truncated signature by reversibility.

- `x̄`: `N×D` output, overwritten with ∂L/∂path.
- `path`: the `N×D` path.
- `S`: the signature of `path` (destroyed: it is rewound to the unit tensor).
- `G`: the cotangent ∂L/∂S (destroyed). Level 0 is ignored.
- `ws`: a `SigBackwardWorkspace`.

Memory use does not depend on `N`.
"""
function sig_pullback!(
    xbar::AbstractMatrix{T},
    path::AbstractMatrix{T},
    S::Tensor{T,D,M},
    G::Tensor{T,D,M},
    ws::SigBackwardWorkspace{T,D,M},
) where {T,D,M}
    N = size(path, 1)
    fill!(xbar, zero(T))
    @inbounds for j in N-1:-1:1
        z = SVector{D,T}(ntuple(d -> path[j+1, d] - path[j, d], Val(D)))
        # S_j -> S_{j-1}
        update_signature_horner!(S, -z, ws.fwd.B1, ws.fwd.B2)
        zbar = _horner_step_adjoint!(G, S, z, ws)
        for d in 1:D
            xbar[j+1, d] += zbar[d]
            xbar[j, d]   -= zbar[d]
        end
    end
    return xbar
end

# Load a flat signature vector (levels 1..M, as returned by `sig`) into a Tensor.
function _unflatten!(t::Tensor{T,D,M}, v::AbstractVector) where {T,D,M}
    fill!(t.coeffs, zero(T))
    _write_unit!(t)
    idx = 1
    @inbounds for k in 1:M
        start = t.offsets[k+1]
        for i in 1:D^k
            t.coeffs[start + i] = T(v[idx])
            idx += 1
        end
    end
    return t
end

# Gradient of dot(sig(path, M), ȳ) w.r.t. path, given the signature value y.
function _sig_vjp!(
    xbar::AbstractMatrix{T}, path::AbstractMatrix{T},
    y::AbstractVector, ybar::AbstractVector,
    S::Tensor{T,D,M}, G::Tensor{T,D,M}, ws::SigBackwardWorkspace{T,D,M},
) where {T,D,M}
    _unflatten!(S, y)
    _unflatten!(G, ybar)
    return sig_pullback!(xbar, path, S, G, ws)
end

function _sig_vjp(path::AbstractMatrix{T}, ::Val{D}, ::Val{M},
                  y::AbstractVector, ybar::AbstractVector) where {T,D,M}
    S = Tensor{T,D,M}()
    G = Tensor{T,D,M}()
    ws = SigBackwardWorkspace{T,D,M}()
    xbar = similar(path, T, size(path))
    return _sig_vjp!(xbar, path, y, ybar, S, G, ws)
end

function _sig_vjp_batch(paths::AbstractArray{T,3}, ::Val{D}, ::Val{M},
                        Y::AbstractMatrix, Ybar::AbstractMatrix,
                        threaded::Bool) where {T,D,M}
    N, _, B = size(paths)
    X̄ = zeros(T, N, D, B)
    if threaded
        Threads.@threads for b in 1:B
            # `local` matters: the serial branch below also binds S, G, ws in
            # the function scope, and without it every task would share them.
            local Sb = Tensor{T,D,M}()
            local Gb = Tensor{T,D,M}()
            local wsb = SigBackwardWorkspace{T,D,M}()
            _sig_vjp!(@view(X̄[:, :, b]), @view(paths[:, :, b]),
                      @view(Y[:, b]), @view(Ybar[:, b]), Sb, Gb, wsb)
        end
    else
        S = Tensor{T,D,M}(); G = Tensor{T,D,M}()
        ws = SigBackwardWorkspace{T,D,M}()
        for b in 1:B
            _sig_vjp!(@view(X̄[:, :, b]), @view(paths[:, :, b]),
                      @view(Y[:, b]), @view(Ybar[:, b]), S, G, ws)
        end
    end
    return X̄
end
