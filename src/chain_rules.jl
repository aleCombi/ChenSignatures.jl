# ChainRules for the signature. The pullbacks use the reversibility-based
# reverse pass in backward.jl, so any ChainRules-aware AD backend (Zygote,
# Diffractor, and others through their ChainRules bridges) gets an O(D^M)-memory
# gradient that does not store the forward trajectory.

_dense(x::Array) = x
_dense(x::AbstractArray) = Array(x)

function ChainRulesCore.rrule(::typeof(sig), path::AbstractMatrix{T}, m::Int) where {T}
    y = sig(path, m)
    function sig_pullback(ȳ_raw)
        ȳ = ChainRulesCore.unthunk(ȳ_raw)
        ȳ isa ChainRulesCore.AbstractZero && return (NoTangent(), ZeroTangent(), NoTangent())
        p = _dense(path)
        x̄ = _sig_vjp(p, Val(size(p, 2)), Val(m), y, ȳ)
        return (NoTangent(), x̄, NoTangent())
    end
    return y, sig_pullback
end

function ChainRulesCore.rrule(::typeof(sig), paths::AbstractArray{T,3}, m::Int;
                              threaded::Bool=true) where {T}
    Y = sig(paths, m; threaded=threaded)
    function sig_batch_pullback(Ȳ_raw)
        Ȳ = ChainRulesCore.unthunk(Ȳ_raw)
        Ȳ isa ChainRulesCore.AbstractZero && return (NoTangent(), ZeroTangent(), NoTangent())
        P = _dense(paths)
        X̄ = _sig_vjp_batch(P, Val(size(P, 2)), Val(m), Y, Ȳ, threaded)
        return (NoTangent(), X̄, NoTangent())
    end
    return Y, sig_batch_pullback
end

function ChainRulesCore.rrule(::typeof(logsig), path::AbstractMatrix{T}, basis::BasisCache;
                              coords::Symbol=:lyndon) where {T}
    N, D = size(path)
    _check_logsig_input(N, D, basis); _check_coords(coords)
    # keep the signature: the reverse pass then skips one forward pass
    y, sflat = _logsig_single(path, basis, coords, Val(D), Val(basis.m); keep_sig=true)
    function logsig_pullback(ȳ_raw)
        ȳ = ChainRulesCore.unthunk(ȳ_raw)
        ȳ isa ChainRulesCore.AbstractZero && return (NoTangent(), ZeroTangent(), NoTangent())
        p = _dense(path)
        x̄ = _logsig_vjp(p, ȳ, basis, coords, Val(size(p, 2)), Val(basis.m), sflat)
        return (NoTangent(), x̄, NoTangent())
    end
    return y, logsig_pullback
end

function ChainRulesCore.rrule(::typeof(logsig), paths::AbstractArray{T,3}, basis::BasisCache;
                              coords::Symbol=:lyndon, threaded::Bool=true) where {T}
    N, D, B = size(paths)
    _check_logsig_input(N, D, basis); _check_coords(coords)
    B >= 1 || throw(ArgumentError("Batch size must be at least 1, got B=$B"))
    Y, Sg = _logsig_batch(paths, basis, coords, threaded, Val(D), Val(basis.m); keep_sig=true)
    function logsig_batch_pullback(Ȳ_raw)
        Ȳ = ChainRulesCore.unthunk(Ȳ_raw)
        Ȳ isa ChainRulesCore.AbstractZero && return (NoTangent(), ZeroTangent(), NoTangent())
        P = _dense(paths)
        X̄ = _logsig_vjp_batch(P, Ȳ, basis, coords, threaded, Val(size(P, 2)), Val(basis.m), Sg)
        return (NoTangent(), X̄, NoTangent())
    end
    return Y, logsig_batch_pullback
end

# PDE signature kernel: the pullbacks run the adjoint of the finite-difference
# recurrence (kernels.jl). The truncated kernels need no rule of their own: they
# are built from `sig`, whose rules above apply.

function ChainRulesCore.rrule(::typeof(sig_kernel), x::AbstractMatrix, y::AbstractMatrix;
                              dyadic_order::Int=0)
    k = sig_kernel(x, y; dyadic_order=dyadic_order)
    function sig_kernel_pullback(k̄_raw)
        k̄ = ChainRulesCore.unthunk(k̄_raw)
        k̄ isa ChainRulesCore.AbstractZero && return (NoTangent(), ZeroTangent(), ZeroTangent())
        x̄, ȳ = _sig_kernel_vjp(_dense(x), _dense(y), dyadic_order, k̄)
        return (NoTangent(), x̄, ȳ)
    end
    return k, sig_kernel_pullback
end

function ChainRulesCore.rrule(::typeof(sig_gram), X::AbstractArray{<:Any,3},
                              Y::AbstractArray{<:Any,3}; dyadic_order::Int=0,
                              threaded::Bool=true)
    K = sig_gram(X, Y; dyadic_order=dyadic_order, threaded=threaded)
    function sig_gram_pullback(K̄_raw)
        K̄ = ChainRulesCore.unthunk(K̄_raw)
        K̄ isa ChainRulesCore.AbstractZero && return (NoTangent(), ZeroTangent(), ZeroTangent())
        wpairs = [(a, b, K̄[a, b]) for a in axes(K̄, 1), b in axes(K̄, 2) if !iszero(K̄[a, b])]
        X̄, Ȳ = _gram_vjp(_dense(X), _dense(Y), wpairs, dyadic_order, threaded, false)
        return (NoTangent(), X̄, Ȳ)
    end
    return K, sig_gram_pullback
end

function ChainRulesCore.rrule(::typeof(sig_gram), X::AbstractArray{<:Any,3};
                              dyadic_order::Int=0, threaded::Bool=true)
    K = sig_gram(X; dyadic_order=dyadic_order, threaded=threaded)
    function sig_gram_sym_pullback(K̄_raw)
        K̄ = ChainRulesCore.unthunk(K̄_raw)
        K̄ isa ChainRulesCore.AbstractZero && return (NoTangent(), ZeroTangent())
        B = size(K̄, 1)
        # K[a,b] = K[b,a] is one value: its cotangent is K̄[a,b] + K̄[b,a]
        wpairs = [(a, b, a == b ? K̄[a, a] : K̄[a, b] + K̄[b, a]) for b in 1:B for a in 1:b]
        filter!(t -> !iszero(t[3]), wpairs)
        X̄, _ = _gram_vjp(_dense(X), _dense(X), wpairs, dyadic_order, threaded, true)
        return (NoTangent(), X̄)
    end
    return K, sig_gram_sym_pullback
end
