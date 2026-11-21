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
