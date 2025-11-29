using ChainRulesCore
using Enzyme
using LinearAlgebra

# Define the gradient computation as a separate, non-inlined function
Base.@noinline function _compute_sig_gradient(path::AbstractMatrix{T}, m::Int, ȳ::Vector{T}) where T
    path_copy = copy(path)
    grad = zeros(T, size(path_copy))
    loss(p) = dot(sig(p, m), ȳ)
    autodiff(Reverse, loss, Active, Duplicated(path_copy, grad))
    return grad
end

function ChainRulesCore.rrule(::typeof(sig), path::AbstractMatrix{T}, m::Int) where T
    result = sig(path, m)
    
    function sig_pullback(ȳ_raw)
        # Unwrap the tangent to get the actual vector
        ȳ = ChainRulesCore.unthunk(ȳ_raw)
        
        # Call the gradient computation in a way Zygote cannot trace
        grad = ChainRulesCore.@ignore_derivatives begin
            Base.invokelatest(_compute_sig_gradient, path, m, ȳ)
        end
        
        return (NoTangent(), grad, NoTangent())
    end
    
    return result, sig_pullback
end