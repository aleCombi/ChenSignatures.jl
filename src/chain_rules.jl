# src/chainrules.jl
# ChainRules integration for automatic differentiation support
# Enables Zygote, ReverseDiff, and other ChainRules-compatible AD systems

using ChainRulesCore
using Enzyme

# ==============================================================================
# sig - Compute signature with gradient support
# ==============================================================================

"""
    rrule(::typeof(sig), path::AbstractMatrix{T}, m::Int) where T

Define reverse-mode AD rule for `sig(path, m)`.

Uses Enzyme internally to compute gradients. This enables Zygote and other
ChainRules-compatible AD systems to differentiate through signature computations.

# Arguments
- `path::AbstractMatrix{T}`: N×D matrix where each row is a point
- `m::Int`: Maximum signature level

# Returns
- `result`: Signature vector (same as `sig(path, m)`)
- `pullback`: Function to compute gradients given output cotangent
"""
function ChainRulesCore.rrule(
    ::typeof(sig),
    path::AbstractMatrix{T},
    m::Int
) where {T<:Real}
    
    # Forward pass
    result = sig(path, m)
    
    # Pullback function (reverse-mode gradient)
    function sig_pullback(Δresult)
        # Δresult is the cotangent (gradient) w.r.t. the output
        # We need to compute the gradient w.r.t. path
        
        # Handle different cotangent types
        Δ = unthunk(Δresult)
        
        # Convert to concrete array if needed
        if !(Δ isa AbstractVector{<:Real})
            Δ = convert(Vector{T}, Δ)
        end
        
        # Allocate gradient w.r.t. path
        ∂path = zero(path)
        
        # Define scalar loss function for Enzyme: L = dot(sig(path, m), Δresult)
        # The gradient of this loss w.r.t. path is what we want
        function loss_for_enzyme(p)
            s = sig(p, m)
            return dot(s, Δ)
        end
        
        # Use Enzyme reverse mode to compute gradient
        Enzyme.autodiff(
            Enzyme.Reverse,
            loss_for_enzyme,
            Active,
            Duplicated(path, ∂path)
        )
        
        # Return cotangents for each input
        return (
            NoTangent(),  # No gradient w.r.t. function itself
            ∂path,        # Gradient w.r.t. path
            NoTangent()   # No gradient w.r.t. m (it's discrete)
        )
    end
    
    return result, sig_pullback
end

# ==============================================================================
# logsig - Compute log-signature with gradient support
# ==============================================================================

"""
    rrule(::typeof(logsig), path::AbstractMatrix{T}, basis::BasisCache) where T

Define reverse-mode AD rule for `logsig(path, basis)`.

# Arguments
- `path::AbstractMatrix{T}`: N×D matrix where each row is a point
- `basis::BasisCache`: Precomputed Lyndon basis (from `prepare(d, m)`)

# Returns
- `result`: Log-signature vector (same as `logsig(path, basis)`)
- `pullback`: Function to compute gradients given output cotangent
"""
function ChainRulesCore.rrule(
    ::typeof(logsig),
    path::AbstractMatrix{T},
    basis::BasisCache
) where {T<:Real}
    
    # Forward pass
    result = logsig(path, basis)
    
    # Pullback function
    function logsig_pullback(Δresult)
        # Handle cotangent
        Δ = unthunk(Δresult)
        
        if !(Δ isa AbstractVector{<:Real})
            Δ = convert(Vector{T}, Δ)
        end
        
        # Allocate gradient
        ∂path = zero(path)
        
        # Scalar loss for Enzyme
        function loss_for_enzyme(p)
            s = logsig(p, basis)
            return dot(s, Δ)
        end
        
        # Compute gradient via Enzyme
        Enzyme.autodiff(
            Enzyme.Reverse,
            loss_for_enzyme,
            Active,
            Duplicated(path, ∂path)
        )
        
        return (
            NoTangent(),  # No gradient w.r.t. function
            ∂path,        # Gradient w.r.t. path
            NoTangent()   # No gradient w.r.t. basis (it's a cache)
        )
    end
    
    return result, logsig_pullback
end

# ==============================================================================
# Notes for advanced users
# ==============================================================================

# For performance-critical applications:
# - These rrules are designed for high-level optimization (e.g., gradient descent)
# - For tight inner loops without AD, use `signature_path!` directly
# - The Enzyme-based gradient computation is efficient but allocates
#
# Future optimizations:
# - Could add rrule for `signature_path` if users want structured Tensor cotangents
# - Could optimize for specific loss patterns (sum, norm, etc.)