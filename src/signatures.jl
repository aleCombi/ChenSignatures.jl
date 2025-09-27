# ---------------- public API ----------------

function signature_path(::Type{AT}, path::Vector{SVector{D,T}}, m::Int) where {D,T, AT <: AbstractTensor{T}}
    out = AT(D,m)
    return signature_path!(out, path)
end

function signature_path!(out::AT, path::Vector{SVector{D,T}}) where {D,T, AT <: AbstractTensor{T}}
    d = D
    a = similar(out)
    b = similar(out)
    segment_tensor = similar(out)
    displacement = Vector{T}(undef, d)

    displacement .= path[2] - path[1] 
    exp!(a, displacement)

    for i in 2:length(path)-1
        displacement .= path[i+1] - path[i] 
        exp!(segment_tensor, displacement)
        mul!(b, a, segment_tensor)
        a, b = b, a
    end

    return a
end

function signature_path!(
    out::AT, 
    path::Matrix{T}  # (n_steps+1) × D
) where {T, AT <: AbstractTensor{T}}
    n_steps, d = size(path)
    n_steps -= 1  # actual number of steps
    
    a = similar(out)
    b = similar(out)
    segment_tensor = similar(out)
    displacement = Vector{T}(undef, d)
    
    # First segment
    @inbounds for j in 1:d
        displacement[j] = path[2, j] - path[1, j]
    end
    exp!(a, displacement)
    
    # Remaining segments
    @inbounds for i in 2:n_steps
        # Compute displacement
        for j in 1:d
            displacement[j] = path[i+1, j] - path[i, j]
        end
        exp!(segment_tensor, displacement)
        mul!(b, a, segment_tensor)
        a, b = b, a
    end
    
    return a
end

# === Batch Signature Functions ===

# For Vector{Vector{SVector}} (SVectorEnsemble)
function batch_signatures!(
    outs::Vector{AT},
    ensemble::SVectorEnsemble{D,T}
) where {D,T,AT<:AbstractTensor{T}}
    # Single set of reusable buffers
    a = similar(outs[1])
    b = similar(outs[1])
    segment = similar(outs[1])
    displacement = Vector{T}(undef, D)
    
    # Process each path sequentially
    for p in 1:ensemble.n_paths
        path = ensemble.paths[p]
        
        # First segment
        @inbounds for j in 1:D
            displacement[j] = path[2][j] - path[1][j]
        end
        exp!(a, displacement)
        
        # Remaining segments
        @inbounds for i in 2:ensemble.n_steps
            for j in 1:D
                displacement[j] = path[i+1][j] - path[i][j]
            end
            exp!(segment, displacement)
            mul!(b, a, segment)
            a, b = b, a
        end
        
        copy!(outs[p], a)
    end
    
    return outs
end

# For 3D Array (ArrayEnsemble) - Path-major access pattern
function batch_signatures!(
    outs::Vector{AT},
    ensemble::ArrayEnsemble{T}
) where {T,AT<:AbstractTensor{T}}
    d = ensemble.dim
    
    # Single set of reusable buffers
    a = similar(outs[1])
    b = similar(outs[1])
    segment = similar(outs[1])
    displacement = Vector{T}(undef, d)
    
    # Process each path sequentially
    for p in 1:ensemble.n_paths
        # First segment
        @inbounds for j in 1:d
            displacement[j] = ensemble.data[2, j, p] - ensemble.data[1, j, p]
        end
        exp!(a, displacement)
        
        # Remaining segments
        @inbounds for i in 2:ensemble.n_steps
            for j in 1:d
                displacement[j] = ensemble.data[i+1, j, p] - ensemble.data[i, j, p]
            end
            exp!(segment, displacement)
            mul!(b, a, segment)
            a, b = b, a
        end
        
        copy!(outs[p], a)
    end
    
    return outs
end

# Convenience function to create output array for batch signatures
function prepare_signature_outputs(::Type{AT}, ensemble, level::Int) where {AT<:AbstractTensor}
    D = get_dimension(ensemble)
    T = eltype(AT)
    return [AT(D, level) for _ in 1:ensemble.n_paths]
end

# High-level batch signature computation
function batch_signatures(::Type{AT}, ensemble, level::Int) where {AT<:AbstractTensor}
    outs = prepare_signature_outputs(AT, ensemble, level)
    return batch_signatures!(outs, ensemble)
end

export batch_signatures!, batch_signatures, prepare_signature_outputs