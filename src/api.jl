using LinearAlgebra
export sig, prepare, logsig, sig_enzyme

# --- 1. Signature (sig) ---
function sig(path::AbstractMatrix{T}, m::Int) where T
    N, d = size(path)
    sv_path = [SVector{d, T}(path[i,:]) for i in 1:N]
    
    # Use Tensor{T} to match input type
    tensor = signature_path(Tensor{T}, sv_path, m)
    
    return _flatten_tensor(tensor)
end

function sig_enzyme(path_matrix::Matrix{Float64}, m::Int)
    D = size(path_matrix, 2)
    M = m
    N = size(path_matrix, 1)
    
    # Allocate working buffers
    max_buffer_size = D^(M-1)
    B1 = Vector{Float64}(undef, max_buffer_size)
    B2 = Vector{Float64}(undef, max_buffer_size)
    
    # Initialize signature tensor
    a = Tensor{Float64, D, M}()
    
    # Process each segment
    @inbounds for i in 1:N-1
        # Create SVector for displacement
        z = SVector{D,Float64}(ntuple(j -> path_matrix[i+1, j] - path_matrix[i, j], D))
        
        update_signature_horner_enzyme!(a, z, B1, B2)
    end
    
    return _flatten_tensor(a)
end


# --- 2. Preparation (prepare) ---
struct BasisCache{T}
    d::Int
    m::Int
    lynds::Vector{Algebra.Word}
    L::Matrix{T} 
end

function prepare(d::Int, m::Int)
    lynds, L, _ = Algebra.build_L(d, m)
    return BasisCache(d, m, lynds, L)
end

# --- 3. Log Signature (logsig) ---
function logsig(path::AbstractMatrix{T}, basis::BasisCache) where T
    N, d = size(path)
    @assert d == basis.d "Dimension mismatch between path and basis"
    
    sv_path = [SVector{d, T}(path[i,:]) for i in 1:N]
    
    sig_tensor = signature_path(Tensor{T}, sv_path, basis.m)
    log_tensor = ChenSignatures.log(sig_tensor)
    
    return Algebra.project_to_lyndon(log_tensor, basis.lynds, basis.L)
end

# --- Helper: Flatten Tensor to Array ---
function _flatten_tensor(t::Tensor{T,D,M}) where {T,D,M}
    total_len = t.offsets[end] - t.offsets[2] 
    out = Vector{T}(undef, total_len)
    
    current_idx = 1
    
    for k in 1:M
        start_offset = t.offsets[k+1]
        len = D^k
        copyto!(out, current_idx, t.coeffs, start_offset + 1, len)
        current_idx += len
    end
    return out
end