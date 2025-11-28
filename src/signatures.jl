# ---------------- public API ----------------

using LinearAlgebra
export sig, prepare, logsig, sig_enzyme

function sig_on_the_fly(path_matrix::Matrix{T}, m::Int) where T
    D = size(path_matrix, 2)
    N = size(path_matrix, 1)
    
    # 1. Allocate buffers manually (Enzyme friendly-ish)
    max_buffer_size = D^(m-1)
    B1 = Vector{T}(undef, max_buffer_size)
    B2 = Vector{T}(undef, max_buffer_size)
    
    # 2. Initialize Tensor
    out = Tensor{T, D, m}()
    
    # 3. Loop: Create SVector on stack -> Call Kernel
    @inbounds for i in 1:N-1
        # Fix: Convert to SVector so it matches the kernel signature
        z = SVector{D, T}(ntuple(j -> path_matrix[i+1, j] - path_matrix[i, j], D))
        
        # Call the existing optimized kernel from your library
        ChenSignatures.update_signature_horner!(out, z, B1, B2)
    end
    
    return ChenSignatures._flatten_tensor(out)
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

function signature_path(
    ::Type{Tensor{T,D,M}},
    path::Vector{SVector{D,T}},
    m::Int,
) where {T,D,M}
    @assert m == M "Requested level m=$m does not match Type level M=$M"
    out = Tensor{T,D,M}()
    signature_path!(out, path)
    return out
end

function signature_path(::Type{Tensor{T}}, path::Vector{SVector{D,T}}, m::Int) where {T,D}
    return _dispatch_sig(Tensor{T}, Val(D), Val(m), path)
end
function signature_path(::Type{Tensor{T,M}}, path::Vector{SVector{D,T}}, m::Int) where {T,D,M}
    return _dispatch_sig(Tensor{T}, Val(D), Val(M), path)
end

@generated function _dispatch_sig(::Type{Tensor{T}}, ::Val{D}, ::Val{M}, path) where {T,D,M}
    quote
        out = Tensor{T,D,M}()
        signature_path!(out, path)
        return out
    end
end

"""
    signature_path!(out, path)
Computes path signature using Block-Optimized Horner's Method.
"""
function signature_path!(
    out::Tensor{T,D,M},
    path::Vector{SVector{D,T}},
) where {T,D,M}
    @assert length(path) ≥ 2

    fill!(out.coeffs, zero(T))
    ChenSignatures._write_unit!(out)

    # Scratch buffers for Ping-Pong
    # Size: D^(M-1) floats
    max_scratch_len = M > 1 ? D^(M-1) : 1
    
    B1 = Vector{T}(undef, max_scratch_len)
    B2 = Vector{T}(undef, max_scratch_len)

    @inbounds begin
        for i in 1:length(path)-1
            Δ = path[i+1] - path[i]
            update_signature_horner!(out, Δ, B1, B2)
        end
    end

    return out
end

function signature_path!(out::AT, path::Vector{SVector{D,T}}) where {D,T,AT<:AbstractTensor{T}}
    @assert length(path) ≥ 2
    a = out; b = similar(out); seg = similar(out)
    Δ1 = path[2] - path[1]; exp!(a, Δ1)
    for i in 2:length(path)-1
        Δ = path[i+1] - path[i]; exp!(seg, Δ); mul!(b, a, seg); a, b = b, a
    end
    if a !== out; copy!(out, a); end
    return out
end