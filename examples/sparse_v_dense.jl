using PathSignatures
# --- Dense → Sparse ---

# --- Dense → Sparse (levels 0..m) ---
# --- Dense → Sparse (levels 0..m), respecting padding ---
function SparseTensor(t::PathSignatures.Tensor{T}) where {T}
    d, m = t.dim, t.level
    s    = t.offsets                     # 0-based starts for levels 0..m; length m+2
    coeffs = Dict{Word,T}()

    len = 1                              # = d^k, start at k=0
    @inbounds for k in 0:m
        start = s[k+1] + 1               # 1-based start of level k block
        if k == 0
            c = t.coeffs[start]
            if !iszero(c)
                coeffs[Word()] = c
            end
        else
            # scan exactly d^k entries (skip any padding after the block)
            for p in 1:len
                c = t.coeffs[start + p - 1]
                if !iszero(c)
                    rem  = p - 1
                    idxs = Vector{Int}(undef, k)
                    base = (d == 1 ? 1 : div(len, d))   # = d^(k-1)
                    @inbounds for j in 1:k
                        q, rem = divrem(rem, base)
                        idxs[j] = q + 1
                        base = (d == 1 ? 1 : div(base, d))
                    end
                    coeffs[Word(idxs)] = c
                end
            end
        end
        len *= d                          # advance to d^(k+1)
    end
    return PathSignatures.SparseTensor{T}(coeffs, d, m)
end


# --- Sparse → Dense (levels 0..m) ---
function Tensor(t::PathSignatures.SparseTensor{T}) where {T}
    d, m = t.dim, t.level
    out  = PathSignatures.Tensor{T}(d, m)     # has padded offsets
    fill!(out.coeffs, zero(T))
    s = out.offsets

    @inbounds for (w, c) in t.coeffs
        k = length(w)
        if k == 0
            out.coeffs[s[1] + 1] = c
        else
            posm1 = 0
            @inbounds for j in 1:k
                posm1 = posm1 * d + (w.indices[j] - 1)
            end
            idx = s[k + 1] + posm1 + 1        # start of level k + position
            out.coeffs[idx] = c
        end
    end
    return out
end

# -------- Dense ↔ Dense (respects per-level padding) --------
function Base.isapprox(a::PathSignatures.Tensor{Ta}, b::PathSignatures.Tensor{Tb};
                       atol::Real=1e-8, rtol::Real=1e-8) where {Ta,Tb}
    a.dim == b.dim && a.level == b.level || return false
    sA, sB = a.offsets, b.offsets
    d, m = a.dim, a.level
    len = 1                     # = d^k
    @inbounds begin
        # level-0
        a0 = a.coeffs[sA[1] + 1]; b0 = b.coeffs[sB[1] + 1]
        if !(abs(a0 - b0) <= atol + rtol*max(abs(a0),abs(b0))); return false; end
        # levels 1..m
        for k in 1:m
            astart = sA[k + 1] + 1
            bstart = sB[k + 1] + 1
            for j in 0:len-1
                va = a.coeffs[astart + j]; vb = b.coeffs[bstart + j]
                if !(abs(va - vb) <= atol + rtol*max(abs(va),abs(vb))); return false; end
            end
            len *= d
        end
    end
    return true
end

# -------- Sparse ↔ Sparse --------
function Base.isapprox(A::PathSignatures.SparseTensor{Ta}, B::PathSignatures.SparseTensor{Tb};
                       atol::Real=1e-8, rtol::Real=1e-8) where {Ta,Tb}
    A.dim == B.dim && A.level == B.level || return false
    RA = promote_type(Ta, Tb)
    aC, bC = A.coeffs, B.coeffs

    # A's keys against B (missing treated as 0)
    @inbounds for (w, va) in aC
        vb = haskey(bC, w) ? bC[w] : zero(RA)
        vaR = RA(va); vbR = RA(vb)
        if !(abs(vaR - vbR) <= atol + rtol*max(abs(vaR),abs(vbR))); return false; end
    end
    # Keys in B that are not in A (A treated as 0)
    @inbounds for (w, vb) in bC
        if !haskey(aC, w)
            vbR = RA(vb); vaR = zero(RA)
            if !(abs(vaR - vbR) <= atol + rtol*max(abs(vaR),abs(vbR))); return false; end
        end
    end
    return true
end


# word_1 = Word([1,2,3])
word_2 = Word([1])
empty_word = Word()

tensor_1 = PathSignatures.SparseTensor(Dict(empty_word=>1.0,word_2=>2.4), 3, 8)

tensor_2 = PathSignatures.SparseTensor(Dict(empty_word=>1.0,word_2=>2.4), 3, 8)

result = PathSignatures.SparseTensor(Dict{Word,Float64}(),3,8)

tensor_result = Tensor(result)
input_1 = Tensor(tensor_1)
input_2 = Tensor(tensor_2)

PathSignatures.mul!(tensor_result, input_1, input_2)
PathSignatures.mul!(result, tensor_1, tensor_2)

@show SparseTensor(tensor_result)
@show result

PathSignatures.exp!(result,tensor_1)

vec = Vector{Float64}(undef, 3)
vec .= [2.0, 3.0, 4.5]
PathSignatures.exp!(tensor_result, vec)
PathSignatures.exp!(result,vec)
isapprox(SparseTensor(tensor_result),result)
