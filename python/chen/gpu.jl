# CUDA entry points for chen.torch, on the same C interface as ChenSignatures.PyAPI:
# torch CUDA tensors cross as device pointers and are wrapped as CuArrays without a copy;
# sig_gram then runs ChenSignatures' GPU kernels (KernelAbstractions extension).
# Pullbacks share PyAPI's handle registry. Loaded by chen._julia.gpu_api().
module ChenPyGPU

using CUDA
using ChenSignatures: sig_gram
using ChenSignatures.PyAPI: Pullback, byprec, static_kernel, rrule, unthunk, AbstractZero,
                            guard, guard_handle, lookup, addr

const I = Int
const P = Ptr{Cvoid}

wrapc(::Type{T}, p::Integer, dims::Integer...) where {T} =
    unsafe_wrap(CuArray, CuPtr{T}(UInt(p)), map(Int, dims))

# Blocking synchronisation: never yield to Julia's scheduler from a foreign thread.
sync() = CUDA.synchronize(; blocking = true)

function _gram(X, Y, sym, λ, kcode, σ)
    κ = static_kernel(kcode, σ)
    return sym ? sig_gram(X; dyadic_order = Int(λ), static_kernel = κ) :
                 sig_gram(X, Y; dyadic_order = Int(λ), static_kernel = κ)
end

sig_gram!(prec, out, x, Nx, Bx, y, Ny, By, d, sym, λ, kcode, σ) = byprec(prec) do T
    X = wrapc(T, x, Nx, d, Bx)
    Y = sym ? X : wrapc(T, y, Ny, d, By)
    K = _gram(X, Y, sym, λ, kcode, σ)
    copyto!(wrapc(T, out, size(K)...), K)
    sync()
    nothing
end

sig_gram_fwd!(prec, out, x, Nx, Bx, y, Ny, By, d, sym, λ, kcode, σ) = byprec(prec) do T
    X = copy(wrapc(T, x, Nx, d, Bx))          # the pullback outlives the torch buffers
    κ = static_kernel(kcode, σ)
    if sym
        K, back = rrule(sig_gram, X; dyadic_order = Int(λ), static_kernel = κ)
        xdims = (size(X),)
    else
        Y = copy(wrapc(T, y, Ny, d, By))
        K, back = rrule(sig_gram, X, Y; dyadic_order = Int(λ), static_kernel = κ)
        xdims = (size(X), size(Y))
    end
    copyto!(wrapc(T, out, size(K)...), K)
    sync()
    Pullback{T,typeof(back)}(back, size(K), xdims)
end

function pull!(pb::Pullback{T}, ybar, outs::Integer...) where {T}
    ȳ = copy(wrapc(T, ybar, pb.ydims...))
    g = pb.back(ȳ)
    for (i, o) in enumerate(outs)
        o == 0 && continue
        dst = wrapc(T, o, pb.xdims[i]...)
        gi = unthunk(g[i + 1])
        gi isa AbstractZero ? fill!(dst, zero(T)) : copyto!(dst, gi)
    end
    sync()
    nothing
end

c_sig_gram(prec::I, out::P, x::P, Nx::I, Bx::I, y::P, Ny::I, By::I, d::I, sym::I,
           λ::I, kcode::I, σ::Float64) =
    guard(() -> sig_gram!(prec, addr(out), addr(x), Nx, Bx, addr(y), Ny, By, d, sym != 0, λ, kcode, σ))
c_sig_gram_fwd(prec::I, out::P, x::P, Nx::I, Bx::I, y::P, Ny::I, By::I, d::I, sym::I,
               λ::I, kcode::I, σ::Float64, hout::Ptr{Int}) =
    guard_handle(() -> sig_gram_fwd!(prec, addr(out), addr(x), Nx, Bx, addr(y), Ny, By, d,
                                     sym != 0, λ, kcode, σ), hout)
c_pull(h::I, ybar::P, o1::P, o2::P) =
    guard(() -> (pb = lookup(h); o2 == C_NULL ? pull!(pb, addr(ybar), addr(o1)) :
                                                pull!(pb, addr(ybar), addr(o1), addr(o2))))

const CPTRS = Ptr{Cvoid}[
    @cfunction(c_sig_gram, Cint, (I, P, P, I, I, P, I, I, I, I, I, I, Float64)),
    @cfunction(c_sig_gram_fwd, Cint, (I, P, P, I, I, P, I, I, I, I, I, I, Float64, Ptr{Int})),
    @cfunction(c_pull, Cint, (I, P, P, P)),
]
cpointers_table() = (Int(UInt(pointer(CPTRS))), length(CPTRS))

end
