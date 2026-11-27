# Entry points for the Python package `chen-signatures` (internal, not Julia API).
#
# Python passes raw pointers to column-major buffers it owns, plus plain integers, and
# allocates the outputs itself; it calls through C function pointers (`cpointers`, via
# ctypes), so neither PythonCall nor any JIT compilation is involved: the functions
# below are compiled by the precompile workload. `prec` is 32 or 64 (the float type).
# Julia objects Python holds on to (Lyndon bases, pullbacks) live in a handle registry;
# errors are returned as -1, with the message from `c_lasterror`.
module PyAPI

using ..ChenSignatures: ChenSignatures, sig, logsig, sig_gram, LinearKernel, RBFKernel
using ..ChenSignatures: BasisCache, _logsig_length
using ChainRulesCore: rrule, unthunk, AbstractZero

const COORDS = (:lyndon, :words, :expanded)

wrap(::Type{T}, p::Integer, dims::Integer...) where {T} =
    unsafe_wrap(Array, Ptr{T}(UInt(p)), map(Int, dims))
path(::Type{T}, p, N, d, B) where {T} = B == 0 ? wrap(T, p, N, d) : wrap(T, p, N, d, B)
static_kernel(code::Integer, σ::Real) =
    code == 0 ? LinearKernel() : code == 1 ? RBFKernel(Float64(σ)) :
    throw(ArgumentError("unknown static kernel code $code"))

# The float type is a runtime value: branch once, at the boundary.
@inline function byprec(f, prec::Integer)
    prec == 64 && return f(Float64)
    prec == 32 && return f(Float32)
    throw(ArgumentError("precision must be 32 or 64, got $prec"))
end

sig_length(d::Integer, m::Integer) = sum(d^k for k in 1:m)
prepare(d::Integer, m::Integer) = ChenSignatures.prepare(Int(d), Int(m))
logsig_length(b::BasisCache, coords::Integer) = _logsig_length(b, COORDS[coords])
basis_d(b::BasisCache) = b.d
basis_m(b::BasisCache) = b.m

# ---------------------------------------------------------------------------
# Forward only: results written into `out`
# ---------------------------------------------------------------------------

sig!(prec, out, x, N, d, B, m, threaded) = byprec(prec) do T
    y = B == 0 ? sig(path(T, x, N, d, 0), Int(m)) : sig(path(T, x, N, d, B), Int(m); threaded)
    copyto!(wrap(T, out, length(y)), y)
    nothing
end

logsig!(prec, out, x, N, d, B, basis::BasisCache, coords, threaded) = byprec(prec) do T
    c = COORDS[coords]
    y = B == 0 ? logsig(path(T, x, N, d, 0), basis; coords = c) :
                 logsig(path(T, x, N, d, B), basis; coords = c, threaded)
    copyto!(wrap(T, out, length(y)), y)
    nothing
end

# Gram matrix (Bx × By, or Bx × Bx when sym). m > 0: truncated at level m; otherwise
# the PDE kernel with dyadic order λ and static kernel (kcode, σ).
function _gram(X, Y, sym, λ, kcode, σ, m, threaded)
    if m > 0
        return sym ? sig_gram(X, Int(m); threaded) : sig_gram(X, Y, Int(m); threaded)
    end
    κ = static_kernel(kcode, σ)
    return sym ? sig_gram(X; dyadic_order = Int(λ), static_kernel = κ, threaded) :
                 sig_gram(X, Y; dyadic_order = Int(λ), static_kernel = κ, threaded)
end

sig_gram!(prec, out, x, Nx, Bx, y, Ny, By, d, sym, λ, kcode, σ, m, threaded) = byprec(prec) do T
    X = wrap(T, x, Nx, d, Bx)
    Y = sym ? X : wrap(T, y, Ny, d, By)
    K = _gram(X, Y, sym, λ, kcode, σ, m, threaded)
    copyto!(wrap(T, out, length(K)), K)
    nothing
end

# ---------------------------------------------------------------------------
# Forward with a pullback (for autograd): the inputs are copied, since the pullback
# outlives the Python buffers; `pull!` writes the cotangents into Python buffers.
# ---------------------------------------------------------------------------

struct Pullback{T,F}
    back::F
    ydims::Tuple
    xdims::Tuple            # dims of each differentiable argument, in order
end

function _record(::Type{T}, out, y, back, xdims) where {T}
    copyto!(wrap(T, out, length(y)), y)
    return Pullback{T,typeof(back)}(back, size(y), xdims)
end

sig_fwd!(prec, out, x, N, d, B, m, threaded) = byprec(prec) do T
    X = copy(path(T, x, N, d, B))
    y, back = B == 0 ? rrule(sig, X, Int(m)) : rrule(sig, X, Int(m); threaded)
    _record(T, out, y, back, (size(X),))
end

logsig_fwd!(prec, out, x, N, d, B, basis::BasisCache, coords, threaded) = byprec(prec) do T
    X = copy(path(T, x, N, d, B))
    c = COORDS[coords]
    y, back = B == 0 ? rrule(logsig, X, basis; coords = c) :
                       rrule(logsig, X, basis; coords = c, threaded)
    _record(T, out, y, back, (size(X),))
end

# PDE kernel only (the truncated kernels have no handwritten adjoint).
sig_gram_fwd!(prec, out, x, Nx, Bx, y, Ny, By, d, sym, λ, kcode, σ, threaded) = byprec(prec) do T
    X = copy(wrap(T, x, Nx, d, Bx))
    κ = static_kernel(kcode, σ)
    if sym
        K, back = rrule(sig_gram, X; dyadic_order = Int(λ), static_kernel = κ, threaded)
        return _record(T, out, K, back, (size(X),))
    end
    Y = copy(wrap(T, y, Ny, d, By))
    K, back = rrule(sig_gram, X, Y; dyadic_order = Int(λ), static_kernel = κ, threaded)
    _record(T, out, K, back, (size(X), size(Y)))
end

function pull!(pb::Pullback{T}, ybar, outs::Integer...) where {T}
    ȳ = copy(wrap(T, ybar, pb.ydims...))
    g = pb.back(ȳ)
    for (i, o) in enumerate(outs)
        o == 0 && continue                       # cotangent not requested
        dst = wrap(T, o, pb.xdims[i]...)
        gi = unthunk(g[i + 1])
        gi isa AbstractZero ? fill!(dst, zero(T)) : copyto!(dst, gi)
    end
    nothing
end

# ---------------------------------------------------------------------------
# C ABI: what the Python package calls (through ctypes)
# ---------------------------------------------------------------------------

const REGISTRY = Dict{Int,Any}()
const REGLOCK = ReentrantLock()
const NEXTHANDLE = Ref(0)
const LASTERROR = Ref("")

function register!(obj)
    lock(REGLOCK) do
        h = (NEXTHANDLE[] += 1)
        REGISTRY[h] = obj
        h
    end
end
lookup(h) = lock(() -> REGISTRY[Int(h)], REGLOCK)
release!(h) = lock(() -> (delete!(REGISTRY, Int(h)); nothing), REGLOCK)

# Run f(); 0 on success, -1 (message kept for c_lasterror) on an exception.
function guard(f)
    try
        f()
        return Cint(0)
    catch e
        LASTERROR[] = sprint(showerror, e)
        return Cint(-1)
    end
end
# As guard, storing a handle to f()'s result in hout[].
guard_handle(f, hout::Ptr{Int}) = guard(() -> unsafe_store!(hout, register!(f())))

const I = Int
const P = Ptr{Cvoid}
addr(p::Ptr) = Int(UInt(p))

c_sig(prec::I, out::P, x::P, N::I, d::I, B::I, m::I, th::I) =
    guard(() -> sig!(prec, addr(out), addr(x), N, d, B, m, th != 0))
c_prepare(d::I, m::I, hout::Ptr{Int}) = guard_handle(() -> prepare(d, m), hout)
c_logsig_length(h::I, coords::I, lout::Ptr{Int}) =
    guard(() -> unsafe_store!(lout, logsig_length(lookup(h), coords)))
c_logsig(prec::I, out::P, x::P, N::I, d::I, B::I, h::I, coords::I, th::I) =
    guard(() -> logsig!(prec, addr(out), addr(x), N, d, B, lookup(h), coords, th != 0))
c_sig_gram(prec::I, out::P, x::P, Nx::I, Bx::I, y::P, Ny::I, By::I, d::I, sym::I,
           λ::I, kcode::I, σ::Float64, m::I, th::I) =
    guard(() -> sig_gram!(prec, addr(out), addr(x), Nx, Bx, addr(y), Ny, By, d, sym != 0,
                          λ, kcode, σ, m, th != 0))
c_sig_fwd(prec::I, out::P, x::P, N::I, d::I, B::I, m::I, th::I, hout::Ptr{Int}) =
    guard_handle(() -> sig_fwd!(prec, addr(out), addr(x), N, d, B, m, th != 0), hout)
c_logsig_fwd(prec::I, out::P, x::P, N::I, d::I, B::I, h::I, coords::I, th::I, hout::Ptr{Int}) =
    guard_handle(() -> logsig_fwd!(prec, addr(out), addr(x), N, d, B, lookup(h), coords, th != 0), hout)
c_sig_gram_fwd(prec::I, out::P, x::P, Nx::I, Bx::I, y::P, Ny::I, By::I, d::I, sym::I,
               λ::I, kcode::I, σ::Float64, th::I, hout::Ptr{Int}) =
    guard_handle(() -> sig_gram_fwd!(prec, addr(out), addr(x), Nx, Bx, addr(y), Ny, By, d,
                                     sym != 0, λ, kcode, σ, th != 0), hout)
# Pullback h applied to the cotangent at ybar; cotangents written to o1 (and o2 if nonzero).
c_pull(h::I, ybar::P, o1::P, o2::P) =
    guard(() -> (pb = lookup(h); o2 == C_NULL ? pull!(pb, addr(ybar), addr(o1)) :
                                                pull!(pb, addr(ybar), addr(o1), addr(o2))))
c_release(h::I) = guard(() -> release!(h))
c_nthreads() = Threads.nthreads()
# Copy the last error message into buf (at most n bytes); returns its full length.
function c_lasterror(buf::Ptr{UInt8}, n::I)
    msg = codeunits(LASTERROR[])
    k = min(n, length(msg))
    k > 0 && GC.@preserve msg unsafe_copyto!(buf, pointer(msg), k)
    return length(msg)
end

# The function pointer table, in the order chen/_julia.py expects. @cfunction pointers
# are only valid in the running process: filled by __init__.
const CNAMES = ("sig", "prepare", "logsig_length", "logsig", "sig_gram", "sig_fwd",
                "logsig_fwd", "sig_gram_fwd", "pull", "release", "nthreads", "lasterror")
const CPTRS = Vector{Ptr{Cvoid}}(undef, length(CNAMES))

function cpointers()
    return Ptr{Cvoid}[
        @cfunction(c_sig, Cint, (I, P, P, I, I, I, I, I)),
        @cfunction(c_prepare, Cint, (I, I, Ptr{Int})),
        @cfunction(c_logsig_length, Cint, (I, I, Ptr{Int})),
        @cfunction(c_logsig, Cint, (I, P, P, I, I, I, I, I, I)),
        @cfunction(c_sig_gram, Cint, (I, P, P, I, I, P, I, I, I, I, I, I, Float64, I, I)),
        @cfunction(c_sig_fwd, Cint, (I, P, P, I, I, I, I, I, Ptr{Int})),
        @cfunction(c_logsig_fwd, Cint, (I, P, P, I, I, I, I, I, I, Ptr{Int})),
        @cfunction(c_sig_gram_fwd, Cint, (I, P, P, I, I, P, I, I, I, I, I, I, Float64, I, Ptr{Int})),
        @cfunction(c_pull, Cint, (I, P, P, P)),
        @cfunction(c_release, Cint, (I,)),
        @cfunction(c_nthreads, Int, ()),
        @cfunction(c_lasterror, Int, (Ptr{UInt8}, I)),
    ]
end

function __init__()
    copyto!(CPTRS, cpointers())
end

# Address of the pointer table and its length, for Python (read with ctypes).
cpointers_table() = (Int(UInt(pointer(CPTRS))), length(CPTRS))

# ---------------------------------------------------------------------------
# Precompile every entry point, both precisions, for small common shapes.
# ---------------------------------------------------------------------------

vp(a) = Ptr{Cvoid}(pointer(a))

# Every C entry point, both precisions, small shapes (the (d, m) specialisations of
# sig/logsig are compiled per shape: the common ones here).
function _workload(shapes)
    h = Ref(0)
    for prec in (32, 64)
        T = prec == 32 ? Float32 : Float64
        for (d, m) in shapes
            p = T[sin(i + 2j) for i in 1:4, j in 1:d]
            P = T[sin(i + 2j + k) for i in 1:4, j in 1:d, k in 1:2]
            L = sig_length(d, m)
            o, O = zeros(T, L), zeros(T, L, 2)
            GC.@preserve p P o O begin
                for th in (0, 1)
                    c_sig(prec, vp(o), vp(p), 4, d, 0, m, th)
                    c_sig(prec, vp(O), vp(P), 4, d, 2, m, th)
                end
                c_sig_fwd(prec, vp(o), vp(p), 4, d, 0, m, 1, Base.unsafe_convert(Ptr{Int}, h))
                c_pull(h[], vp(o), vp(p), C_NULL); c_release(h[])
                c_sig_fwd(prec, vp(O), vp(P), 4, d, 2, m, 1, Base.unsafe_convert(Ptr{Int}, h))
                c_pull(h[], vp(O), vp(P), C_NULL); c_release(h[])
                hb = Ref(0)
                c_prepare(d, m, Base.unsafe_convert(Ptr{Int}, hb))
                for coords in 1:3
                    lr = Ref(0)
                    c_logsig_length(hb[], coords, Base.unsafe_convert(Ptr{Int}, lr))
                    lo, LO = zeros(T, lr[]), zeros(T, lr[], 2)
                    GC.@preserve lo LO begin
                        for th in (0, 1)
                            c_logsig(prec, vp(lo), vp(p), 4, d, 0, hb[], coords, th)
                            c_logsig(prec, vp(LO), vp(P), 4, d, 2, hb[], coords, th)
                        end
                        c_logsig_fwd(prec, vp(lo), vp(p), 4, d, 0, hb[], coords, 1, Base.unsafe_convert(Ptr{Int}, h))
                        c_pull(h[], vp(lo), vp(p), C_NULL); c_release(h[])
                        c_logsig_fwd(prec, vp(LO), vp(P), 4, d, 2, hb[], coords, 1, Base.unsafe_convert(Ptr{Int}, h))
                        c_pull(h[], vp(LO), vp(P), C_NULL); c_release(h[])
                    end
                end
                c_release(hb[])
            end
        end
        # kernels: not specialised on the shape
        P = T[sin(i + 2j + k) for i in 1:4, j in 1:2, k in 1:3]
        K = zeros(T, 3, 3)
        GC.@preserve P K for sym in (0, 1), th in (0, 1)
            for (kcode, m) in ((0, 0), (1, 0), (0, 3))
                c_sig_gram(prec, vp(K), vp(P), 4, 3, vp(P), 4, 3, 2, sym, 0, kcode, 1.0, m, th)
            end
            for kcode in (0, 1)
                c_sig_gram_fwd(prec, vp(K), vp(P), 4, 3, vp(P), 4, 3, 2, sym, 0, kcode, 1.0, th,
                               Base.unsafe_convert(Ptr{Int}, h))
                c_pull(h[], vp(K), vp(P), sym == 1 ? C_NULL : vp(P)); c_release(h[])
            end
        end
    end
    # error path and the pointer table
    lr = Ref(0)
    c_logsig_length(-1, 1, Base.unsafe_convert(Ptr{Int}, lr))     # unknown handle: error path
    buf = zeros(UInt8, 64)
    GC.@preserve buf c_lasterror(pointer(buf), 64)
    cpointers(); c_nthreads(); cpointers_table()
    empty!(REGISTRY); LASTERROR[] = ""; NEXTHANDLE[] = 0
    nothing
end

end
