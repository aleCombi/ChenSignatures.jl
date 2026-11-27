# The C entry points the Python package calls (ChenSignatures.PyAPI), called from Julia
# with raw pointers exactly as ctypes does.
using ChenSignatures, ChainRulesCore, Test
const PA = ChenSignatures.PyAPI

@testset "Python C API" begin
    vp(a) = Ptr{Cvoid}(pointer(a))
    href() = Ref(0)
    hp(r) = Base.unsafe_convert(Ptr{Int}, r)

    X = 0.1 .* cumsum(randn(20, 3, 4); dims = 1)
    out = zeros(sum(3^k for k in 1:4), 4)
    GC.@preserve X out @test PA.c_sig(64, vp(out), vp(X), 20, 3, 4, 4, 1) == 0
    @test out ≈ sig(X, 4)

    X32 = Float32.(X)
    out32 = zeros(Float32, size(out))
    GC.@preserve X32 out32 PA.c_sig(32, vp(out32), vp(X32), 20, 3, 4, 4, 0)
    @test out32 ≈ sig(X32, 4)

    # basis handle, logsig, error path
    hb = href()
    @test PA.c_prepare(3, 4, hp(hb)) == 0
    L = href()
    PA.c_logsig_length(hb[], 1, hp(L))
    lo = zeros(L[], 4)
    GC.@preserve X lo PA.c_logsig(64, vp(lo), vp(X), 20, 3, 4, hb[], 1, 1)
    @test lo ≈ logsig(X, prepare(3, 4))
    @test PA.c_logsig_length(-7, 1, hp(L)) == -1
    buf = zeros(UInt8, 200)
    n = GC.@preserve buf PA.c_lasterror(pointer(buf), 200)
    @test occursin("KeyError", String(buf[1:min(n, 200)]))
    @test PA.c_release(hb[]) == 0

    # Gram and its pullback
    Y = 0.1 .* cumsum(randn(15, 3, 5); dims = 1)
    K = zeros(4, 5)
    W = randn(4, 5)
    X̄, Ȳ = zeros(size(X)), zeros(size(Y))
    h = href()
    GC.@preserve X Y K W X̄ Ȳ begin
        @test PA.c_sig_gram_fwd(64, vp(K), vp(X), 20, 4, vp(Y), 15, 5, 3, 0, 1, 0, 1.0, 1, hp(h)) == 0
        @test PA.c_pull(h[], vp(W), vp(X̄), vp(Ȳ)) == 0
    end
    PA.c_release(h[])
    _, pb = ChainRulesCore.rrule(sig_gram, X, Y; dyadic_order = 1)
    _, gX, gY = pb(W)
    @test K ≈ sig_gram(X, Y; dyadic_order = 1)
    @test X̄ ≈ gX
    @test Ȳ ≈ gY
    @test isempty(PA.REGISTRY)

    addr, len = PA.cpointers_table()
    @test len == length(PA.CNAMES) && addr != 0
end
