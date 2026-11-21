using ChenSignatures
using ChainRulesCore
using Enzyme
using Zygote
using LinearAlgebra
using Random
using Test

const CS = ChenSignatures

# Enzyme differentiates sig natively (it does not use the rrule), so it is an
# independent reference for the reversibility-based pullback.
function enzyme_grad(p, m, y)
    g = zeros(size(p))
    Enzyme.autodiff(Reverse, (q, yy) -> dot(sig(q, m), yy), Active,
                    Duplicated(copy(p), g), Const(y))
    return g
end

@testset "Reversible backward pass" begin
    Random.seed!(7)

    @testset "matches Enzyme (N=$N, D=$D, M=$M)" for (N, D, M) in [
        (2, 1, 1), (2, 2, 3), (5, 1, 4), (10, 2, 1), (20, 3, 4),
        (15, 4, 5), (8, 16, 4), (30, 5, 6),
    ]
        p = randn(N, D)
        y = randn(sum(D^k for k in 1:M))
        g = Zygote.gradient(q -> dot(sig(q, M), y), p)[1]
        ge = enzyme_grad(p, M, y)
        @test isapprox(g, ge; rtol=1e-10, atol=1e-10 * maximum(abs, ge))
    end

    @testset "batched rrule matches per-path, threaded and serial" begin
        P = randn(20, 3, 64)
        Y = randn(3 + 9 + 27, 64)
        per_path = cat([Zygote.gradient(q -> dot(sig(q, 3), Y[:, b]), P[:, :, b])[1]
                        for b in 1:64]...; dims=3)
        for threaded in (true, false)
            g = Zygote.gradient(q -> sum(sig(q, 3; threaded=threaded) .* Y), P)[1]
            @test g ≈ per_path
        end
    end

    @testset "Float32" begin
        p = randn(50, 3)
        y = randn(120)
        g64 = Zygote.gradient(q -> dot(sig(q, 4), y), p)[1]
        g32 = Zygote.gradient(q -> dot(sig(q, 4), Float32.(y)), Float32.(p))[1]
        @test eltype(g32) == Float32
        @test maximum(abs.(g32 .- g64)) / maximum(abs.(g64)) < 1e-4
    end

    @testset "zero cotangent" begin
        _, back = ChainRulesCore.rrule(sig, randn(5, 2), 3)
        @test back(ZeroTangent())[2] isa ZeroTangent
    end

    @testset "rewinding returns to the unit tensor" begin
        N, D, M = 5_000, 4, 5
        p = randn(N, D)
        S = CS.Tensor{Float64,D,M}()
        CS.signature_path!(S, p)
        G = CS.Tensor{Float64,D,M}()
        ws = CS.SigBackwardWorkspace{Float64,D,M}()
        CS.sig_pullback!(zeros(N, D), p, S, G, ws)
        S.coeffs[S.offsets[1] + 1] -= 1
        @test maximum(abs, S.coeffs) < 1e-8
    end

    @testset "sig_pullback! does not allocate" begin
        N, D, M = 200, 3, 4
        p = randn(N, D)
        S = CS.Tensor{Float64,D,M}(); CS.signature_path!(S, p)
        G = CS.Tensor{Float64,D,M}()
        ws = CS.SigBackwardWorkspace{Float64,D,M}()
        xb = zeros(N, D)
        f(xb, p, S, G, ws) = CS.sig_pullback!(xb, p, S, G, ws)
        f(xb, p, S, G, ws)
        @test (@allocated f(xb, p, S, G, ws)) == 0
    end
end
