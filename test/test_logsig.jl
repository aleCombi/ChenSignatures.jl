using ChenSignatures
using ChainRulesCore
using ForwardDiff
using Zygote
using LinearAlgebra
using Random
using Test

const CSL = ChenSignatures
include("lyndon_reference.jl")
using .LyndonReference: build_L, project_to_lyndon

@testset "logsig" begin
    Random.seed!(11)

    @testset "matches the reference projection (d=$d, m=$m)" for (d, m) in [
        (1, 3), (2, 1), (2, 6), (3, 4), (4, 5), (5, 3), (8, 3),
    ]
        b = prepare(d, m)
        lyn, L, _ = build_L(d, m)
        @test [w.indices for w in b.lynds] == [w.indices for w in lyn]
        p = randn(30, d)
        S = CSL.signature_path(CSL.Tensor{Float64}, p, m)
        lt = CSL.log(S)
        @test logsig(p, b) ≈ project_to_lyndon(lt, lyn, L) atol=1e-12
        @test logsig(p, b; coords=:expanded) ≈ CSL._flatten_tensor(lt) atol=1e-12
        @test logsig(p, b; coords=:words) == logsig(p, b; coords=:expanded)[b.idx]
    end

    @testset "output lengths and errors" begin
        b = prepare(3, 4)
        p = randn(10, 3)
        @test length(logsig(p, b)) == 32
        @test length(logsig(p, b; coords=:words)) == 32
        @test length(logsig(p, b; coords=:expanded)) == 3 + 9 + 27 + 81
        @test_throws ArgumentError logsig(p, b; coords=:nope)
        @test_throws ArgumentError logsig(randn(10, 2), b)
        @test eltype(logsig(Float32.(p), b)) == Float32
    end

    @testset "batched equals per-path (coords=$coords, threaded=$thr)" for
            coords in (:lyndon, :words, :expanded), thr in (true, false)
        b = prepare(3, 4)
        P = randn(12, 3, 16)
        Y = logsig(P, b; coords=coords, threaded=thr)
        @test Y ≈ hcat([logsig(P[:, :, i], b; coords=coords) for i in 1:16]...)
    end

    @testset "gradient matches ForwardDiff (N=$N, d=$d, m=$m, $coords)" for
            (N, d, m) in [(2, 1, 3), (5, 2, 1), (10, 2, 5), (20, 3, 4), (8, 8, 3)],
            coords in (:lyndon, :words, :expanded)
        b = prepare(d, m)
        p = randn(N, d)
        y, back = ChainRulesCore.rrule(logsig, p, b; coords=coords)
        ȳ = randn(length(y))
        g = back(ȳ)[2]
        gf = ForwardDiff.gradient(q -> dot(logsig(q, b; coords=coords), ȳ), p)
        @test g ≈ gf rtol=1e-10 atol=1e-12
    end

    @testset "batched gradient equals per-path (threaded=$thr)" for thr in (true, false)
        b = prepare(3, 4)
        P = randn(12, 3, 64)
        Ȳ = randn(32, 64)
        G = Zygote.gradient(q -> sum(logsig(q, b; threaded=thr) .* Ȳ), P)[1]
        Gs = cat([Zygote.gradient(q -> dot(logsig(q, b), Ȳ[:, i]), P[:, :, i])[1]
                  for i in 1:64]...; dims=3)
        @test G ≈ Gs
    end

    @testset "Zygote with coords keyword" begin
        b = prepare(2, 4)
        p = randn(15, 2)
        g = Zygote.gradient(q -> sum(abs2, logsig(q, b; coords=:words)), p)[1]
        gf = ForwardDiff.gradient(q -> sum(abs2, logsig(q, b; coords=:words)), p)
        @test g ≈ gf rtol=1e-10
    end

    @testset "prepare stays small at larger sizes" begin
        b = prepare(8, 5)
        @test length(b.lynds) == 7764
        @test Base.summarysize(b.L) < 2^21          # dense was 460 MB
    end
end
