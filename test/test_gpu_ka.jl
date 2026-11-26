# The KernelAbstractions extension (GPU sig_gram) run on JLArrays, the CPU reference
# implementation of the GPU array interface, so CI covers it without a GPU.
using JLArrays, KernelAbstractions, GPUArraysCore, ChainRulesCore, Zygote, Random, Test

@testset "GPU sig_gram (KernelAbstractions ext, JLArrays)" begin
    Random.seed!(7)
    paths(N, d, B) = 0.1 .* cumsum(randn(N, d, B); dims = 1)

    @testset "Nx=$Nx Ny=$Ny d=$d λ=$λ" for (Nx, Ny, d, λ) in
            ((10, 10, 3, 0), (13, 8, 2, 0), (7, 11, 4, 1), (6, 6, 3, 2))
        X, Y, W = paths(Nx, d, 3), paths(Ny, d, 4), randn(3, 4)
        K = sig_gram(JLArray(X), JLArray(Y); dyadic_order = λ)
        @test K isa JLArray
        @test Array(K) ≈ sig_gram(X, Y; dyadic_order = λ) rtol = 1e-12
        _, pb = ChainRulesCore.rrule(sig_gram, JLArray(X), JLArray(Y); dyadic_order = λ)
        _, X̄, Ȳ = pb(JLArray(W))
        _, pbc = ChainRulesCore.rrule(sig_gram, X, Y; dyadic_order = λ)
        _, cX̄, cȲ = pbc(W)
        @test Array(X̄) ≈ cX̄ rtol = 1e-10
        @test Array(Ȳ) ≈ cȲ rtol = 1e-10
    end

    @testset "symmetric sig_gram(X)" begin
        X, W = paths(9, 3, 4), randn(4, 4)
        @test Array(sig_gram(JLArray(X))) ≈ sig_gram(X) rtol = 1e-12
        _, pb = ChainRulesCore.rrule(sig_gram, JLArray(X))
        _, pbc = ChainRulesCore.rrule(sig_gram, X)
        @test Array(pb(JLArray(W))[2]) ≈ pbc(W)[2] rtol = 1e-10
    end

    @testset "Float32" begin
        X, Y = Float32.(paths(10, 2, 2)), Float32.(paths(12, 2, 3))
        K = sig_gram(JLArray(X), JLArray(Y))
        @test eltype(K) == Float32
        @test Array(K) ≈ sig_gram(Float64.(X), Float64.(Y)) rtol = 1e-5
    end

    @testset "Zygote through the rrule" begin
        X, Y, W = paths(8, 2, 2), paths(9, 2, 3), randn(2, 3)
        g = Zygote.gradient((x, y) -> sum(JLArray(W) .* sig_gram(x, y)), JLArray(X), JLArray(Y))
        gc = Zygote.gradient((x, y) -> sum(W .* sig_gram(x, y)), X, Y)
        @test Array(g[1]) ≈ gc[1] rtol = 1e-10
        @test Array(g[2]) ≈ gc[2] rtol = 1e-10
    end

    @testset "unsupported arguments" begin
        X = JLArray(paths(5, 2, 2))
        @test_throws ArgumentError sig_gram(X, X; static_kernel = RBFKernel(1.0))
        @test_throws ArgumentError sig_gram(X; dyadic_order = -1)
        @test_throws DimensionMismatch sig_gram(X, JLArray(paths(5, 3, 2)))
    end
end
