using ChenSignatures
using ForwardDiff
using Zygote
using LinearAlgebra
using Random
using Test

@testset "signature kernels" begin
    Random.seed!(21)
    walk(n, d, b...) = 0.4 .* cumsum(randn(n, d, b...); dims=1)

    @testset "PDE kernel converges to the signature inner product" begin
        x, y = walk(10, 2), walk(7, 2)
        exact = sig_kernel(x, y, 14)
        errs = [abs(sig_kernel(x, y; dyadic_order=λ) - exact) for λ in 0:5]
        @test errs[end] < 1e-4
        # second order: once refined, each step cuts the error by about 4
        @test all(3 .< errs[end-2:end-1] ./ errs[end-1:end] .< 5)
    end

    @testset "one segment each: Σ cᵏ/(k!)²" begin
        x = [0.0 0.0; 0.3 -0.2]; y = [0.0 0.0; 0.5 0.4]
        c = dot(x[2, :], y[2, :])
        exact = sum(c^k / factorial(k)^2 for k in 0:15)
        @test sig_kernel(x, y; dyadic_order=6) ≈ exact rtol=1e-6
        @test sig_kernel(x, y, 15) ≈ exact
    end

    @testset "degenerate and invalid input" begin
        @test sig_kernel(zeros(1, 2), randn(5, 2)) == 1
        @test_throws ArgumentError sig_kernel(randn(3, 2), randn(3, 3))
        @test_throws ArgumentError sig_kernel(randn(3, 2), randn(3, 2); dyadic_order=-1)
        @test_throws ArgumentError sig_gram(randn(3, 2, 2), randn(3, 3, 2))
    end

    @testset "Gram matrices equal pairwise kernels (threaded=$thr)" for thr in (true, false)
        X, Y = walk(9, 3, 5), walk(6, 3, 4)
        pw(λ) = [sig_kernel(X[:, :, a], Y[:, :, b]; dyadic_order=λ) for a in 1:5, b in 1:4]
        @test sig_gram(X, Y; dyadic_order=1, threaded=thr) ≈ pw(1)
        @test sig_gram(X; dyadic_order=1, threaded=thr) ≈ sig_gram(X, X; dyadic_order=1)
        @test issymmetric(sig_gram(X; threaded=thr))
        @test sig_gram(X, Y, 4; threaded=thr) ≈
              [sig_kernel(X[:, :, a], Y[:, :, b], 4) for a in 1:5, b in 1:4]
        @test sig_gram(X, 4; threaded=thr) ≈ sig_gram(X, X, 4)
    end

    @testset "PDE kernel gradient matches ForwardDiff (λ=$λ)" for λ in (0, 2)
        x, y = walk(8, 3), walk(5, 3)
        gx, gy = Zygote.gradient((a, b) -> sig_kernel(a, b; dyadic_order=λ), x, y)
        @test gx ≈ ForwardDiff.gradient(a -> sig_kernel(a, y; dyadic_order=λ), x) rtol=1e-10
        @test gy ≈ ForwardDiff.gradient(b -> sig_kernel(x, b; dyadic_order=λ), y) rtol=1e-10
    end

    @testset "Gram gradients match ForwardDiff (threaded=$thr)" for thr in (true, false)
        X, Y = walk(7, 2, 4), walk(5, 2, 3)
        W, Ws = randn(4, 3), randn(4, 4)
        f(A, B) = sum(sig_gram(A, B; dyadic_order=1, threaded=thr) .* W)
        gX, gY = Zygote.gradient(f, X, Y)
        @test gX ≈ ForwardDiff.gradient(A -> f(A, Y), X) rtol=1e-10
        @test gY ≈ ForwardDiff.gradient(B -> f(X, B), Y) rtol=1e-10

        fs(A) = sum(sig_gram(A; dyadic_order=1, threaded=thr) .* Ws)
        @test Zygote.gradient(fs, X)[1] ≈ ForwardDiff.gradient(fs, X) rtol=1e-10

        ft(A) = sum(sig_gram(A, Y, 4; threaded=thr) .* W)
        @test Zygote.gradient(ft, X)[1] ≈ ForwardDiff.gradient(ft, X) rtol=1e-10
    end

    @testset "RBF static kernel" begin
        x, y = walk(9, 2), walk(6, 2)
        κ = RBFKernel(0.7)
        k = sig_kernel(x, y; static_kernel=κ)
        @test sig_kernel(x .+ 3.0, y .+ 3.0; static_kernel=κ) ≈ k     # translation invariant
        # large σ: second differences of κ tend to ⟨Δx, Δy⟩ / σ²
        σ = 1e3
        k_rbf = sig_kernel(x, y; static_kernel=RBFKernel(σ), dyadic_order=1)
        k_lin = sig_kernel(x ./ σ, y ./ σ; dyadic_order=1)
        @test (k_rbf - 1) ≈ (k_lin - 1) rtol=1e-4
        @test_throws ArgumentError RBFKernel(0.0)
        @test sig_kernel(x, y; static_kernel=LinearKernel()) == sig_kernel(x, y)
    end

    @testset "RBF Gram equals pairwise, gradients match ForwardDiff (threaded=$thr)" for thr in (true, false)
        κ = RBFKernel(0.5)
        X, Y = walk(7, 2, 5), walk(6, 2, 3)
        @test sig_gram(X, Y; static_kernel=κ, dyadic_order=1, threaded=thr) ≈
              [sig_kernel(X[:, :, a], Y[:, :, b]; static_kernel=κ, dyadic_order=1)
               for a in 1:5, b in 1:3]
        @test sig_gram(X; static_kernel=κ, threaded=thr) ≈ sig_gram(X, X; static_kernel=κ)

        x, y = X[:, :, 1], Y[:, :, 1]
        gx, gy = Zygote.gradient((a, b) -> sig_kernel(a, b; static_kernel=κ, dyadic_order=1), x, y)
        @test gx ≈ ForwardDiff.gradient(a -> sig_kernel(a, y; static_kernel=κ, dyadic_order=1), x) rtol=1e-10
        @test gy ≈ ForwardDiff.gradient(b -> sig_kernel(x, b; static_kernel=κ, dyadic_order=1), y) rtol=1e-10

        W, Ws = randn(5, 3), randn(5, 5)
        f(A, B) = sum(sig_gram(A, B; static_kernel=κ, threaded=thr) .* W)
        gX, gY = Zygote.gradient(f, X, Y)
        @test gX ≈ ForwardDiff.gradient(A -> f(A, Y), X) rtol=1e-10
        @test gY ≈ ForwardDiff.gradient(B -> f(X, B), Y) rtol=1e-10
        fs(A) = sum(sig_gram(A; static_kernel=κ, dyadic_order=1, threaded=thr) .* Ws)
        @test Zygote.gradient(fs, X)[1] ≈ ForwardDiff.gradient(fs, X) rtol=1e-10
    end

    @testset "kernel ridge on a Gram matrix" begin
        X = walk(12, 2, 30)
        K = sig_gram(X; dyadic_order=1)
        @test isposdef(Symmetric(K + 1e-8I))
        @test isposdef(Symmetric(sig_gram(X; static_kernel=RBFKernel(0.5)) + 1e-8I))
    end
end
