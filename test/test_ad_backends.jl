# Every differentiable entry point, through each reverse-mode backend, against
# ForwardDiff. Zygote uses the ChainRules rules directly; Enzyme and Mooncake reach
# the same adjoints through the package extensions.
using ChenSignatures
using DifferentiationInterface
import DifferentiationInterface as DI
using Enzyme: Enzyme
using Mooncake: Mooncake
using Zygote: Zygote
using ForwardDiff
using Random
using Test

@testset "AD backends" begin
    Random.seed!(5)
    walk(n, d, b...) = 0.3 .* cumsum(randn(n, d, b...); dims=1)
    x, y = walk(8, 3), walk(6, 3)
    X, Y = walk(8, 3, 5), walk(6, 3, 4)
    b = prepare(3, 4)
    κ = RBFKernel(0.5)

    cases = [
        ("sig", a -> sum(abs2, sig(a, 4)), x),
        ("sig batch", A -> sum(abs2, sig(A, 4)), X),
        ("sig batch unthreaded", A -> sum(abs2, sig(A, 4; threaded=false)), X),
        ("logsig lyndon", a -> sum(abs2, logsig(a, b)), x),
        ("logsig words", a -> sum(abs2, logsig(a, b; coords=:words)), x),
        ("logsig expanded", a -> sum(abs2, logsig(a, b; coords=:expanded)), x),
        ("logsig batch", A -> sum(abs2, logsig(A, b)), X),
        ("sig_kernel", a -> sig_kernel(a, y), x),
        ("sig_kernel λ=1 RBF", a -> sig_kernel(a, y; dyadic_order=1, static_kernel=κ), x),
        ("sig_kernel truncated", a -> sig_kernel(a, y, 4), x),
        ("sig_gram", A -> sum(abs2, sig_gram(A, Y; dyadic_order=1)), X),
        ("sig_gram RBF", A -> sum(abs2, sig_gram(A, Y; static_kernel=κ)), X),
        ("sig_gram symmetric", A -> sum(abs2, sig_gram(A)), X),
        ("sig_gram symmetric unthreaded", A -> sum(abs2, sig_gram(A; threaded=false)), X),
        ("sig_gram truncated", A -> sum(abs2, sig_gram(A, Y, 3)), X),
    ]

    backends = [
        ("Zygote", AutoZygote()),
        ("Enzyme", AutoEnzyme(; mode=Enzyme.Reverse, function_annotation=Enzyme.Const)),
        ("Mooncake", AutoMooncake(; config=nothing)),
    ]

    @testset "$bname: $name" for (bname, be) in backends, (name, f, a) in cases
        @test DI.gradient(f, be, a) ≈ ForwardDiff.gradient(f, a) rtol=1e-9 atol=1e-12
    end

    @testset "rules are used (Enzyme cannot differentiate the threaded Gram code itself)" begin
        @test Base.get_extension(ChenSignatures, :ChenSignaturesEnzymeExt) !== nothing
        @test Base.get_extension(ChenSignatures, :ChenSignaturesMooncakeExt) !== nothing
    end

    @testset "zero cotangent" begin
        _, back = Zygote.pullback(A -> sig_gram(A, Y), X)
        @test iszero(back(zeros(5, 4))[1])
    end
end
