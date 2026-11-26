# GPU signature-kernel Gram (ChenSignaturesKernelAbstractionsExt) on CUDA:
# correctness against the CPU implementation, then forward and gradient timings,
# the gradient taken through Zygote, Enzyme and Mooncake.
#
#   julia --project=benchmark/gpu -t 8 benchmark/gpu/bench.jl
using CUDA, KernelAbstractions, GPUArraysCore, ChenSignatures, ChainRulesCore
using Zygote, Enzyme, Mooncake, Printf, Random

relerr(a, ref) = maximum(abs.(Array(a) .- ref)) / maximum(abs.(ref))
med(f, k) = (f(); ts = [(@elapsed f()) for _ in 1:k]; sort(ts)[cld(k, 2)] * 1e3)
gpu(f) = () -> CUDA.@sync f()
paths(N, d, B) = 0.1 .* cumsum(randn(N, d, B); dims = 1)

loss(X, Y, W) = sum(W .* sig_gram(X, Y))

function cpu_grads(X, Y, W; kw...)
    _, pb = ChainRulesCore.rrule(sig_gram, X, Y; kw...)
    g = pb(W)
    g[2], g[3]
end

grad_zygote(X, Y, W) = Zygote.gradient((x, y) -> loss(x, y, W), X, Y)
function grad_enzyme(X, Y, W)
    gX, gY = zero(X), zero(Y)
    # runtime activity: needed for the CuArray broadcast in `loss`, not for the rule
    Enzyme.autodiff(set_runtime_activity(Reverse), loss, Active, Duplicated(X, gX), Duplicated(Y, gY), Const(W))
    gX, gY
end
const MCACHE = Dict{Any,Any}()
function grad_mooncake(X, Y, W)
    f = xy -> loss(xy[1], xy[2], W)
    c = get!(() -> Mooncake.prepare_gradient_cache(f, (X, Y)), MCACHE, (size(X), size(Y), eltype(X)))
    _, (_, g) = Mooncake.value_and_gradient!!(c, f, (X, Y))
    g[1], g[2]
end
const AD = (("Zygote", grad_zygote), ("Enzyme", grad_enzyme), ("Mooncake", grad_mooncake))

println("GPU: ", CUDA.name(CUDA.device()), "   CPU threads: ", Threads.nthreads())

# ---------------------------------------------------------------------------
println("\n== correctness against the CPU (Float64)")
Random.seed!(0)
for (Nx, Ny, d, Bx, By, λ) in ((20, 20, 3, 4, 5, 0), (30, 17, 2, 3, 4, 0), (12, 25, 4, 5, 3, 1), (9, 9, 3, 2, 2, 2))
    X, Y, W = paths(Nx, d, Bx), paths(Ny, d, By), randn(Bx, By)
    Xc, Yc, Wc = CuArray(X), CuArray(Y), CuArray(W)
    eK = relerr(sig_gram(Xc, Yc; dyadic_order = λ), sig_gram(X, Y; dyadic_order = λ))
    gx, gy = cpu_grads(X, Y, W; dyadic_order = λ)
    _, pb = ChainRulesCore.rrule(sig_gram, Xc, Yc; dyadic_order = λ)
    _, X̄, Ȳ = pb(Wc)
    @printf("  Nx=%2d Ny=%2d d=%d %dx%d λ=%d   K %.1e   x̄ %.1e   ȳ %.1e\n", Nx, Ny, d, Bx, By, λ,
            eK, relerr(X̄, gx), relerr(Ȳ, gy))
end
let X = paths(15, 3, 4), W = randn(4, 4)
    Xc = CuArray(X)
    _, pbc = ChainRulesCore.rrule(sig_gram, X)
    _, pbg = ChainRulesCore.rrule(sig_gram, Xc)
    @printf("  sig_gram(X) symmetric           K %.1e   x̄ %.1e\n",
            relerr(sig_gram(Xc), sig_gram(X)), relerr(pbg(CuArray(W))[2], pbc(W)[2]))
end

# ---------------------------------------------------------------------------
println("\n== timings, loss sum(W .* sig_gram(X, Y)), gradient w.r.t. X and Y (ms)")
gpu_mib(f) = (CUDA.reclaim(); GC.gc(true); CUDA.@allocated(CUDA.@sync f()) / 2^20)
for (N, d, B) in ((64, 3, 64), (64, 3, 128), (128, 3, 64), (256, 3, 32))
    Random.seed!(1)
    X, Y, W = paths(N, d, B), paths(N, d, B), randn(B, B)
    gx, gy = cpu_grads(X, Y, W)
    @printf("--- N=%d d=%d, %d pairs\n", N, d, B^2)
    for E in (Float32, Float64)
        Xe, Ye, We = E.(X), E.(Y), E.(W)
        @printf("  %-7s CPU      forward %7.2f   gradient %7.2f\n", E,
                med(() -> sig_gram(Xe, Ye), 5), med(() -> cpu_grads(Xe, Ye, We), 3))
        Xc, Yc, Wc = CuArray(Xe), CuArray(Ye), CuArray(We)
        @printf("  %-7s GPU      forward %7.2f (%6.1f MiB)   rrule+pullback %.1f MiB\n", E,
                med(gpu(() -> sig_gram(Xc, Yc)), 10), gpu_mib(() -> sig_gram(Xc, Yc)),
                gpu_mib(() -> last(ChainRulesCore.rrule(sig_gram, Xc, Yc))(Wc)))
        for (name, g) in AD
            try
                a, b = g(Xc, Yc, Wc)
                @printf("  %-7s %-8s gradient %7.2f   err x̄ %.1e  ȳ %.1e\n", E, name,
                        med(gpu(() -> g(Xc, Yc, Wc)), 5), relerr(a, gx), relerr(b, gy))
            catch e
                println("  $E $name FAIL: ", first(split(sprint(showerror, e), '\n'))[1:min(end, 200)])
            end
        end
    end
end
