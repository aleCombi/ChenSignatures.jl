# Our side of the rival comparison, on the data bench_rivals.py wrote: CPU and GPU
# sig_gram, forward and gradient (loss sum(W .* K), gradient w.r.t. X, like the
# Python side), and every library's accuracy against our CPU Float64 result.
#
#   cd benchmark/gpu/rivals && .venv/bin/python bench_rivals.py
#   julia --project=.. -t 8 compare_rivals.jl
using CUDA, KernelAbstractions, GPUArraysCore, ChenSignatures, ChainRulesCore, Printf

readbin(path, dims) = (a = Array{Float64}(undef, dims); read!(path, a); a)
# Python C-order (B, N, d) -> Julia (N, d, B); (B, B) -> transpose
paths(tag, s, N, d, B) = permutedims(readbin(joinpath(@__DIR__, "data", "$(tag)_$(s).bin"), (d, N, B)), (2, 1, 3))
relerr(a, ref) = maximum(abs.(Array(a) .- ref)) / maximum(abs.(ref))
med(f, k) = (f(); ts = [(@elapsed f()) for _ in 1:k]; sort(ts)[cld(k, 2)] * 1e3)
gpu(f) = () -> CUDA.@sync f()
gpu_mib(f) = (CUDA.reclaim(); GC.gc(true); CUDA.@allocated(CUDA.@sync f()) / 2^20)
grad_x(X, Y, W) = last(ChainRulesCore.rrule(sig_gram, X, Y))(W)[2]

println("threads=$(Threads.nthreads()) GPU=$(CUDA.name(CUDA.device()))")
for (N, d, B) in ((64, 3, 64), (64, 3, 128), (128, 3, 64), (256, 3, 32))
    tag = "N$(N)_d$(d)_B$(B)"
    X, Y = paths(tag, "X", N, d, B), paths(tag, "Y", N, d, B)
    W = permutedims(readbin(joinpath(@__DIR__, "data", "$(tag)_W.bin"), (B, B)))
    Kref, Gref = sig_gram(X, Y), grad_x(X, Y, W)
    @printf("--- N=%d d=%d B=%d (%d pairs)\n", N, d, B, B^2)
    for E in (Float32, Float64)
        Xe, Ye, We = E.(X), E.(Y), E.(W)
        @printf("  Chen CPU  %-7s forward %8.2f ms   gradient %8.2f ms   err K %.1e  grad %.1e\n", E,
                med(() -> sig_gram(Xe, Ye), 5), med(() -> grad_x(Xe, Ye, We), 3),
                relerr(sig_gram(Xe, Ye), Kref), relerr(grad_x(Xe, Ye, We), Gref))
    end
    for E in (Float32, Float64)
        Xc, Yc, Wc = CuArray(E.(X)), CuArray(E.(Y)), CuArray(E.(W))
        @printf("  Chen GPU  %-7s forward %8.2f ms (%7.1f MiB)   gradient %8.2f ms (%7.1f MiB)   err K %.1e  grad %.1e\n",
                E, med(gpu(() -> sig_gram(Xc, Yc)), 10), gpu_mib(() -> sig_gram(Xc, Yc)),
                med(gpu(() -> grad_x(Xc, Yc, Wc)), 5), gpu_mib(() -> grad_x(Xc, Yc, Wc)),
                relerr(sig_gram(Xc, Yc), Kref), relerr(grad_x(Xc, Yc, Wc), Gref))
    end
    for lib in ("pySigLib", "sigkernel"), dt in ("float32", "float64")
        fK = joinpath(@__DIR__, "out", "$(tag)_$(lib)_$(dt)_K.bin")
        isfile(fK) || continue
        K = permutedims(readbin(fK, (B, B)))
        G = permutedims(readbin(joinpath(@__DIR__, "out", "$(tag)_$(lib)_$(dt)_G.bin"), (d, N, B)), (2, 1, 3))
        @printf("  %-9s %-7s accuracy vs Chen CPU Float64:   err K %.1e  grad %.1e\n", lib, dt,
                relerr(K, Kref), relerr(G, Gref))
    end
end
