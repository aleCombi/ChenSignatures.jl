using Revise
using BenchmarkTools, StaticArrays, LinearAlgebra
using PathSignatures
# === Benchmark Function ===

function benchmark_batch_signatures(; D=3, n_paths=100, n_steps=100, m=5)
    println("="^70)
    println("Batch Benchmark: D=$D, n_paths=$n_paths, n_steps=$n_steps, m=$m")
    println("="^70)
    
    # Generate test data
    svec_paths = [
        [SVector{D,Float64}(randn(D)) for _ in 1:(n_steps+1)]
        for _ in 1:n_paths
    ]
    
    array_data = Array{Float64,3}(undef, n_steps+1, D, n_paths)
    for p in 1:n_paths, t in 1:(n_steps+1), d in 1:D
        array_data[t, d, p] = svec_paths[p][t][d]
    end
    
    svec_ensemble = PathSignatures.SVectorEnsemble{D,Float64}(svec_paths, n_paths, n_steps)
    array_ensemble = PathSignatures.ArrayEnsemble{Float64}(array_data, n_paths, n_steps, D)
    
    # Pre-allocate outputs
    outs_svec = [Tensor{Float64}(D, m) for _ in 1:n_paths]
    outs_array1 = [Tensor{Float64}(D, m) for _ in 1:n_paths]
    outs_array2 = [Tensor{Float64}(D, m) for _ in 1:n_paths]
    
    # Warmup and correctness check
    PathSignatures.batch_signatures!(outs_svec, svec_ensemble)
    PathSignatures.batch_signatures!(outs_array1, array_ensemble)
    PathSignatures.batch_signatures!(outs_array2, array_ensemble)
    
    @assert all(isapprox(outs_svec[i], outs_array1[i], atol=1e-10) for i in 1:n_paths)
    @assert all(isapprox(outs_svec[i], outs_array2[i], atol=1e-10) for i in 1:n_paths)
    
    println("\n1. Vector{Vector{SVector}} approach:")
    t1 = @benchmark PathSignatures.batch_signatures!($outs_svec, $svec_ensemble)
    display(t1)
    
    println("\n2. 3D Array (path-major access):")
    t2 = @benchmark PathSignatures.batch_signatures!($outs_array1, $array_ensemble)
    display(t2)
    
    println("\n3. 3D Array (time-major access):")
    t3 = @benchmark PathSignatures.batch_signatures!($outs_array2, $array_ensemble)
    display(t3)
    
    # Summary
    println("\n" * "="^70)
    println("Summary:")
    med1 = median(t1).time / 1e6
    med2 = median(t2).time / 1e6
    med3 = median(t3).time / 1e6
    
    println("Vector{Vector{SVector}}: $(round(med1, digits=3)) ms")
    println("3D Array (path-major):   $(round(med2, digits=3)) ms ($(round(med2/med1, digits=2))x)")
    println("3D Array (time-major):   $(round(med3, digits=3)) ms ($(round(med3/med1, digits=2))x)")
    
    # Memory access analysis
    println("\nMemory access patterns:")
    println("  SVec: $(n_paths) separate allocations, each path contiguous")
    println("  Array: 1 allocation, $(n_paths * (n_steps+1) * D) total elements")
    cache_line_size = 64  # bytes
    bytes_per_elem = 8  # Float64
    elems_per_line = cache_line_size ÷ bytes_per_elem
    println("  Cache lines per timestep: $(ceil(Int, D*n_paths/elems_per_line)) (time-major)")
end

# Run benchmarks
benchmark_batch_signatures(D=3, n_paths=10, n_steps=100, m=5)
benchmark_batch_signatures(D=3, n_paths=100, n_steps=100, m=5)
benchmark_batch_signatures(D=3, n_paths=1000, n_steps=100, m=5)
benchmark_batch_signatures(D=5, n_paths=100, n_steps=500, m=4)