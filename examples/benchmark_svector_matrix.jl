using Revise
using BenchmarkTools, StaticArrays, LinearAlgebra
using PathSignatures

# Generate test paths
function generate_path_svector(D::Int, N::Int, T::Type=Float64)
    [SVector{D,T}(randn(T, D)) for _ in 1:N]
end

function generate_path_matrix(D::Int, N::Int, T::Type=Float64)
    randn(T, N, D)
end

# Benchmark function
function benchmark_signatures(; D=3, N=100, m=5)
    println("="^60)
    println("Benchmarking: D=$D, N=$N steps, truncation level m=$m")
    println("="^60)
    
    # Generate paths
    path_svec = generate_path_svector(D, N, Float64)
    path_mat = generate_path_matrix(D, N, Float64)
    
    # Ensure they represent the same path
    for i in 1:N
        path_mat[i, :] .= path_svec[i]
    end
    
    # Pre-allocate outputs
    out_svec = Tensor{Float64}(D, m)
    out_mat = Tensor{Float64}(D, m)
    out_view = Tensor{Float64}(D, m)
    
    # Warmup & correctness check
    sig_svec = signature_path!(copy(out_svec), path_svec)
    sig_mat = signature_path!(copy(out_mat), path_mat)
    sig_view = PathSignatures.signature_path_view!(copy(out_view), path_mat)
    
    # Verify all produce same result
    @assert isapprox(sig_svec, sig_mat, atol=1e-10) "Results don't match!"
    @assert isapprox(sig_svec, sig_view, atol=1e-10) "View results don't match!"
    
    println("\n1. Vector{SVector} approach:")
    t1 = @benchmark signature_path!($out_svec, $path_svec)
    display(t1)
    
    println("\n2. Matrix approach (with displacement vector):")
    t2 = @benchmark signature_path!($out_mat, $path_mat)
    display(t2)
    
    println("\n3. Matrix approach (with views):")
    t3 = @benchmark PathSignatures.signature_path_view!($out_view, $path_mat)
    display(t3)
    
    # Summary
    println("\n" * "="^60)
    println("Summary:")
    med1 = median(t1).time / 1e6  # to ms
    med2 = median(t2).time / 1e6
    med3 = median(t3).time / 1e6
    
    println("Vector{SVector}: $(round(med1, digits=3)) ms")
    println("Matrix (copy):   $(round(med2, digits=3)) ms ($(round(med2/med1, digits=2))x)")
    println("Matrix (view):   $(round(med3, digits=3)) ms ($(round(med3/med1, digits=2))x)")
end

# Run benchmarks with different parameters
benchmark_signatures(D=2, N=100, m=5)
benchmark_signatures(D=3, N=1000, m=6)
benchmark_signatures(D=5, N=500, m=4)