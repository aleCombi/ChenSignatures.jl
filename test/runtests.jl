using Test
using PathSignatures

@testset "PathSignatures Tests" begin
    include("exp_log.jl")
    include("correctness.jl")
end