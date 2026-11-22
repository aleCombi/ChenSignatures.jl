using Test
using ChenSignatures

@testset "ChenSignatures Tests" begin
    include("exp_log.jl")
    include("correctness.jl")
    include("test_float32.jl")
    include("test_add_scaled.jl")
    include("edge_cases.jl")
    include("test_batch.jl")
    include("test_rolling_sig.jl")
    include("test_enzyme.jl")
    include("chain_rules.jl")
    include("test_backward.jl")
    include("test_logsig.jl")
    include("test_words.jl")
    include("test_kernels.jl")
    include("test_augmentations.jl")
end
