using ChenSignatures
using Test

@testset "word labels" begin
    @test sig_words(2, 2) == [[1], [2], [1, 1], [1, 2], [2, 1], [2, 2]]
    @test sig_words(1, 3) == [[1], [1, 1], [1, 1, 1]]

    @testset "sig_index inverts sig_words (d=$d, m=$m)" for (d, m) in [(1, 4), (2, 5), (3, 4), (5, 3)]
        ws = sig_words(d, m)
        @test length(ws) == sum(d^k for k in 1:m)
        @test all(sig_index(w, d) == i for (i, w) in enumerate(ws))
    end

    # A single straight segment from 0 to z has signature z^{⊗k}/k! at level k,
    # so the coordinate of word w is prod(z[w]) / length(w)!. Distinct primes
    # make every word's value distinct, so this pins the layout.
    @testset "labels match sig's layout (d=$d)" for d in 1:4
        z = [2.0, 3.0, 5.0, 7.0][1:d]
        s = sig(vcat(zeros(1, d), z'), 4)
        for (i, w) in enumerate(sig_words(d, 4))
            @test s[i] ≈ prod(z[w]) / factorial(length(w))
        end
    end

    @testset "logsig_words" begin
        b = prepare(3, 4)
        lw = logsig_words(b)
        @test length(lw) == length(logsig(randn(10, 3), b))
        @test lw[1:3] == [[1], [2], [3]]
        @test lw[4:6] == [[1, 2], [1, 3], [2, 3]]
        # :words coordinates are the expanded log at these words
        p = randn(10, 3)
        @test logsig(p, b; coords=:words) == logsig(p, b; coords=:expanded)[[sig_index(w, 3) for w in lw]]
    end

    @test_throws ArgumentError sig_index([3], 2)
    @test_throws ArgumentError sig_index(Int[], 2)
end
