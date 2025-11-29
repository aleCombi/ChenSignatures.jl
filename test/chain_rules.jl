using ChenSignatures
using Zygote
using Enzyme
using LinearAlgebra
using ChainRulesCore

println("Testing rrule for sig...")

# Create a simple path
path = [0.0 0.0; 1.0 1.0; 2.0 3.0]

# Test 0: Forward pass
println("\n=== Test 0: Forward pass ===")
flush(stdout)
result = sig(path, 3)
println("Forward result (first 5 elements): ", result[1:5])
flush(stdout)

# Test 1: Direct Enzyme
println("\n=== Test 1: Direct Enzyme (baseline) ===")
flush(stdout)
grad_direct = zeros(size(path))
loss_direct(p) = sum(sig(p, 3))
autodiff(Reverse, loss_direct, Active, Duplicated(path, grad_direct))
println("Direct Enzyme gradient:")
println(grad_direct)
flush(stdout)

# Test 2: Test the rrule directly
println("\n=== Test 2: Direct rrule call ===")
flush(stdout)
result_rrule, pullback = ChainRulesCore.rrule(sig, path, 3)
println("rrule forward pass OK")
flush(stdout)

ȳ = ones(length(result_rrule))
∂self, ∂path, ∂m = pullback(ȳ)
println("Pullback OK")
println("rrule gradient:")
println(∂path)
println("Matches Enzyme: ", isapprox(∂path, grad_direct))
flush(stdout)

println("\n=== Test 3: Zygote ===")
flush(stdout)
println("Calling Zygote.gradient...")
flush(stdout)
grad_zygote = Zygote.gradient(p -> sum(sig(p, 3)), path)[1]
println("SUCCESS! Zygote gradient:")
println(grad_zygote)
println("Matches Enzyme: ", isapprox(grad_zygote, grad_direct))