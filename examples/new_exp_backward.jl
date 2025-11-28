using ChenSignatures
using Enzyme
using LinearAlgebra

function test_exp_vec_verified()
    D, M = 2, 3
    
    function loss(x_vec)
        out = ChenSignatures.Tensor{Float64,D,M}()
        ChenSignatures.non_generated_exp_vec!(out, x_vec)
        return sum(out.coeffs)
    end
    
    x = [1.0, 2.0]
    grad = zeros(2)
    
    autodiff(Reverse, loss, Active, Duplicated(x, grad))
    println("Enzyme gradient: ", grad)
    
    # Finite differences
    eps = 1e-6
    fd = [(loss([x[1]+eps, x[2]]) - loss([x[1]-eps, x[2]]))/(2eps),
          (loss([x[1], x[2]+eps]) - loss([x[1], x[2]-eps]))/(2eps)]
    println("Finite diff:     ", fd)
    println("Error:           ", norm(grad - fd))
    println("Relative error:  ", norm(grad - fd) / norm(fd))
    
    if isapprox(grad, fd, rtol=1e-4)
        println("\n✓ TEST PASSED - Gradient is correct!")
    else
        println("\n✗ TEST FAILED - Gradient mismatch!")
    end
end

test_exp_vec_verified()