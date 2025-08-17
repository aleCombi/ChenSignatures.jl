using Test
using PathSignatures

function test_tensor_interface(x::AbstractTensor)
    # Must not throw:
    T = eltype(x)
    d = dim(x)
    m = level(x)
    y = similar(x)
    copy!(y, x)
    return true
end
