module PathSignatures

include("tensors.jl")
include("dense_tensors.jl")
include("sparse_tensors.jl")
include("sparse_dense_conversions.jl")
include("signatures.jl")
export Tensor, signature_path, SparseTensor   

end # module PathSignatures
