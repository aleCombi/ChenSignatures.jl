# Mooncake reverse mode for sig_gram on GPU arrays, through the rrule of
# ChenSignaturesKernelAbstractionsExt (the handwritten adjoint kernel).
module ChenSignaturesGPUMooncakeExt

using ChenSignatures: sig_gram
using GPUArraysCore: AbstractGPUArray
using KernelAbstractions
using Mooncake: @from_rrule, DefaultCtx
using Base: IEEEFloat

@from_rrule DefaultCtx Tuple{typeof(sig_gram), AbstractGPUArray{P,3}, AbstractGPUArray{P,3}} where {P<:IEEEFloat} true
@from_rrule DefaultCtx Tuple{typeof(sig_gram), AbstractGPUArray{P,3}} where {P<:IEEEFloat} true

end
