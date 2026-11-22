# Mooncake reverse mode through the ChainRules rules, so it gets the same
# handwritten adjoints as Zygote. The truncated kernels need no entry: they are
# built from `sig`.
module ChenSignaturesMooncakeExt

using ChenSignatures
using ChenSignatures: sig, logsig, sig_kernel, sig_gram, BasisCache
using Mooncake: @from_rrule, DefaultCtx
using Base: IEEEFloat

@from_rrule DefaultCtx Tuple{typeof(sig), Array{P,2}, Int} where {P<:IEEEFloat}
@from_rrule DefaultCtx Tuple{typeof(sig), Array{P,3}, Int} where {P<:IEEEFloat} true
@from_rrule DefaultCtx Tuple{typeof(logsig), Array{P,2}, BasisCache} where {P<:IEEEFloat} true
@from_rrule DefaultCtx Tuple{typeof(logsig), Array{P,3}, BasisCache} where {P<:IEEEFloat} true
@from_rrule DefaultCtx Tuple{typeof(sig_kernel), Array{P,2}, Array{P,2}} where {P<:IEEEFloat} true
@from_rrule DefaultCtx Tuple{typeof(sig_gram), Array{P,3}, Array{P,3}} where {P<:IEEEFloat} true
@from_rrule DefaultCtx Tuple{typeof(sig_gram), Array{P,3}} where {P<:IEEEFloat} true

end
