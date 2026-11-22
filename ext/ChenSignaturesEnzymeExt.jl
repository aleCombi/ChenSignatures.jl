# Enzyme reverse mode through the same handwritten adjoints as the ChainRules
# rules (reversible signature backward pass, Lyndon solve, PDE kernel adjoint),
# instead of differentiating the forward code. Without these rules Enzyme would
# store the forward trajectory, and it cannot differentiate the threaded Gram code.
#
# The rules forward to `ChainRulesCore.rrule`. (`Enzyme.@import_rrule` would do
# the same, but it lives in Enzyme's own ChainRulesCore extension, which is not
# guaranteed to be loaded when this extension is.)
module ChenSignaturesEnzymeExt

using ChenSignatures
using ChenSignatures: sig, logsig, sig_kernel, sig_gram, BasisCache
using ChainRulesCore
using Enzyme
using Enzyme: EnzymeRules, Annotation, Const, Active, Duplicated
using Base: IEEEFloat

function _augmented(config, f, args; kwargs...)
    ow = EnzymeRules.overwritten(config)
    # the pullback reads the inputs later: copy any the caller may overwrite
    vals = ntuple(i -> (ow[i + 1] && args[i].val isa AbstractArray) ? copy(args[i].val) :
                       args[i].val, length(args))
    y, pb = ChainRulesCore.rrule(f, vals...; kwargs...)
    primal = EnzymeRules.needs_primal(config) ? y : nothing
    shadow = (y isa AbstractArray && EnzymeRules.needs_shadow(config)) ? zero(y) : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, (pb, shadow))
end

function _reverse(dret, tape, args)
    pb, shadow = tape
    ȳ = dret isa Active ? dret.val : shadow
    if ȳ !== nothing && any(a -> !(a isa Const), args)
        grads = pb(ȳ)
        for (i, a) in enumerate(args)
            a isa Const && continue
            g = ChainRulesCore.unthunk(grads[i + 1])
            g isa ChainRulesCore.AbstractZero || (a.dval .+= g)
        end
    end
    return ntuple(_ -> nothing, length(args))
end

const _Mat = AbstractMatrix{<:IEEEFloat}
const _Arr3 = AbstractArray{<:IEEEFloat,3}

for (F, argtypes) in [
        (:sig, (_Mat, Int)),
        (:sig, (_Arr3, Int)),
        (:logsig, (_Mat, BasisCache)),
        (:logsig, (_Arr3, BasisCache)),
        (:sig_kernel, (_Mat, _Mat)),
        (:sig_gram, (_Arr3, _Arr3)),
        (:sig_gram, (_Arr3,)),
    ]
    names = [Symbol(:arg, i) for i in eachindex(argtypes)]
    typed = [:($(names[i])::Annotation{<:$(argtypes[i])}) for i in eachindex(argtypes)]
    @eval begin
        function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfigWidth{1},
                                              ::Const{typeof($F)}, ::Type{RT},
                                              $(typed...); kwargs...) where {RT}
            return _augmented(config, $F, ($(names...),); kwargs...)
        end
        function EnzymeRules.reverse(config::EnzymeRules.RevConfigWidth{1},
                                     ::Const{typeof($F)}, dret, tape,
                                     $(typed...); kwargs...)
            return _reverse(dret, tape, ($(names...),))
        end
    end
end

end
