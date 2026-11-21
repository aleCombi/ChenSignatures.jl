# Compile the common entry points for common shapes when the package is
# precompiled, so the first call in a session does not pay ~0.1-0.2 s of JIT
# compilation per (dimension, level). Each (d, m) is a separate specialisation
# (the kernels are compiled for the exact sizes), so the list is kept modest.

using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    shapes = [(d, m) for d in 1:6 for m in 2:5]
    @compile_workload begin
        for (d, m) in shapes
            p = [sin(i + 2j) for i in 1:4, j in 1:d]
            P = [sin(i + 2j + k) for i in 1:4, j in 1:d, k in 1:2]
            b = prepare(d, m)
            y, back = ChainRulesCore.rrule(sig, p, m); back(y)
            Y, backB = ChainRulesCore.rrule(sig, P, m); backB(Y)
            # `coords` is a runtime value: one mode compiles the code for all three
            l, lback = ChainRulesCore.rrule(logsig, p, b); lback(l)
            L, lbackB = ChainRulesCore.rrule(logsig, P, b); lbackB(L)
        end
    end
end
