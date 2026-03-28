import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using IGAros, LinearAlgebra, SparseArrays, Printf

include(joinpath(@__DIR__, "curved_patch_test.jl"))

# Conditioning vs ε for TM (elem-based) and DPM (seg-based), p=2,3,4, exp=1
eps_range = 10.0 .^ range(-2, 7, length=19)

for (label, form, strat) in [
    ("TM",  TwinMortarFormulation(), ElementBasedIntegration()),
    ("DPM", DualPassFormulation(),   SegmentBasedIntegration()),
]
    println("\n% $label conditioning (pgfplots coordinates)")
    flush(stdout)
    for p in [2, 3, 4]
        println("    % $label p=$p")
        flush(stdout)
        for eps in eps_range
            try
                d = _cpt_solve_diag(p, 1; epss=eps, formulation=form, strategy=strat)
                A = Matrix([d.K d.C; d.C' -d.Z])
                kappa = cond(A)
                @printf "            (%.3e, %.4e)\n" eps kappa
                flush(stdout)
            catch e
                @printf "            %% (%.3e, ERR: %s)\n" eps sprint(showerror, e)
                flush(stdout)
            end
        end
    end
end
println()
