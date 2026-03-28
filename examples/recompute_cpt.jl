import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using IGAros, LinearAlgebra, SparseArrays, Printf

include(joinpath(@__DIR__, "curved_patch_test.jl"))

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Table: Method comparison at p=2 across mesh sizes
#    TM  = TwinMortarFormulation + ElementBasedIntegration
#    DPM = DualPassFormulation   + SegmentBasedIntegration
# ═══════════════════════════════════════════════════════════════════════════════
println("\n  TABLE: Curved patch test p=2, non-conforming 3:2")
println("  " * "="^80)

methods = [
    ("SPMS",  SinglePassFormulation(),  SegmentBasedIntegration(),  0.0),
    ("SPME",  SinglePassFormulation(),  ElementBasedIntegration(),  0.0),
    ("DPM",   DualPassFormulation(),    SegmentBasedIntegration(),  1e2),
    ("TM",    TwinMortarFormulation(),  ElementBasedIntegration(),  1e2),
]

@printf "  %10s |" "Method"
for ex in 0:3
    h_label = ex == 0 ? "h=1" : ex == 1 ? "h=1/2" : ex == 2 ? "h=1/4" : "h=1/8"
    @printf " %14s |" h_label
end
println()
@printf "  %10s-|" "----------"
for _ in 0:3; @printf " %14s-|" "--------------"; end
println()

for (label, form, strat, eps) in methods
    @printf "  %10s |" label
    for ex in 0:3
        try
            U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
                nen, nel, IEN, INC, E, nu, NQUAD, _ =
                _cpt_solve(2, ex; epss=eps, formulation=form, strategy=strat)
            rms_zz, _, _, _ = stress_error_cpt(U, ID, npc, nsd, npd, p_mat, n_mat_ref,
                KV_ref, P_ref, B_ref, nen, nel, IEN, INC, E, nu, NQUAD)
            @printf " %14.1e |" rms_zz
        catch e
            @printf " %14s |" "ERR"
            @warn "$label exp=$ex: $(sprint(showerror, e))"
        end
    end
    println()
end
println()

# ═══════════════════════════════════════════════════════════════════════════════
# 2. eps-sensitivity: TM (elem-based) and DPM (seg-based) at p=2,3,4, exp=1
# ═══════════════════════════════════════════════════════════════════════════════
eps_range = 10.0 .^ range(-2, 7, length=19)

for (label, form, strat) in [
    ("TM",  TwinMortarFormulation(), ElementBasedIntegration()),
    ("DPM", DualPassFormulation(),   SegmentBasedIntegration()),
]
    println("\n% $label eps-sweep (pgfplots coordinates)")
    for p in [2, 3, 4]
        println("    % $label p=$p")
        for eps in eps_range
            try
                U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
                    nen, nel, IEN, INC, E, nu, NQUAD, _ =
                    _cpt_solve(p, 1; epss=eps, formulation=form, strategy=strat)
                rms_zz, _, _, _ = stress_error_cpt(U, ID, npc, nsd, npd, p_mat, n_mat_ref,
                    KV_ref, P_ref, B_ref, nen, nel, IEN, INC, E, nu, NQUAD)
                @printf "            (%.3e, %.4e)\n" eps rms_zz
            catch e
                @printf "            %% (%.3e, ERR: %s)\n" eps sprint(showerror, e)
            end
        end
    end
end
println()
