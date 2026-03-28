import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using IGAros, LinearAlgebra, SparseArrays, Printf

include(joinpath(@__DIR__, "curved_patch_test.jl"))

# Fine ε-sweep (half-decade steps) for TM and DPM, p=2,3,4
eps_range = 10.0 .^ range(-2, 7, length=37)  # half-decade steps

for (label, form) in [("TM", TwinMortarFormulation()), ("DPM", DualPassFormulation())]
    println("\n% $label eps-sweep data")
    for p in [2, 3, 4]
        println("% $label p=$p")
        for eps in eps_range
            try
                U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
                    nen, nel, IEN, INC, E, nu, NQUAD, _ =
                    _cpt_solve(p, 1; epss=eps, formulation=form)
                rms_zz, _, _, _ = stress_error_cpt(U, ID, npc, nsd, npd, p_mat, n_mat_ref,
                    KV_ref, P_ref, B_ref, nen, nel, IEN, INC, E, nu, NQUAD)
                @printf "            (%.3e, %.4e)\n" eps rms_zz
            catch
                @printf "            %% (%.3e, ERR)\n" eps
            end
        end
    end
end
