import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using IGAros, LinearAlgebra, SparseArrays, Printf

include(joinpath(@__DIR__, "bending_beam.jl"))

# ============================================================================
#  Convergence (L2 displacement + energy norm)  TM, p=1,2,3,4
# ============================================================================

form  = TwinMortarFormulation()
strat = ElementBasedIntegration()

println("% ── Convergence: L2 displacement + energy norm (TM, elem-based) ──")
flush(stdout)

for p in [1, 2, 3, 4]
    exp_max = p <= 2 ? 4 : 3
    epss = p == 1 ? 1e5 : 1e6
    @printf "\n    %% TM p=%d (exp=0..%d, eps=%.0e)\n" p exp_max epss
    @printf "    %% exp | L2_abs      | L2_rate | En_abs      | En_rate\n"
    flush(stdout)
    l2_prev = NaN; en_prev = NaN
    for exp in 0:exp_max
        try
            d = solve_beam_diag(p, exp; epss=epss, formulation=form, strategy=strat)
            l2_rate = isnan(l2_prev) ? NaN : log(l2_prev / d.l2_abs) / log(2.0)
            en_rate = isnan(en_prev) ? NaN : log(en_prev / d.en_abs) / log(2.0)
            @printf "      %2d  | %.4e  | %+5.2f   | %.4e  | %+5.2f\n" exp d.l2_abs (isnan(l2_rate) ? 0.0 : l2_rate) d.en_abs (isnan(en_rate) ? 0.0 : en_rate)
            flush(stdout)
            l2_prev = d.l2_abs; en_prev = d.en_abs
        catch e
            @printf "      %2d  | ERR: %s\n" exp sprint(showerror, e)
            flush(stdout)
        end
    end
end

println("\n% Done.")
