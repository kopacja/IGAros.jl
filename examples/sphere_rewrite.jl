import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using IGAros, LinearAlgebra, SparseArrays, Printf

include(joinpath(@__DIR__, "pressurized_sphere.jl"))

# ============================================================================
#  Convergence (L2 displacement + energy norm)  TM, p=1,2,3,4
# ============================================================================

form  = TwinMortarFormulation()
strat = ElementBasedIntegration()

println("% ── Convergence: L2 displacement + energy norm (TM, elem-based) ──")
flush(stdout)

for p in [2, 3, 4]
    exp_max = 3
    epss = 1e3
    @printf "\n    %% TM p=%d (exp=0..%d, eps=%.0e)\n" p exp_max epss
    @printf "    %% exp | h_rel       | L2_abs      | L2_rate | En_abs      | En_rate\n"
    flush(stdout)
    l2_prev = NaN; en_prev = NaN
    for exp in 0:exp_max
        try
            d = solve_sphere_diag(p, exp; epss=epss, formulation=form, strategy=strat)
            h_rel = 1.0 / 2.0^exp
            l2_rate = isnan(l2_prev) ? NaN : log(l2_prev / d.l2_abs) / log(2.0)
            en_rate = isnan(en_prev) ? NaN : log(en_prev / d.en_abs) / log(2.0)
            @printf "      %2d  | %.4e  | %.4e  | %+5.2f   | %.4e  | %+5.2f\n" exp h_rel d.l2_abs (isnan(l2_rate) ? 0.0 : l2_rate) d.en_abs (isnan(en_rate) ? 0.0 : en_rate)
            flush(stdout)
            l2_prev = d.l2_abs; en_prev = d.en_abs
        catch e
            @printf "      %2d  | ERR: %s\n" exp sprint(showerror, e)
            flush(stdout)
        end
    end
end

println("\n% ── ε-sensitivity sweep (TM, elem-based, exp=2) ──")
flush(stdout)

for p in [2, 3, 4]
    @printf "\n    %% TM p=%d, ε sweep at exp=2\n" p
    @printf "    %% eps        | L2_abs      | En_abs      | σ_abs\n"
    flush(stdout)
    for log_eps in [0, 1, 2, 3, 4, 5, 6, 7]
        eps_val = 10.0^log_eps
        try
            d = solve_sphere_diag(p, 2; epss=eps_val, formulation=form, strategy=strat)
            @printf "      1e%d  | %.4e  | %.4e  | %.4e\n" log_eps d.l2_abs d.en_abs d.σ_abs
            flush(stdout)
        catch e
            @printf "      1e%d  | ERR: %s\n" log_eps sprint(showerror, e)
            flush(stdout)
        end
    end
end

println("\n% Done.")
