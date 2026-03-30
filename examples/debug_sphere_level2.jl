# debug_sphere_level2.jl — Non-conforming mortar + ε sweep for p=1

import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using IGAros
using LinearAlgebra, SparseArrays, Printf

include(joinpath(@__DIR__, "pressurized_sphere.jl"))
include(joinpath(@__DIR__, "pressurized_sphere_deltoidal.jl"))
include(joinpath(@__DIR__, "pressurized_sphere_3patch.jl"))

E = 1000.0; nu = 0.3; p_i = 1.0; r_i = 1.0; r_o = 1.2; n_base = 2

# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL 2: Non-conforming mortar convergence at several ε values
# ═══════════════════════════════════════════════════════════════════════════════

println("\n" * "="^70)
println("LEVEL 2: Non-conforming mortar convergence — p=1, ratio=2.0")
println("="^70)

for epss in [1e2, 1e4, 1e6]
    @printf("\n--- ε = %.0e ---\n", epss)
    @printf("  %4s  %12s  %6s  %12s  %6s  %12s  %6s\n",
            "exp", "l2_disp", "rate", "energy", "rate", "stress", "rate")
    prev_l2 = prev_en = prev_σ = 0.0
    for exp in 0:4
        try
            res = solve_sphere_3patch_mortar(1, exp;
                conforming = false, mesh_ratio = 2.0,
                E = E, nu = nu, p_i = p_i, epss = epss,
                r_i = r_i, r_o = r_o, n_base = n_base,
                NQUAD_mortar = 3)
            rate_l2 = exp > 0 ? log2(prev_l2 / res.l2_rel) : NaN
            rate_en = exp > 0 ? log2(prev_en / res.en_rel) : NaN
            rate_σ  = exp > 0 ? log2(prev_σ  / res.σ_rel)  : NaN
            @printf("  %4d  %12.4e  %6.2f  %12.4e  %6.2f  %12.4e  %6.2f\n",
                    exp, res.l2_rel, rate_l2, res.en_rel, rate_en, res.σ_rel, rate_σ)
            prev_l2 = res.l2_rel; prev_en = res.en_rel; prev_σ = res.σ_rel
        catch e
            @printf("  %4d  ERROR: %s\n", exp, sprint(showerror, e))
            println("  ", join(split(sprint(showerror, e, catch_backtrace()), "\n")[1:min(5, end)], "\n  "))
            break
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL 3: ε sweep at exp=2 (non-conforming)
# ═══════════════════════════════════════════════════════════════════════════════

println("\n" * "="^70)
println("LEVEL 3: ε sweep — p=1, exp=2, non-conforming ratio=2.0")
println("="^70)

@printf("  %12s  %12s  %12s  %12s\n", "ε", "l2_disp", "energy", "stress")
for epss in 10.0 .^ (-1:8)
    try
        res = solve_sphere_3patch_mortar(1, 2;
            conforming = false, mesh_ratio = 2.0,
            E = E, nu = nu, p_i = p_i, epss = epss,
            r_i = r_i, r_o = r_o, n_base = n_base, NQUAD_mortar = 3)
        @printf("  %12.0e  %12.4e  %12.4e  %12.4e\n",
                epss, res.l2_rel, res.en_rel, res.σ_rel)
    catch e
        @printf("  %12.0e  ERROR: %s\n", epss, sprint(showerror, e))
    end
end

println("\nDone.")
