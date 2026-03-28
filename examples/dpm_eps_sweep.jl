"""
DPM vs TM ε-sweep: accuracy + conditioning on concentric cylinders.
Tests whether ε-sensitivity is caused by quadrature error (TM, element-based)
or by the Z-block perturbation itself (would affect DPM too).
"""

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using IGAros
using Printf

include(joinpath(@__DIR__, "concentric_cylinders.jl"))

function run_dpm_vs_tm_eps()
    degrees   = [2, 3, 4]
    exp_level = 3
    eps_range = 10 .^ (-2:0.5:7)

    println("═"^78)
    println("  DPM vs TM: ε-sweep at exp = $exp_level")
    println("  DPM = segment-based (exact integration)")
    println("  TM  = element-based (approximate integration)")
    println("═"^78)

    for p in degrees
        println("\n── p = $p ──────────────────────────────────────────────")
        @printf("  %10s  %14s  %14s  %14s  %14s\n",
                "ε", "TM L²_stress", "DPM L²_stress", "TM κ(A)", "DPM κ(A)")
        @printf("  %s\n", "─"^70)

        for eps in eps_range
            # TM (element-based)
            tm_err = try
                rel, _ = solve_cylinder(p, exp_level;
                    epss=Float64(eps),
                    formulation=TwinMortarFormulation(),
                    strategy=ElementBasedIntegration())
                rel
            catch; NaN; end

            # DPM (segment-based, exact integration)
            dpm_err = try
                rel, _ = solve_cylinder(p, exp_level;
                    epss=Float64(eps),
                    formulation=DualPassFormulation(),
                    strategy=SegmentBasedIntegration())
                rel
            catch; NaN; end

            # Conditioning
            tm_kappa = try
                κ, _ = cylinder_kappa(p, exp_level;
                    formulation=TwinMortarFormulation(),
                    strategy=ElementBasedIntegration(),
                    epss=Float64(eps))
                κ
            catch; NaN; end

            dpm_kappa = try
                κ, _ = cylinder_kappa(p, exp_level;
                    formulation=DualPassFormulation(),
                    strategy=SegmentBasedIntegration(),
                    epss=Float64(eps))
                κ
            catch; NaN; end

            @printf("  %10.1e  %14.4e  %14.4e  %14.3e  %14.3e\n",
                    eps, tm_err, dpm_err, tm_kappa, dpm_kappa)
        end
    end
end

run_dpm_vs_tm_eps()
