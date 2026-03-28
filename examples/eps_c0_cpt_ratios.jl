"""
3D Curved Patch Test: C₀ vs non-conformity ratio and Young's modulus.
Investigate why integer ratios (2:1, 3:1) behave differently from non-integer (3:2, 4:3).
"""

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using IGAros
using Printf

include(joinpath(@__DIR__, "curved_patch_test.jl"))

function run_cpt_ratio_E()
    p = 2; exp = 1

    ratios = [
        (2, 2, "1:1 (conf)"),
        (3, 2, "3:2"),
        (4, 3, "4:3"),
        (5, 4, "5:4"),
        (4, 2, "2:1"),
        (5, 3, "5:3"),
        (5, 2, "5:2"),
        (6, 2, "3:1"),
        (7, 3, "7:3"),
        (8, 2, "4:1"),
        (7, 4, "7:4"),
        (6, 4, "3:2 (scaled)"),
        (9, 6, "3:2 (×3)"),
    ]

    E_values = [1e2, 1e3, 1e4, 1e5]

    println("═"^90)
    println("  3D CPT: C₀(ε=1) vs ratio and E  (p=$p, exp=$exp, arc_amp=0.3)")
    println("═"^90)

    # Header
    @printf("  %-16s  %6s  %6s", "ratio", "n_s", "n_m")
    for E in E_values
        @printf("  %12s", "E=$(Int(E))")
    end
    println()
    @printf("  %s\n", "─"^(30 + 14*length(E_values)))

    for (nxl, nxu, label) in ratios
        conf = (nxl == nxu)
        ns = nxl * 2^exp
        nm = nxu * 2^exp
        @printf("  %-16s  %6d  %6d", label, ns, nm)
        for E in E_values
            rms, _ = check_cpt(p, exp; epss=1.0, E=E,
                formulation=TwinMortarFormulation(),
                strategy=ElementBasedIntegration(),
                conforming=conf,
                n_x_lower_base=nxl, n_x_upper_base=nxu,
                n_y_lower_base=nxl, n_y_upper_base=nxu)
            @printf("  %12.4e", rms)
        end
        println()
    end

    # ── Also check: is the "small C₀" for integer ratios real or just discretization? ──
    println("\n\n── ε-sweep for 2:1 (integer) vs 3:2 (non-integer), E=1e3 ──")
    for (nxl, nxu, label) in [(4, 2, "2:1"), (3, 2, "3:2")]
        println("\n  $label:")
        @printf("  %10s  %14s  %14s\n", "ε", "RMS σ_zz", "RMS×ε")
        @printf("  %s\n", "─"^40)
        conf = (nxl == nxu)
        for k in 0:7
            eps = 10.0^k
            rms, _ = check_cpt(p, exp; epss=eps, E=1e3,
                formulation=TwinMortarFormulation(),
                strategy=ElementBasedIntegration(),
                conforming=conf,
                n_x_lower_base=nxl, n_x_upper_base=nxu,
                n_y_lower_base=nxl, n_y_upper_base=nxu)
            @printf("  %10.0e  %14.4e  %14.4e\n", eps, rms, rms*eps)
        end
    end
end

run_cpt_ratio_E()
