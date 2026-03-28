"""
C₀ for genuinely different discretization ratios (including non-integer like 3:2, 5:4).
Then test C₀ on the 3D curved patch test.
"""

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using IGAros
using Printf

include(joinpath(@__DIR__, "concentric_cylinders.jl"))

function run_c0_ratios()
    p = 2; exp = 3

    println("═"^78)
    println("  C₀ for varied discretization ratios (p=$p, exp=$exp)")
    println("  Cylinder: E=100, ν=0.3, r_i=1, r_c=1.5, r_o=2")
    println("═"^78)

    # n_ang_p1_base = inner (slave), n_ang_p2_base = outer (master)
    # After refinement: n_elem = base * 2^exp
    # Ratio = n_ang_p1_base / n_ang_p2_base
    ratios = [
        (2, 2, "1:1 (conforming)"),
        (3, 2, "3:2"),
        (4, 3, "4:3"),
        (5, 4, "5:4"),
        (5, 3, "5:3"),
        (5, 2, "5:2"),
        (7, 4, "7:4"),
        (7, 3, "7:3"),
        (4, 2, "2:1"),
        (6, 3, "2:1 (default)"),
        (9, 3, "3:1"),
        (8, 3, "8:3"),
        (12, 3, "4:1"),
    ]

    @printf("  %-20s  %6s  %6s  %8s  %12s\n",
            "ratio", "n_s", "n_m", "ratio_f", "C₀")
    @printf("  %s\n", "─"^60)

    for (n1, n2, label) in ratios
        ns = n1 * 2^exp
        nm = n2 * 2^exp
        c0, _ = solve_cylinder(p, exp; epss=1.0,
            formulation=DualPassFormulation(),
            strategy=SegmentBasedIntegration(),
            n_ang_p1_base=n1, n_ang_p2_base=n2)
        @printf("  %-20s  %6d  %6d  %8.3f  %12.6e\n",
                label, ns, nm, ns/nm, c0)
    end
end

run_c0_ratios()
