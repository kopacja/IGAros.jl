"""
Test: does C₀ scale with interface curvature and/or loading?
"""

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using IGAros
using Printf

include(joinpath(@__DIR__, "concentric_cylinders.jl"))

function run_c0_curvature()
    p = 2; exp = 3

    println("═"^78)
    println("  C₀ dependence on curvature and loading")
    println("  p = $p, exp = $exp, DPM segment-based")
    println("═"^78)

    # ── Test 1: Different interface radii (curvature = 1/r_c) ────────────
    println("\n── Test 1: Interface radius r_c (curvature κ = 1/r_c) ──")
    @printf("  %8s  %10s  %12s  %12s\n", "r_c", "κ=1/r_c", "C₀", "C₀×r_c")
    @printf("  %s\n", "─"^48)
    for r_c in [1.1, 1.25, 1.5, 1.75, 1.9]
        c0, _ = solve_cylinder(p, exp; epss=1.0,
            formulation=DualPassFormulation(),
            strategy=SegmentBasedIntegration(),
            r_c=r_c)
        @printf("  %8.2f  %10.4f  %12.6e  %12.6e\n", r_c, 1/r_c, c0, c0*r_c)
    end

    # ── Test 2: Different loading levels ─────────────────────────────────
    println("\n── Test 2: External pressure p_o ──")
    @printf("  %8s  %12s  %12s\n", "p_o", "C₀", "C₀/p_o")
    @printf("  %s\n", "─"^36)
    for p_o in [0.1, 0.5, 1.0, 2.0, 10.0]
        c0, _ = solve_cylinder(p, exp; epss=1.0,
            formulation=DualPassFormulation(),
            strategy=SegmentBasedIntegration(),
            p_o=p_o)
        @printf("  %8.1f  %12.6e  %12.6e\n", p_o, c0, c0/p_o)
    end

    # ── Test 3: Different Young's modulus ─────────────────────────────────
    println("\n── Test 3: Young's modulus E ──")
    @printf("  %8s  %12s  %12s\n", "E", "C₀", "C₀×E")
    @printf("  %s\n", "─"^36)
    for E in [10.0, 50.0, 100.0, 500.0, 1000.0]
        c0, _ = solve_cylinder(p, exp; epss=1.0,
            formulation=DualPassFormulation(),
            strategy=SegmentBasedIntegration(),
            E=E)
        @printf("  %8.1f  %12.6e  %12.6e\n", E, c0, c0*E)
    end
end

run_c0_curvature()
