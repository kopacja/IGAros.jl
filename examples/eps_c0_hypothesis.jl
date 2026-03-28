"""
Test hypothesis: C₀ depends on curvature × non-conformity ratio.

Predictions:
  - Conforming curved interface (1:1): C₀ ≈ 0
  - Different non-conformity ratios on curved: C₀ changes
  - Flat non-conforming: C₀ ≈ 0 (already known from patch test)
"""

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using IGAros
using Printf

include(joinpath(@__DIR__, "concentric_cylinders.jl"))

function measure_c0(p, exp; kwargs...)
    # Measure error at ε=1 → this IS C₀ (since error ≈ C₀/ε)
    rel, _ = solve_cylinder(p, exp; epss=1.0,
        formulation=DualPassFormulation(),
        strategy=SegmentBasedIntegration(),
        kwargs...)
    return rel
end

function run_c0_hypothesis()
    p = 2; exp = 3  # representative case

    println("═"^78)
    println("  C₀ hypothesis test: curvature × non-conformity")
    println("  p = $p, exp = $exp")
    println("═"^78)

    # ── Test 1: Conforming vs non-conforming on curved interface ──────────
    println("\n── Test 1: Conforming vs non-conforming (curved interface) ──")
    c0_nc = measure_c0(p, exp; conforming=false)
    c0_c  = measure_c0(p, exp; conforming=true)
    @printf("  Non-conforming (2:1): C₀ = %.6e\n", c0_nc)
    @printf("  Conforming     (1:1): C₀ = %.6e\n", c0_c)
    @printf("  Ratio (NC/C):         %.1f\n", c0_nc / c0_c)

    # ── Test 2: Different non-conformity ratios ──────────────────────────
    println("\n── Test 2: Non-conformity ratio sweep (curved interface) ──")
    # n_ang_p1_base / n_ang_p2_base = ratio
    # Default: 6/3 = 2:1
    ratios = [(3, 3, "1:1"), (4, 3, "4:3"), (5, 3, "5:3"),
              (6, 3, "2:1"), (9, 3, "3:1"), (12, 3, "4:1")]
    @printf("  %8s  %8s  %8s  %12s\n", "ratio", "n_slave", "n_master", "C₀")
    @printf("  %s\n", "─"^44)
    for (n1, n2, label) in ratios
        c0 = try
            rel, _ = solve_cylinder(p, exp; epss=1.0,
                formulation=DualPassFormulation(),
                strategy=SegmentBasedIntegration(),
                n_ang_p1_base=n1, n_ang_p2_base=n2)
            rel
        catch ex
            @warn "  ratio $label FAILED: $ex"
            NaN
        end
        @printf("  %8s  %8d  %8d  %12.6e\n", label, n1*2^exp, n2*2^exp, c0)
    end

    # ── Test 3: Full ε-sweep for conforming curved ──────────────────────
    println("\n── Test 3: ε-sweep for CONFORMING curved interface (p=$p, exp=$exp) ──")
    @printf("  %10s  %14s  %14s\n", "ε", "error", "error×ε")
    @printf("  %s\n", "─"^40)
    for k in -1:7
        eps = 10.0^k
        rel, _ = solve_cylinder(p, exp; epss=Float64(eps),
            formulation=DualPassFormulation(),
            strategy=SegmentBasedIntegration(),
            conforming=true)
        @printf("  %10.0e  %14.4e  %14.4e\n", eps, rel, rel*eps)
    end

    # ── Test 4: Full ε-sweep for 3:1 ratio ───────────────────────────────
    println("\n── Test 4: ε-sweep for 3:1 non-conforming (p=$p, exp=$exp) ──")
    @printf("  %10s  %14s  %14s\n", "ε", "error", "error×ε")
    @printf("  %s\n", "─"^40)
    for k in -1:7
        eps = 10.0^k
        rel, _ = solve_cylinder(p, exp; epss=Float64(eps),
            formulation=DualPassFormulation(),
            strategy=SegmentBasedIntegration(),
            n_ang_p1_base=9, n_ang_p2_base=3)
        @printf("  %10.0e  %14.4e  %14.4e\n", eps, rel, rel*eps)
    end
end

run_c0_hypothesis()
