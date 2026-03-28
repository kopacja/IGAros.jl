"""
ε-scaling analysis on 3D curved patch test.
Does C₀/ε scaling hold in 3D? Is C₀ independent of non-conformity ratio?
"""

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using IGAros
using Printf

include(joinpath(@__DIR__, "curved_patch_test.jl"))

function run_c0_cpt()
    p = 2; exp = 1   # exp=1 is sufficient, 3D is expensive

    println("═"^78)
    println("  3D Curved Patch Test: ε-scaling analysis")
    println("  p = $p, exp = $exp, E = 1e5, arc_amp = 0.3")
    println("═"^78)

    # ── Part 1: Full ε-sweep (non-conforming, default 3:2 ratio) ──────────
    println("\n── Part 1: ε-sweep (non-conforming, default ratio) ──")
    @printf("  %10s  %14s  %14s  %14s\n", "ε", "RMS σ_zz", "RMS×ε", "max σ_zz")
    @printf("  %s\n", "─"^56)
    for k in 0:7
        eps = 10.0^k
        rms, maxe = check_cpt(p, exp; epss=eps,
            formulation=TwinMortarFormulation(),
            strategy=ElementBasedIntegration(),
            conforming=false)
        @printf("  %10.0e  %14.4e  %14.4e  %14.4e\n", eps, rms, rms*eps, maxe)
    end

    # ── Part 2: C₀ for different non-conformity ratios ─────────────────────
    println("\n── Part 2: C₀ = RMS(ε=1) for different non-conformity ratios ──")
    ratios = [
        (2, 2, 2, 2, "1:1 (conforming)"),
        (3, 2, 3, 2, "3:2 (default NC)"),
        (4, 2, 4, 2, "2:1"),
        (5, 2, 5, 2, "5:2"),
        (6, 2, 6, 2, "3:1"),
        (4, 3, 4, 3, "4:3"),
        (5, 3, 5, 3, "5:3"),
    ]
    @printf("  %-20s  %6s  %6s  %14s\n", "ratio", "n_x_l", "n_x_u", "C₀ (RMS)")
    @printf("  %s\n", "─"^50)

    for (nxl, nxu, nyl, nyu, label) in ratios
        conf = (nxl == nxu && nyl == nyu)
        rms, _ = check_cpt(p, exp; epss=1.0,
            formulation=TwinMortarFormulation(),
            strategy=ElementBasedIntegration(),
            conforming=conf,
            n_x_lower_base=nxl, n_x_upper_base=nxu,
            n_y_lower_base=nyl, n_y_upper_base=nyu)
        ns = nxl * 2^exp
        nm = nxu * 2^exp
        @printf("  %-20s  %6d  %6d  %14.6e\n", label, ns, nm, rms)
    end

    # ── Part 3: C₀ scaling with E ──────────────────────────────────────────
    println("\n── Part 3: C₀ vs Young's modulus E ──")
    @printf("  %10s  %14s  %14s\n", "E", "C₀ (RMS)", "C₀×E")
    @printf("  %s\n", "─"^40)
    for E in [1e3, 1e4, 1e5, 1e6]
        rms, _ = check_cpt(p, exp; epss=1.0, E=E,
            formulation=TwinMortarFormulation(),
            strategy=ElementBasedIntegration(),
            conforming=false)
        @printf("  %10.0e  %14.6e  %14.6e\n", E, rms, rms*E)
    end

    # ── Part 4: C₀ scaling with curvature (arc_amp) ────────────────────────
    println("\n── Part 4: C₀ vs interface curvature (arc_amp) ──")
    @printf("  %10s  %14s\n", "arc_amp", "C₀ (RMS)")
    @printf("  %s\n", "─"^28)
    for amp in [0.0, 0.1, 0.2, 0.3, 0.5]
        rms, _ = check_cpt(p, exp; epss=1.0, arc_amp=amp, arc_amp_y=amp,
            formulation=TwinMortarFormulation(),
            strategy=ElementBasedIntegration(),
            conforming=false)
        @printf("  %10.2f  %14.6e\n", amp, rms)
    end

    # ── Part 5: h-scaling of C₀ ────────────────────────────────────────────
    println("\n── Part 5: C₀ vs mesh refinement (h-scaling) ──")
    @printf("  %3s  %8s  %14s  %8s\n", "exp", "h", "C₀ (RMS)", "rate")
    @printf("  %s\n", "─"^38)
    prev_c0 = NaN
    for e in 0:2
        rms, _ = check_cpt(p, e; epss=1.0,
            formulation=TwinMortarFormulation(),
            strategy=ElementBasedIntegration(),
            conforming=false)
        h = 1.0 / (2 * 2^e)  # approximate element size
        rate = isnan(prev_c0) ? NaN : log2(prev_c0 / rms)
        @printf("  %3d  %8.4f  %14.6e  %8.2f\n", e, h, rms, rate)
        prev_c0 = rms
    end
end

run_c0_cpt()
