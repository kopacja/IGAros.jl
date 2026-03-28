"""
ε-scaling analysis: extract C₀ from error ≈ C₀/ε and check h-dependence.

The goal is to derive a practical formula for ε_opt.
From DPM vs TM comparison we know the ε-sensitivity is NOT from quadrature error
but from the Z-block perturbation. Here we characterise it quantitatively.
"""

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using IGAros
using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "concentric_cylinders.jl"))

function run_eps_scaling()
    degrees   = [2, 3, 4]
    exp_range = 1:4          # multiple refinement levels
    # Use DPM (exact integration) to isolate the Z-perturbation from quadrature
    eps_probe = [1.0, 10.0]  # two points in the 1/ε regime to extract slope

    println("═"^78)
    println("  ε-scaling analysis: C₀ from error ≈ C₀/ε")
    println("  Using DPM (segment-based, exact integration)")
    println("═"^78)

    # ── Part 1: Extract C₀ for each (p, exp) ────────────────────────────────
    println("\n── Part 1: C₀ extraction (error at ε=1 and ε=10) ──")
    @printf("  %3s  %3s  %8s  %12s  %12s  %12s  %12s\n",
            "p", "exp", "h", "err(ε=1)", "err(ε=10)", "C₀≈err·ε", "C₀(from10)")
    @printf("  %s\n", "─"^72)

    C0_data = Dict{Tuple{Int,Int}, Float64}()

    for p in degrees
        for exp in exp_range
            h = 0.5 / 2^exp
            errs = Float64[]
            for eps in eps_probe
                rel, _ = solve_cylinder(p, exp;
                    epss=eps,
                    formulation=DualPassFormulation(),
                    strategy=SegmentBasedIntegration())
                push!(errs, rel)
            end
            # C₀ = error × ε  (should be constant in the 1/ε regime)
            c0_1  = errs[1] * 1.0
            c0_10 = errs[2] * 10.0
            C0_data[(p,exp)] = c0_1
            @printf("  %3d  %3d  %8.4f  %12.4e  %12.4e  %12.4e  %12.4e\n",
                    p, exp, h, errs[1], errs[2], c0_1, c0_10)
        end
        println()
    end

    # ── Part 2: h-scaling of C₀ ──────────────────────────────────────────────
    println("\n── Part 2: h-scaling of C₀ ──")
    @printf("  %3s  %3s  %8s  %12s  %8s\n", "p", "exp", "h", "C₀", "rate")
    @printf("  %s\n", "─"^40)
    for p in degrees
        prev_c0 = NaN
        for exp in exp_range
            h = 0.5 / 2^exp
            c0 = C0_data[(p,exp)]
            rate = isnan(prev_c0) ? NaN : log2(prev_c0 / c0)
            @printf("  %3d  %3d  %8.4f  %12.4e  %8.2f\n", p, exp, h, c0, rate)
            prev_c0 = c0
        end
        println()
    end

    # ── Part 3: Full ε-sweep at two exp levels to verify 1/ε scaling ─────────
    println("\n── Part 3: Full ε-sweep verification (p=2, exp=2 and exp=4) ──")
    for exp in [2, 4]
        println("\n  exp = $exp, h = $(0.5/2^exp)")
        eps_full = 10 .^ (-1:0.5:8)
        @printf("  %10s  %14s  %14s\n", "ε", "DPM error", "error×ε")
        @printf("  %s\n", "─"^40)
        for eps in eps_full
            rel, _ = solve_cylinder(2, exp;
                epss=Float64(eps),
                formulation=DualPassFormulation(),
                strategy=SegmentBasedIntegration())
            @printf("  %10.1e  %14.4e  %14.4e\n", eps, rel, rel*eps)
        end
    end

    # ── Part 4: Schur complement analysis — ratio λ_max(S₀)/λ_min(K) ────────
    println("\n\n── Part 4: Key spectral ratios ──")
    @printf("  %3s  %3s  %8s  %12s  %12s  %12s  %12s  %12s\n",
            "p", "exp", "h", "λ_min(K)", "λ_max(S₀)", "C₀_meas", "S₀/K ratio", "C₀/ratio")
    @printf("  %s\n", "─"^90)
    for p in degrees
        for exp in [2, 3]
            h = 0.5 / 2^exp
            K_bc, C, Z = cylinder_matrices(p, exp;
                formulation=TwinMortarFormulation(),
                strategy=ElementBasedIntegration(),
                epss=1e6)
            Kd = Matrix(K_bc); Cd = Matrix(C); Zd = Matrix(Z)
            Md = Zd ./ 1e6

            # K eigenvalues
            λ_K = eigvals(Hermitian(0.5*(Kd+Kd')))
            λ_K_min = minimum(filter(x -> x > 1e-14*maximum(λ_K), λ_K))
            λ_K_max = maximum(λ_K)

            # S₀ = C^T K^{-1} C
            KinvC = lu(Kd) \ Cd
            S0 = Cd' * KinvC
            S0 = 0.5*(S0+S0')
            λ_S = eigvals(Hermitian(S0))
            λ_S_pos = filter(x -> abs(x) > 1e-14*maximum(abs.(λ_S)), λ_S)
            λ_S_max = maximum(abs.(λ_S_pos))

            c0 = C0_data[(p,exp)]
            ratio = λ_S_max / λ_K_min
            @printf("  %3d  %3d  %8.4f  %12.4e  %12.4e  %12.4e  %12.4e  %12.4e\n",
                    p, exp, h, λ_K_min, λ_S_max, c0, ratio, c0/ratio)
        end
    end
end

run_eps_scaling()
