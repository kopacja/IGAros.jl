"""
Spectral analysis of Schur complement S₀ = CᵀK⁻¹C vs mortar mass M = Z/ε
for the concentric cylinder benchmark.

Goal: understand the ε-scaling by comparing eigenvalue spectra of S₀ and M.
The optimal ε balances ε·λ(M) ≈ λ(S₀).
"""

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using IGAros
using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "concentric_cylinders.jl"))

function run_schur_spectrum(;
    degrees::Vector{Int} = [1, 2, 3, 4],
    exp_level::Int       = 3,
    epss_ref::Float64    = 1e6,   # used to build Z; M = Z/ε
)
    println("═"^78)
    println("  Schur complement spectral analysis: S₀ = CᵀK⁻¹C  vs  M = Z/ε")
    println("  Concentric cylinders, exp = $exp_level")
    println("═"^78)

    for p in degrees
        println("\n── p = $p ──────────────────────────────────────────────────")

        # Build matrices with TM formulation
        K_bc, C, Z = cylinder_matrices(p, exp_level;
            formulation = TwinMortarFormulation(),
            strategy    = ElementBasedIntegration(),
            epss        = epss_ref)

        Kd = Matrix(K_bc)
        Cd = Matrix(C)
        Zd = Matrix(Z)

        n_u = size(Kd, 1)
        n_λ = size(Cd, 2)

        # M = Z / ε  (the mortar mass matrix without ε scaling)
        Md = Zd ./ epss_ref

        # Schur complement S₀ = Cᵀ K⁻¹ C
        # Use factorisation for numerical stability
        K_fact = lu(Kd)
        KinvC = K_fact \ Cd           # K⁻¹ C  (n_u × n_λ)
        S0 = Cd' * KinvC              # Cᵀ K⁻¹ C  (n_λ × n_λ)

        # Symmetrise (S₀ should be symmetric but may have roundoff)
        S0 = 0.5 * (S0 + S0')
        Md_sym = 0.5 * (Md + Md')

        # Eigenvalues (real symmetric → use eigen for Hermitian)
        λ_S = eigvals(Hermitian(S0))
        λ_M = eigvals(Hermitian(Md_sym))

        # Filter out near-zero eigenvalues (numerical noise)
        tol = 1e-14 * maximum(abs.(λ_S))
        λ_S_pos = filter(x -> abs(x) > tol, λ_S)
        λ_M_pos = filter(x -> abs(x) > tol, λ_M)

        @printf("  System sizes:  n_u = %d,  n_λ = %d\n", n_u, n_λ)
        @printf("  S₀ eigenvalues:  min = %10.3e,  max = %10.3e,  κ(S₀) = %10.3e\n",
                minimum(abs.(λ_S_pos)), maximum(abs.(λ_S_pos)),
                maximum(abs.(λ_S_pos)) / minimum(abs.(λ_S_pos)))
        @printf("  M  eigenvalues:  min = %10.3e,  max = %10.3e,  κ(M)  = %10.3e\n",
                minimum(abs.(λ_M_pos)), maximum(abs.(λ_M_pos)),
                maximum(abs.(λ_M_pos)) / minimum(abs.(λ_M_pos)))

        # The balance point: ε_opt ≈ λ(S₀) / λ(M)
        # Report ratio for min, max, and geometric mean
        r_min = minimum(abs.(λ_S_pos)) / minimum(abs.(λ_M_pos))
        r_max = maximum(abs.(λ_S_pos)) / maximum(abs.(λ_M_pos))
        r_geo = sqrt(r_min * r_max)

        @printf("\n  Balance ratios  ε_opt = λ(S₀)/λ(M):\n")
        @printf("    from smallest eigvals:  ε_min = %10.3e\n", r_min)
        @printf("    from largest  eigvals:  ε_max = %10.3e\n", r_max)
        @printf("    geometric mean:         ε_geo = %10.3e\n", r_geo)

        # Full spectrum table (sorted)
        println("\n  Full spectrum comparison (sorted ascending):")
        @printf("  %5s  %14s  %14s  %14s\n", "idx", "λ(S₀)", "λ(M)", "λ(S₀)/λ(M)")
        @printf("  %s\n", "─"^52)
        n_show = min(length(λ_S), length(λ_M))
        # Sort both ascending by absolute value
        λ_S_sorted = sort(λ_S, by=abs)
        λ_M_sorted = sort(λ_M, by=abs)
        for i in 1:n_show
            ratio = abs(λ_S_sorted[i]) > 1e-20 && abs(λ_M_sorted[i]) > 1e-20 ?
                    λ_S_sorted[i] / λ_M_sorted[i] : NaN
            @printf("  %5d  %14.4e  %14.4e  %14.4e\n",
                    i, λ_S_sorted[i], λ_M_sorted[i], ratio)
        end

        # Also compute: what ε makes κ(S₀ + ε·M) minimal?
        println("\n  κ(S₀ + ε·M) scan:")
        @printf("  %10s  %14s\n", "ε", "κ(S₀+ε·M)")
        @printf("  %s\n", "─"^26)
        for k in -2:8
            eps_test = 10.0^k
            S_reg = S0 + eps_test .* Md_sym
            λ_reg = eigvals(Hermitian(S_reg))
            λ_reg_pos = filter(x -> abs(x) > 1e-20, λ_reg)
            if !isempty(λ_reg_pos)
                κ_reg = maximum(abs.(λ_reg_pos)) / minimum(abs.(λ_reg_pos))
                @printf("  %10.0e  %14.3e\n", eps_test, κ_reg)
            else
                @printf("  %10.0e  %14s\n", eps_test, "singular")
            end
        end
    end
end

run_schur_spectrum()
