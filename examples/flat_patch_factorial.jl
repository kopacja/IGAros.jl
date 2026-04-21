# examples/flat_patch_factorial.jl
#
# 2D flat patch test: 3×2 factorial + ε sweep.
# {SP, TM, DPM} × {Elem, Seg} for p = 1, 2, 3, 4.
#
# On a flat interface both integration strategies should give
# equivalent results — any differences are purely formulation + conditioning.

import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using IGAros
using LinearAlgebra, SparseArrays, Printf

include(joinpath(@__DIR__, "cz_cancellation.jl"))

# ─── Solver returning stress error + κ + δ₂ ──────────────────────────────────

function solve_flat_factorial(
    p_ord::Int, n_s::Int, n_m::Int;
    E::Float64 = 1000.0,
    epss::Float64 = 0.0,
    NQUAD_mortar::Int = p_ord + 2,
    strategy::IntegrationStrategy = ElementBasedIntegration(),
    formulation::FormulationStrategy = TwinMortarFormulation(),
)
    nsd = 2; npd = 2; ned = 2

    eps_use = formulation isa SinglePassFormulation ? 0.0 :
              (epss > 0.0 ? epss : 10.0)

    r = flat_patch_test_hp(p_ord, n_s, n_m; E=E, epss=eps_use,
            NQUAD_mortar=NQUAD_mortar, strategy=strategy, formulation=formulation)

    # Displacement error
    disp_err = disp_error(r.U, r.ID, r.B, r.ncp, r.E)

    # Multiplier error: |‖λ^(s)‖_∞ − 1|
    _, Lambda = solve_mortar(r.K_bc, r.C, r.Z, r.F_bc)
    if formulation isa SinglePassFormulation
        n_lam = length(Lambda) ÷ ned
        lam_n = Lambda[1:n_lam]
    else
        n_lam = length(Lambda) ÷ (2 * ned)
        lam_n = Lambda[1:n_lam]
    end
    lam_err = maximum(abs.(lam_n .- 1.0))

    # Condition number
    kappa = NaN
    try; kappa = compute_kappa_flat(r.K_bc, r.C, r.Z); catch; end

    # Force moments
    npc = 2
    p_mat = fill(p_ord, npc, npd)
    # Reconstruct n_mat, KV, P from the solve
    # For p=1: n_mat is known from n_s, n_m
    # For p≥2: after k-refinement
    # Simplest: rebuild just for D,M computation
    pair1 = InterfacePair(1, 2, 2, 4)
    pair2 = InterfacePair(2, 4, 1, 2)

    moments = nothing
    try
        # Re-extract patch data from r
        # The r tuple from flat_patch_test_hp has B, but not n_mat, KV, P, nnp
        # We need to rebuild — use flat_patch_test_hp internals
        # Actually, let's just call build_mortar_mass_matrices via a fresh setup
        r2 = flat_patch_test_hp(p_ord, n_s, n_m; E=E, epss=eps_use,
                NQUAD_mortar=NQUAD_mortar, strategy=strategy, formulation=formulation)
        # Can't easily extract — skip for now if too complex
    catch; end

    return (disp_err=disp_err, lam_err=lam_err, kappa=kappa)
end

# Helper for condition number (2D version)
function compute_kappa_flat(K_bc, C, Z)
    Kd = Matrix(K_bc); Cd = Matrix(C); Zd = Matrix(Z)
    A = [Kd Cd; Cd' -Zd]
    sv = svdvals(A)
    return sv[1] / sv[end]
end

# ─── Factorial study ──────────────────────────────────────────────────────────

function run_flat_factorial(;
    degrees::Vector{Int} = [1, 2, 3, 4],
    n_s::Int = 4, n_m::Int = 7,
    E::Float64 = 1000.0,
    epss::Float64 = 10.0,
)
    configs = [
        ("TME",   TwinMortarFormulation(),  ElementBasedIntegration()),
        ("TMS",   TwinMortarFormulation(),  SegmentBasedIntegration()),
        ("DPME",  DualPassFormulation(),    ElementBasedIntegration()),
        ("DPMS",  DualPassFormulation(),    SegmentBasedIntegration()),
        ("SPME",  SinglePassFormulation(),  ElementBasedIntegration()),
        ("SPMS",  SinglePassFormulation(),  SegmentBasedIntegration()),
    ]

    println("\n", "=" ^ 90)
    @printf("  2D flat patch test: 3×2 factorial, n_s=%d, n_m=%d, E=%.0f, ε=%.0f\n",
            n_s, n_m, E, epss)
    println("=" ^ 90)

    for p in degrees
        @printf("\n─── p = %d ─────────────────────────────────────────────────\n", p)
        @printf("  %-8s  %12s  %12s\n",
                "method", "|u_err|_rel", "κ(A)")
        for (label, form, strat) in configs
            try
                eps_use = form isa SinglePassFormulation ? 0.0 : epss
                r = flat_patch_test_hp(p, n_s, n_m; E=E, epss=eps_use,
                        NQUAD_mortar=p+2, strategy=strat, formulation=form)
                d_err = disp_error(r.U, r.ID, r.B, r.ncp, r.E)
                kappa = compute_kappa_flat(r.K_bc, r.C, r.Z)

                @printf("  %-8s  %12.4e  %12.4e\n", label, d_err, kappa)
            catch ex
                @printf("  %-8s  ERROR: %s\n", label, string(ex)[1:min(60,end)])
            end
            flush(stdout)
        end
    end
end

# ─── ε sweep for flat patch test ──────────────────────────────────────────────

function run_flat_eps_sweep(;
    degrees::Vector{Int} = [1, 2, 3, 4],
    n_s::Int = 4, n_m::Int = 7,
    E::Float64 = 1000.0,
    eps_range = 10.0 .^ (-2:0.5:8),
)
    configs = [
        ("TME",   TwinMortarFormulation(),  ElementBasedIntegration()),
        ("TMS",   TwinMortarFormulation(),  SegmentBasedIntegration()),
        ("DPME",  DualPassFormulation(),    ElementBasedIntegration()),
        ("DPMS",  DualPassFormulation(),    SegmentBasedIntegration()),
    ]

    println("\n", "=" ^ 100)
    @printf("  2D flat patch test: ε sweep, n_s=%d, n_m=%d\n", n_s, n_m)
    println("=" ^ 100)

    for p in degrees
        @printf("\n─── p = %d ───────────────────────────────────────────────\n", p)
        @printf("  %10s  %-8s  %12s  %12s\n",
                "ε", "method", "|u_err|_rel", "κ(A)")
        for eps in eps_range
            for (label, form, strat) in configs
                try
                    r = flat_patch_test_hp(p, n_s, n_m; E=E, epss=Float64(eps),
                            NQUAD_mortar=p+2, strategy=strat, formulation=form)
                    d_err = disp_error(r.U, r.ID, r.B, r.ncp, r.E)
                    kappa = compute_kappa_flat(r.K_bc, r.C, r.Z)

                    @printf("  %10.2e  %-8s  %12.4e  %12.4e\n",
                            eps, label, d_err, kappa)
                catch ex
                    @printf("  %10.2e  %-8s  ERROR: %s\n", eps, label,
                            string(ex)[1:min(50,end)])
                end
            end
            flush(stdout)
        end
    end
end
