# examples/poly_patch_test.jl
#
# Polynomial-matched curved-interface patch test.
# For each degree p, the interface is a degree-p polynomial:
#   p=2: z_int = z0 + δ·ξ·(1-ξ)                    (parabolic)
#   p=3: z_int = z0 + δ·ξ·(1-ξ)·(1+ξ)              (cubic)
#   p=4: z_int = z0 + δ·ξ²·(1-ξ)²                  (quartic)
#
# The interface lies exactly in the B-spline space of degree p,
# so any stress error is PURELY due to the mortar coupling method.
#
# Exact solution: σ_zz = −1, all others 0, u_z = −z/E.

import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using IGAros
using LinearAlgebra, SparseArrays, Printf

# Load helpers
include(joinpath(@__DIR__, "bending_beam.jl"))     # _elevate_x/y/z_beam
include(joinpath(@__DIR__, "curved_patch_test.jl")) # stress_error_cpt
include(joinpath(@__DIR__, "cz_cancellation.jl"))   # compute_force_moments
include(joinpath(@__DIR__, "pressurized_sphere.jl")) # _compute_kappa

# ─── Bernstein CPs for the degree-p arc (1D, in ξ ∈ [0,1]) ──────────────────

"""
    _arc_bernstein_cps(p, δ) -> Vector{Float64}

Bernstein control points for the degree-p interface arc f(ξ) on [0,1],
where f(0) = f(1) = 0 and max|f| ≈ δ.

  p=2: f(ξ) = δ·ξ·(1-ξ)           → CPs = [0, δ/2, 0]
  p=3: f(ξ) = δ·ξ·(1-ξ)·(1+ξ)    → CPs = [0, δ/3, 2δ/3, 0]
  p=4: f(ξ) = 16δ/3·ξ²·(1-ξ)²    → CPs = [0, 0, 8δ/9, 0, 0]
"""
function _arc_bernstein_cps(p::Int, δ::Float64)
    if p == 2
        # f(ξ) = δ·ξ·(1-ξ) in Bernstein-2: B₀=1·(1-ξ)², B₁=2ξ(1-ξ), B₂=ξ²
        # f = δ·(ξ - ξ²) = δ·(½·B₁) = 0·B₀ + (δ/2)·B₁ + 0·B₂
        return [0.0, δ/2, 0.0]
    elseif p == 3
        # f(ξ) = δ·ξ·(1-ξ)·(1+ξ) = δ·(ξ - ξ³)
        # Bernstein-3: B₀=(1-ξ)³, B₁=3ξ(1-ξ)², B₂=3ξ²(1-ξ), B₃=ξ³
        # ξ   = 0·B₀ + (1/3)·B₁ + (2/3)·B₂ + 1·B₃
        # ξ³  = 0·B₀ + 0·B₁ + 0·B₂ + 1·B₃
        # f = δ·(ξ - ξ³) = δ·[0, 1/3, 2/3, 0]
        return [0.0, δ/3, 2δ/3, 0.0]
    elseif p == 4
        # f(ξ) = c·ξ²·(1-ξ)² with c chosen so max ≈ δ: max at ξ=1/2 → c/16
        # Set c = 16δ so f(1/2) = δ.  f = 16δ·ξ²·(1-ξ)²
        # Bernstein-4: B₀=(1-ξ)⁴, B₁=4ξ(1-ξ)³, B₂=6ξ²(1-ξ)², B₃=4ξ³(1-ξ), B₄=ξ⁴
        # ξ²(1-ξ)² = B₂/6
        # f = 16δ · B₂/6 = (8δ/3) · B₂
        # CPs: [0, 0, 8δ/3, 0, 0]
        return [0.0, 0.0, 8δ/3, 0.0, 0.0]
    else
        error("_arc_bernstein_cps: only p = 2, 3, 4 supported")
    end
end

# ─── Geometry construction ────────────────────────────────────────────────────

"""
    ppt_geometry(p_ord; L_x, L_y, L_z, δ) -> (B, P)

Two-patch NURBS geometry for the polynomial-matched patch test.
The interface is a tensor-product degree-p polynomial in x and y.
z-direction is linear (2 CPs).

Initial Bezier element: n_x = n_y = p+1, n_z = 2.
"""
function ppt_geometry(p_ord::Int;
                       L_x=1.0, L_y=1.0, L_z=1.0, δ=0.15)

    arc_x = _arc_bernstein_cps(p_ord, δ)
    arc_y = _arc_bernstein_cps(p_ord, δ)
    n_ang = p_ord + 1  # CPs per surface direction (single Bezier element)
    n_z   = 2          # linear in z

    # x and y Bezier CPs (equally spaced for the identity map ξ → x)
    x_cps = [L_x * i / p_ord for i in 0:p_ord]
    y_cps = [L_y * j / p_ord for j in 0:p_ord]

    B1 = zeros(n_ang * n_ang * n_z, 4)
    B2 = zeros(n_ang * n_ang * n_z, 4)

    for k in 1:n_z, j in 1:n_ang, i in 1:n_ang
        ai = (k-1)*n_ang*n_ang + (j-1)*n_ang + i
        z_int = L_z/2 + arc_x[i] + arc_y[j]
        B1[ai,:] = [x_cps[i], y_cps[j], (k==1 ? 0.0  : z_int), 1.0]
        B2[ai,:] = [x_cps[i], y_cps[j], (k==1 ? z_int : L_z),  1.0]
    end

    # Elevate z from linear (p=1) to degree p
    B1, _ = _elevate_z_beam(B1, n_ang, n_ang, n_z, p_ord - 1)
    B2, _ = _elevate_z_beam(B2, n_ang, n_ang, n_z, p_ord - 1)

    ncp1 = size(B1, 1); ncp2 = size(B2, 1)
    return vcat(B1, B2), [collect(1:ncp1), collect(ncp1+1:ncp1+ncp2)]
end

# ─── Solver ───────────────────────────────────────────────────────────────────

"""
    solve_ppt(p_ord, exp_level; ...) -> NamedTuple

Solve the polynomial-matched patch test.  Returns RMS stress error,
condition number, and optionally force-moment δ₂ data.
"""
function solve_ppt(
    p_ord::Int, exp_level::Int;
    L_x=1.0, L_y=1.0, L_z=1.0, δ=0.15,
    E=1e5, nu=0.3,
    epss=0.0,
    NQUAD::Int = p_ord + 1,
    NQUAD_mortar::Int = p_ord + 4,
    strategy::IntegrationStrategy = ElementBasedIntegration(),
    formulation::FormulationStrategy = TwinMortarFormulation(),
    n_x_lower_base::Int = 3, n_x_upper_base::Int = 2,
    n_y_lower_base::Int = 3, n_y_upper_base::Int = 2,
    return_matrices::Bool = false,
)
    nsd = 3; npd = 3; ned = 3; npc = 2

    # Build geometry
    B0, P = ppt_geometry(p_ord; L_x=L_x, L_y=L_y, L_z=L_z, δ=δ)
    p_mat = fill(p_ord, npc, npd)
    n_mat = fill(p_ord + 1, npc, npd)
    KV = generate_knot_vectors(npc, npd, p_mat, n_mat)

    # z-offset hack to prevent CP merging during k-refinement
    B0_hack = copy(B0); B0_hack[P[1], 3] .+= 1000.0

    n_x  = n_x_upper_base * 2^exp_level
    n_xl = n_x_lower_base * 2^exp_level
    n_y  = n_y_upper_base * 2^exp_level
    n_yl = n_y_lower_base * 2^exp_level
    n_z  = max(1, 2^exp_level)

    kref_data = Vector{Float64}[
        vcat([1.0,1.0], [i/n_xl for i in 1:n_xl-1]),
        vcat([1.0,2.0], [i/n_yl for i in 1:n_yl-1]),
        vcat([1.0,3.0], [i/n_z  for i in 1:n_z -1]),
        vcat([2.0,1.0], [i/n_x  for i in 1:n_x -1]),
        vcat([2.0,2.0], [i/n_y  for i in 1:n_y -1]),
        vcat([2.0,3.0], [i/n_z  for i in 1:n_z -1]),
    ]

    n_mat_ref, _, KV_ref, B_hack, P_ref = krefinement(
        nsd, npd, npc, n_mat, p_mat, KV, B0_hack, P, kref_data)
    B_ref = copy(B_hack)
    for i in axes(B_ref, 1)
        B_ref[i,3] > 100.0 && (B_ref[i,3] -= 1000.0)
    end

    epss_use = epss > 0.0 ? epss : 1e6
    ncp = size(B_ref, 1)

    nel, nnp, nen = patch_metrics(npc, npd, p_mat, n_mat_ref)
    IEN = build_ien(nsd, npd, npc, p_mat, n_mat_ref, nel, nnp, nen)
    INC = [build_inc(n_mat_ref[pc,:]) for pc in 1:npc]

    dBC = [1 4 2 1 2;
           2 5 2 1 2;
           3 1 1 1 0]
    bc_per_dof = dirichlet_bc_control_points(p_mat, n_mat_ref, KV_ref, P_ref,
                                              npd, nnp, ned, dBC)
    neq, ID = build_id(bc_per_dof, ned, ncp)
    LM = build_lm(nen, ned, npc, nel, ID, IEN, P_ref)

    mats = [LinearElastic(E, nu, :three_d), LinearElastic(E, nu, :three_d)]
    Ub0 = zeros(ncp, nsd)
    K, _ = build_stiffness_matrix(npc, nsd, npd, ned, neq,
                                   p_mat, n_mat_ref, KV_ref, P_ref, B_ref, Ub0,
                                   nen, nel, IEN, INC, LM, mats, NQUAD, 1.0)

    traction_fn = (x,y,z) -> (σ = zeros(3,3); σ[3,3] = -1.0; σ)
    F = zeros(neq)
    F = segment_load(n_mat_ref[2,:], p_mat[2,:], KV_ref[2], P_ref[2], B_ref,
                     nnp[2], nen[2], nsd, npd, ned,
                     Int[], 6, ID, F, traction_fn, 1.0, NQUAD)
    K_bc, F_bc = enforce_dirichlet(Tuple{Int,Float64}[], K, F)

    pairs_full = [InterfacePair(1, 6, 2, 1), InterfacePair(2, 1, 1, 6)]
    pairs_sp   = [InterfacePair(1, 6, 2, 1)]
    pairs = formulation isa SinglePassFormulation ? pairs_sp : pairs_full

    Pc = build_interface_cps(pairs, p_mat, n_mat_ref, KV_ref, P_ref, npd, nnp, formulation)
    eps_use = formulation isa SinglePassFormulation ? 0.0 : epss_use
    C, Z = build_mortar_coupling(Pc, pairs, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
                                  ID, nnp, ned, nsd, npd, neq, NQUAD_mortar, eps_use,
                                  strategy, formulation)

    U, _ = solve_mortar(K_bc, C, Z, F_bc)

    rms_zz, max_zz, rms_all, max_all = stress_error_cpt(
        U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
        nen, nel, IEN, INC, E, nu, NQUAD)

    # Force moments — always use element-based D,M so we measure the
    # element-based integration error for ALL methods (segment-based
    # build_mortar_mass_matrices is not available for 3D surfaces).
    moments = nothing
    if return_matrices
        eb = ElementBasedIntegration()
        pair1 = InterfacePair(1, 6, 2, 1)
        pair2 = InterfacePair(2, 1, 1, 6)
        D1, M12, s1, m1 = build_mortar_mass_matrices(
            pair1, p_mat, n_mat_ref, KV_ref, P_ref, B_ref, nnp, nsd, npd,
            NQUAD_mortar, eb)
        mom1 = compute_force_moments(D1, M12, s1, m1, B_ref; dim=3)
        if !(formulation isa SinglePassFormulation)
            D2, M21, s2, m2 = build_mortar_mass_matrices(
                pair2, p_mat, n_mat_ref, KV_ref, P_ref, B_ref, nnp, nsd, npd,
                NQUAD_mortar, eb)
            mom2 = compute_force_moments(D2, M21, s2, m2, B_ref; dim=3)
            moments = (pass1=mom1, pass2=mom2, sum_δ2=mom1.δ_2 + mom2.δ_2)
        else
            moments = (pass1=mom1, pass2=nothing, sum_δ2=mom1.δ_2)
        end
    end

    kappa = NaN
    try; kappa = _compute_kappa(K_bc, C, Z); catch; end

    return (rms_zz=rms_zz, max_zz=max_zz, rms_all=rms_all, max_all=max_all,
            kappa=kappa, moments=moments)
end

# ─── Factorial study ──────────────────────────────────────────────────────────

function run_ppt_factorial(;
    degrees::Vector{Int} = [2, 3, 4],
    exp_range::UnitRange{Int} = 0:3,
    epss::Float64 = 1e6,
    δ::Float64 = 0.15,
    kwargs...
)
    configs = [
        ("TME",   TwinMortarFormulation(),  ElementBasedIntegration()),
        ("TMS",   TwinMortarFormulation(),  SegmentBasedIntegration()),
        ("DPME",  DualPassFormulation(),    ElementBasedIntegration()),
        ("DPMS",  DualPassFormulation(),    SegmentBasedIntegration()),
        ("SPME",  SinglePassFormulation(),  ElementBasedIntegration()),
        ("SPMS",  SinglePassFormulation(),  SegmentBasedIntegration()),
    ]

    println("\n", "=" ^ 120)
    @printf("  Polynomial-matched patch test: 3×2 factorial, δ=%.2f, ε=%.0e\n", δ, epss)
    println("  Interface: degree p polynomial for each p (exact in B-spline space)")
    println("=" ^ 120)

    for p in degrees
        @printf("\n─── p = %d ──────────────────────────────────────────────────────────────────────────\n", p)
        for (label, form, strat) in configs
            @printf("\n  Method: %s\n", label)
            @printf("  %5s  %12s  %12s  %14s  %14s  %14s  %8s\n",
                    "exp", "RMS σ_zz", "κ(A)",
                    "δ₂(pass1)", "δ₂(pass2)", "δ₂(sum)", "t(s)")
            for e in exp_range
                t0 = time()
                try
                    r = solve_ppt(p, e; epss=epss, δ=δ,
                            strategy=strat, formulation=form,
                            return_matrices=true, kwargs...)
                    dt = time() - t0
                    if !isnothing(r.moments)
                        d2_1 = r.moments.pass1.δ_2
                        d2_2 = isnothing(r.moments.pass2) ? NaN : r.moments.pass2.δ_2
                        d2_s = r.moments.sum_δ2
                        @printf("  %5d  %12.4e  %12.4e  %14.4e  %14.4e  %14.4e  %8.1f\n",
                                e, r.rms_zz, r.kappa, d2_1, d2_2, d2_s, dt)
                    else
                        @printf("  %5d  %12.4e  %12.4e  %14s  %14s  %14s  %8.1f\n",
                                e, r.rms_zz, r.kappa, "—", "—", "—", dt)
                    end
                catch ex
                    dt = time() - t0
                    @printf("  %5d  ERROR (%.1fs): %s\n", e, dt, string(ex)[1:min(80, end)])
                end
                flush(stdout)
            end
        end
    end
end

# ─── ε sweep ──────────────────────────────────────────────────────────────────

function run_ppt_eps_sweep(;
    degrees::Vector{Int} = [2, 3, 4],
    exp_level::Int = 1,
    eps_range::Vector{Float64} = 10.0 .^ (-2:0.5:8),
    δ::Float64 = 0.15,
    kwargs...
)
    configs = [
        ("TME",   TwinMortarFormulation(),  ElementBasedIntegration()),
        ("DPME",  DualPassFormulation(),    ElementBasedIntegration()),
    ]

    println("\n", "=" ^ 100)
    @printf("  Polynomial-matched patch test: ε sweep, exp=%d, δ=%.2f\n", exp_level, δ)
    println("=" ^ 100)

    for p in degrees
        @printf("\n─── p = %d ───────────────────────────────────────────────\n", p)
        @printf("  %10s  %-8s  %12s  %12s\n", "ε", "method", "RMS σ_zz", "κ(A)")
        for eps in eps_range
            for (label, form, strat) in configs
                try
                    r = solve_ppt(p, exp_level; epss=eps, δ=δ,
                            strategy=strat, formulation=form, kwargs...)
                    @printf("  %10.2e  %-8s  %12.4e  %12.4e\n",
                            eps, label, r.rms_zz, r.kappa)
                catch ex
                    @printf("  %10.2e  %-8s  ERROR: %s\n", eps, label,
                            string(ex)[1:min(60,end)])
                end
            end
            flush(stdout)
        end
    end
end
