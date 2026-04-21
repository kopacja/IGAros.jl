# examples/spherical_patch_test.jl
#
# 3D curved-interface patch test with a SPHERICAL interface.
# Unit box [0,1]³ split by z_int(x,y) = z0 + √(R² − (x−cx)² − (y−cy)²) − R₀
# where R₀ = √(R² − cx² − cy²) so that z_int(0,0) = z0.
#
# The sphere is irrational — no polynomial B-spline represents it exactly.
# Geometry is constructed via Greville interpolation at each refinement level.
#
# Exact solution: σ_zz = −1, all others 0, u_z = −z/E.
# Loading: Neumann t_z = −1 on top face of upper patch.
# BCs: ux=0 on x=0, uy=0 on y=0, uz=0 on z=0 (lower patch).

import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using IGAros
using LinearAlgebra, SparseArrays, Printf

# Load helpers
include(joinpath(@__DIR__, "pressurized_sphere.jl"))
include(joinpath(@__DIR__, "curved_patch_test.jl"))
include(joinpath(@__DIR__, "cz_cancellation.jl"))

# ─── Spherical interface ──────────────────────────────────────────────────────

"""
    sphere_z(x, y; R, cx, cy, z0) -> z_int

Spherical cap interface:
  z_int(x,y) = z0 + √(R² − (x−cx)² − (y−cy)²) − √(R² − cx² − cy²)

Anchored so z_int(0,0) = z0.  R must be large enough that the argument
of the square root is positive over [0,1]².
"""
function sphere_z(x::Real, y::Real; R=1.2, cx=0.5, cy=0.5, z0=0.5)
    R0 = sqrt(R^2 - cx^2 - cy^2)
    arg = R^2 - (x - cx)^2 - (y - cy)^2
    return z0 + sqrt(arg) - R0
end

# ─── Geometry via Greville interpolation ──────────────────────────────────────

"""
    spt_geometry(p_ord, n_x_lower, n_x_upper, n_y_lower, n_y_upper, n_z;
                 R, cx, cy, z0, L_z) -> (B, P, p_mat, n_mat, KV, npc)

Build two-patch geometry for the spherical patch test using Greville
interpolation.  CPs are computed so the B-spline surface interpolates
the exact sphere at Greville abscissae.
"""
function spt_geometry(
    p_ord::Int, n_x_lower::Int, n_x_upper::Int,
    n_y_lower::Int, n_y_upper::Int, n_z::Int;
    R=1.2, cx=0.5, cy=0.5, z0=0.5, L_z=1.0
)
    npc = 2; npd = 3

    function _build_patch(n_x, n_y, z_bot_fn, z_top_fn)
        n_cpx = n_x + p_ord
        n_cpy = n_y + p_ord
        n_cpz = n_z + p_ord
        kv_x = open_uniform_kv(n_x, p_ord)
        kv_y = open_uniform_kv(n_y, p_ord)
        kv_z = open_uniform_kv(n_z, p_ord)
        g_x = _greville(kv_x, p_ord, n_cpx)
        g_y = _greville(kv_y, p_ord, n_cpy)
        g_z = _greville(kv_z, p_ord, n_cpz)
        N_x = _basis_matrix(kv_x, p_ord, n_cpx)
        N_y = _basis_matrix(kv_y, p_ord, n_cpy)
        N_z = _basis_matrix(kv_z, p_ord, n_cpz)

        # Target positions at Greville points
        T = zeros(n_cpx, n_cpy, n_cpz, 3)
        for kk in 1:n_cpz
            ζ = g_z[kk]
            for jj in 1:n_cpy
                y_val = g_y[jj]
                for ii in 1:n_cpx
                    x_val = g_x[ii]
                    zb = z_bot_fn(x_val, y_val)
                    zt = z_top_fn(x_val, y_val)
                    T[ii, jj, kk, 1] = x_val
                    T[ii, jj, kk, 2] = y_val
                    T[ii, jj, kk, 3] = zb + ζ * (zt - zb)
                end
            end
        end

        # Solve for CPs via tensor-product interpolation: N_x P N_y^T N_z^T = T
        # Apply dimension by dimension
        CPs = zeros(n_cpx, n_cpy, n_cpz, 3)
        for d in 1:3
            # Solve in x: N_x * Cx = Tx  for each (j,k)
            Cx = zeros(n_cpx, n_cpy, n_cpz)
            for kk in 1:n_cpz, jj in 1:n_cpy
                Cx[:, jj, kk] = N_x \ T[:, jj, kk, d]
            end
            # Solve in y: Cy * N_y^T = Cx  for each (i,k)
            Cy = zeros(n_cpx, n_cpy, n_cpz)
            for kk in 1:n_cpz, ii in 1:n_cpx
                Cy[ii, :, kk] = N_y \ Cx[ii, :, kk]
            end
            # Solve in z: Cz * N_z^T = Cy  for each (i,j)
            for jj in 1:n_cpy, ii in 1:n_cpx
                CPs[ii, jj, :, d] = N_z \ Cy[ii, jj, :]
            end
        end

        # Flatten to (ncp, 4)
        ncp = n_cpx * n_cpy * n_cpz
        B = zeros(ncp, 4)
        for kk in 1:n_cpz, jj in 1:n_cpy, ii in 1:n_cpx
            a = (kk-1)*n_cpx*n_cpy + (jj-1)*n_cpx + ii
            B[a, 1:3] .= CPs[ii, jj, kk, :]
            B[a, 4] = 1.0
        end
        return B, [kv_x, kv_y, kv_z], n_cpx, n_cpy, n_cpz
    end

    z_int(x, y) = sphere_z(x, y; R=R, cx=cx, cy=cy, z0=z0)

    # Patch 1 (lower): z ∈ [0, z_int(x,y)]
    B1, KV1, n1x, n1y, n1z = _build_patch(n_x_lower, n_y_lower,
        (x,y) -> 0.0, z_int)
    # Patch 2 (upper): z ∈ [z_int(x,y), L_z]
    B2, KV2, n2x, n2y, n2z = _build_patch(n_x_upper, n_y_upper,
        z_int, (x,y) -> L_z)

    ncp1 = size(B1, 1); ncp2 = size(B2, 1)
    B = vcat(B1, B2)
    P = [collect(1:ncp1), collect(ncp1+1:ncp1+ncp2)]

    p_mat = fill(p_ord, npc, npd)
    n_mat = [n1x n1y n1z; n2x n2y n2z]
    KV = [KV1, KV2]

    return B, P, p_mat, n_mat, KV, npc
end

# ─── Solver ───────────────────────────────────────────────────────────────────

"""
    solve_spt(p_ord, exp_level; ...) -> NamedTuple

Solve the spherical-interface patch test.  Returns stress errors and
optionally the mortar mass matrices for force-moment analysis.
"""
function solve_spt(
    p_ord::Int, exp_level::Int;
    R=1.2, cx=0.5, cy=0.5, z0=0.5, L_z=1.0,
    E=1e5, nu=0.3,
    epss=0.0,
    NQUAD::Int = p_ord + 1,
    NQUAD_mortar::Int = p_ord + 4,
    strategy::IntegrationStrategy = ElementBasedIntegration(),
    formulation::FormulationStrategy = TwinMortarFormulation(),
    n_x_lower_base::Int = 3, n_x_upper_base::Int = 2,
    n_y_lower_base::Int = 3, n_y_upper_base::Int = 2,
    n_z_base::Int = 1,
    return_matrices::Bool = false,
)
    nsd = 3; npd = 3; ned = 3; npc = 2

    n_x_l = n_x_lower_base * 2^exp_level
    n_x_u = n_x_upper_base * 2^exp_level
    n_y_l = n_y_lower_base * 2^exp_level
    n_y_u = n_y_upper_base * 2^exp_level
    n_z   = n_z_base * 2^exp_level

    B_ref, P_ref, p_mat, n_mat_ref, KV_ref, _ =
        spt_geometry(p_ord, n_x_l, n_x_u, n_y_l, n_y_u, n_z;
                     R=R, cx=cx, cy=cy, z0=z0, L_z=L_z)
    ncp = size(B_ref, 1)
    epss_use = epss > 0.0 ? epss : 1e6

    nel, nnp, nen = patch_metrics(npc, npd, p_mat, n_mat_ref)
    IEN = build_ien(nsd, npd, npc, p_mat, n_mat_ref, nel, nnp, nen)
    INC = [build_inc(n_mat_ref[pc,:]) for pc in 1:npc]

    # BCs: ux=0 on x=0 (facet 4), uy=0 on y=0 (facet 5), uz=0 on z=0 (patch 1 facet 1)
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

    # Neumann: t_z = -1 on Patch 2 top face (facet 6)
    traction_fn = (x,y,z) -> (σ = zeros(3,3); σ[3,3] = -1.0; σ)
    F = zeros(neq)
    F = segment_load(n_mat_ref[2,:], p_mat[2,:], KV_ref[2], P_ref[2], B_ref,
                     nnp[2], nen[2], nsd, npd, ned,
                     Int[], 6, ID, F, traction_fn, 1.0, NQUAD)
    K_bc, F_bc = enforce_dirichlet(Tuple{Int,Float64}[], K, F)

    # Mortar coupling
    pairs_full = [InterfacePair(1, 6, 2, 1), InterfacePair(2, 1, 1, 6)]
    pairs_sp   = [InterfacePair(1, 6, 2, 1)]
    pairs = formulation isa SinglePassFormulation ? pairs_sp : pairs_full

    Pc = build_interface_cps(pairs, p_mat, n_mat_ref, KV_ref, P_ref, npd, nnp, formulation)
    eps_use = formulation isa SinglePassFormulation ? 0.0 : epss_use
    C, Z = build_mortar_coupling(Pc, pairs, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
                                  ID, nnp, ned, nsd, npd, neq, NQUAD_mortar, eps_use,
                                  strategy, formulation)

    U, _ = solve_mortar(K_bc, C, Z, F_bc)

    # Stress error
    rms_zz, max_zz, rms_all, max_all = stress_error_cpt(
        U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
        nen, nel, IEN, INC, E, nu, NQUAD)

    # Force moments (per half-pass) — element-based only (segment-based
    # build_mortar_mass_matrices not implemented for 3D surfaces)
    moments = nothing
    if return_matrices && strategy isa ElementBasedIntegration
        pair1 = InterfacePair(1, 6, 2, 1)
        pair2 = InterfacePair(2, 1, 1, 6)
        D1, M12, s1, m1 = build_mortar_mass_matrices(
            pair1, p_mat, n_mat_ref, KV_ref, P_ref, B_ref, nnp, nsd, npd,
            NQUAD_mortar, strategy)
        mom1 = compute_force_moments(D1, M12, s1, m1, B_ref; dim=3)

        if !(formulation isa SinglePassFormulation)
            D2, M21, s2, m2 = build_mortar_mass_matrices(
                pair2, p_mat, n_mat_ref, KV_ref, P_ref, B_ref, nnp, nsd, npd,
                NQUAD_mortar, strategy)
            mom2 = compute_force_moments(D2, M21, s2, m2, B_ref; dim=3)
            moments = (pass1=mom1, pass2=mom2,
                       sum_δ2 = mom1.δ_2 + mom2.δ_2)
        else
            moments = (pass1=mom1, pass2=nothing, sum_δ2=mom1.δ_2)
        end
    end

    # Condition number (small systems only)
    kappa = NaN
    if neq + size(C, 2) < 5000
        try
            kappa = _compute_kappa(K_bc, C, Z)
        catch; end
    end

    return (rms_zz=rms_zz, max_zz=max_zz, rms_all=rms_all, max_all=max_all,
            kappa=kappa, moments=moments)
end

# ─── Factorial study ──────────────────────────────────────────────────────────

"""
    run_spt_factorial(; degrees, exp_range, epss, R)

Run the 3×2 factorial: {SP, TM, DPM} × {Elem, Seg} on the spherical
patch test.  Reports RMS stress error, convergence rate, κ(A), and
force-moment δ₂ cancellation.
"""
function run_spt_factorial(;
    degrees::Vector{Int} = [2, 3, 4],
    exp_range::UnitRange{Int} = 0:3,
    epss::Float64 = 1e6,
    R::Float64 = 1.2,
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
    @printf("  Spherical patch test: 3×2 factorial, R=%.1f, ε=%.0e\n", R, epss)
    println("=" ^ 120)

    for p in degrees
        @printf("\n─── p = %d ──────────────────────────────────────────────────────────────────────────\n", p)

        for (label, form, strat) in configs
            @printf("\n  Method: %s\n", label)
            @printf("  %5s  %12s  %6s  %12s  %14s  %14s  %14s  %8s\n",
                    "exp", "RMS σ_zz", "rate", "κ(A)",
                    "δ₂(pass1)", "δ₂(pass2)", "δ₂(sum)", "t(s)")
            prev = NaN

            for e in exp_range
                t0 = time()
                try
                    r = solve_spt(p, e; epss=epss, R=R,
                            strategy=strat, formulation=form,
                            return_matrices=true, kwargs...)
                    dt = time() - t0
                    rate = isnan(prev) ? NaN : log(prev / r.rms_zz) / log(2.0)

                    if !isnothing(r.moments)
                        d2_1 = r.moments.pass1.δ_2
                        d2_2 = isnothing(r.moments.pass2) ? NaN : r.moments.pass2.δ_2
                        d2_s = r.moments.sum_δ2
                        @printf("  %5d  %12.4e  %6.2f  %12.4e  %14.4e  %14.4e  %14.4e  %8.1f\n",
                                e, r.rms_zz, rate, r.kappa, d2_1, d2_2, d2_s, dt)
                    else
                        @printf("  %5d  %12.4e  %6.2f  %12.4e  %14s  %14s  %14s  %8.1f\n",
                                e, r.rms_zz, rate, r.kappa, "—", "—", "—", dt)
                    end
                    prev = r.rms_zz
                catch ex
                    dt = time() - t0
                    @printf("  %5d  ERROR (%.1fs): %s\n", e, dt, string(ex)[1:min(80, end)])
                    prev = NaN
                end
                flush(stdout)
            end
        end
    end

    println("\n", "=" ^ 120)
    println("  Done")
    println("=" ^ 120)
end
