# examples/farah_patch_test.jl
#
# 3D planar patch test with T-junction (Farah et al. 2015 setup).
# Two blocks of different size: lower (larger) and upper (smaller).
# Uniform compression σ_zz = -p₀ with free-rim loading.
#
# Configurations:
#   A) "Overlapping master": upper=slave, lower=master (easy case)
#   B) "Overlapping slave":  lower=slave, upper=master (T-junction, hard case)

import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using IGAros
using LinearAlgebra, SparseArrays, Printf

include(joinpath(@__DIR__, "cz_cancellation.jl"))  # compute_force_moments
include(joinpath(@__DIR__, "pressurized_sphere.jl")) # _compute_kappa, open_uniform_kv

# ─── Geometry ─────────────────────────────────────────────────────────────────

"""
    farah_geometry_p1(n_lower, n_upper, n_z;
        L_lower=1.0, L_upper=0.6, H=0.5)
        -> (B, P, p_mat, n_mat, KV, npc)

Two-block p=1 geometry for the Farah patch test.
  Block 1 (lower): [0, L_lower]² × [0, H]
  Block 2 (upper): [offset, offset+L_upper]² × [H, 2H]
where offset = (L_lower - L_upper)/2 (centered).

Parametrisation per block (ξ=x, η=y, ζ=z):
  Facet 6 (ζ=n₃): top face
  Facet 1 (ζ=1):   bottom face
"""
function farah_geometry_p1(n_lower::Int, n_upper::Int, n_z::Int;
    L_lower::Float64 = 1.0,
    L_upper::Float64 = 0.6,
    H::Float64       = 0.5
)
    npc = 2; npd = 3; p_ord = 1
    offset = (L_lower - L_upper) / 2

    function build_block(nx, ny, x0, x1, y0, y1, z0, z1)
        n1 = nx + 1; n2 = ny + 1; n3 = n_z + 1
        B = zeros(n1 * n2 * n3, 4)
        for k in 1:n3, j in 1:n2, i in 1:n1
            a = (k-1)*n1*n2 + (j-1)*n1 + i
            B[a, 1] = x0 + (i-1)/nx * (x1 - x0)
            B[a, 2] = y0 + (j-1)/ny * (y1 - y0)
            B[a, 3] = z0 + (k-1)/n_z * (z1 - z0)
            B[a, 4] = 1.0
        end
        kv_x = open_uniform_kv(nx, 1)
        kv_y = open_uniform_kv(ny, 1)
        kv_z = open_uniform_kv(n_z, 1)
        return B, [kv_x, kv_y, kv_z], n1, n2, n3
    end

    # Block 1 (lower, larger)
    B1, KV1, n1x, n1y, n1z = build_block(n_lower, n_lower,
        0.0, L_lower, 0.0, L_lower, 0.0, H)
    # Block 2 (upper, smaller)
    B2, KV2, n2x, n2y, n2z = build_block(n_upper, n_upper,
        offset, offset + L_upper, offset, offset + L_upper, H, 2H)

    ncp1 = size(B1, 1); ncp2 = size(B2, 1)
    B = vcat(B1, B2)
    P = [collect(1:ncp1), collect(ncp1+1:ncp1+ncp2)]

    p_mat = fill(1, npc, npd)
    n_mat = [n1x n1y n1z; n2x n2y n2z]
    KV = [KV1, KV2]

    return B, P, p_mat, n_mat, KV, npc
end

# ─── Free-rim Neumann loading ────────────────────────────────────────────────

"""
    segment_load_free_rim(... ; master_p, master_n, master_KV, master_P, master_B)

Apply Neumann traction on a face, but SKIP Gauss points that project
inside the master surface footprint (contact zone).
Used for loading the free rim of the lower block's top face.
"""
function segment_load_free_rim(
    n, p, KV, P, B, nnp, nen_full, nsd, npd, ned,
    facet, ID, F, traction, thickness, NQUAD;
    master_p, master_n, master_KV, master_P, master_B
)
    F = copy(F)

    ps, ns, KVs, Ps, nsn, nsen, nsel, norm_sign, _, _, _ =
        get_segment_patch(p, n, KV, P, npd, nnp, facet)

    ien_s = build_ien(nsd, npd - 1, 1,
                      reshape(ps, 1, :), reshape(ns, 1, :),
                      [nsel], [nsn], [nsen])
    ien = ien_s[1]
    inc = build_inc(ns)

    # Master boundary data (for projection check)
    master_nnp = prod(master_n)
    pm, nm, KVm, Pm, _, _, _, _, _, _, _ =
        get_segment_patch(master_p, master_n, master_KV, master_P, npd, master_nnp, 1)
        # facet 1 = bottom of upper block

    GPW = gauss_product(NQUAD, npd - 1)

    for el in 1:nsel
        anchor = ien[el, 1]
        n0     = inc[anchor]
        Fs = zeros(ned, nsen)

        for (gp, gw) in GPW
            R, _, _, detJ, n_vec = shape_function(
                ps, ns, KVs, B, Ps, gp, nsen, nsd, npd - 1, el, n0, ien, inc)
            detJ <= 0 && continue
            n_vec .*= norm_sign

            # Physical coordinates of this GP
            Xe = B[Ps[ien[el, :]], :]
            X  = Xe' * R

            # Check if this point lies inside the master surface footprint.
            # For a flat interface, check physical bounding box of master CPs.
            master_x = [master_B[cp, 1] for cp in Pm]
            master_y = [master_B[cp, 2] for cp in Pm]
            xmin_m, xmax_m = extrema(master_x)
            ymin_m, ymax_m = extrema(master_y)
            tol_geom = 1e-10
            inside = (X[1] > xmin_m + tol_geom && X[1] < xmax_m - tol_geom &&
                      X[2] > ymin_m + tol_geom && X[2] < ymax_m - tol_geom)

            # Skip if inside contact zone
            inside && continue

            gwJ = gw * detJ * thickness

            if traction isa Function
                σ = traction(X[1:nsd]...)
                Fp = σ * n_vec
            else
                Fp = traction .* n_vec
            end

            Fs .+= Fp * R' .* gwJ
        end

        for a in 1:nsen
            cp = Ps[ien[el, a]]
            for i in 1:ned
                eq = ID[i, cp]
                eq != 0 && (F[eq] += Fs[i, a])
            end
        end
    end

    return F
end

# ─── Solver ───────────────────────────────────────────────────────────────────

"""
    solve_farah_patch(n_lower, n_upper, n_z;
        config=:overlapping_master,
        E, nu, p0, epss, NQUAD, NQUAD_mortar,
        strategy, formulation) -> NamedTuple

Farah-style 3D patch test.
  config = :overlapping_master → upper=slave, lower=master (easy)
  config = :overlapping_slave  → lower=slave, upper=master (T-junction)
"""
function solve_farah_patch(
    n_lower::Int = 4, n_upper::Int = 3, n_z::Int = 2;
    config::Symbol             = :overlapping_master,
    L_lower::Float64           = 1.0,
    L_upper::Float64           = 0.6,
    H::Float64                 = 0.5,
    E::Float64                 = 100.0,
    nu::Float64                = 0.0,
    p0::Float64                = 0.5,
    epss::Float64              = 0.0,
    NQUAD::Int                 = 2,
    NQUAD_mortar::Int          = 3,
    strategy::IntegrationStrategy    = ElementBasedIntegration(),
    formulation::FormulationStrategy = TwinMortarFormulation(),
)
    nsd = 3; npd = 3; ned = 3; npc = 2

    B_ref, P_ref, p_mat, n_mat, KV, _ =
        farah_geometry_p1(n_lower, n_upper, n_z;
                          L_lower=L_lower, L_upper=L_upper, H=H)
    ncp = size(B_ref, 1)
    epss_use = epss > 0.0 ? epss : 100.0

    nel, nnp, nen = patch_metrics(npc, npd, p_mat, n_mat)
    IEN = build_ien(nsd, npd, npc, p_mat, n_mat, nel, nnp, nen)
    INC = [build_inc(n_mat[pc, :]) for pc in 1:npc]

    # BCs: uz=0 on bottom of lower block (facet 1, patch 1)
    # Also fix rigid body: ux=0 on x=0, uy=0 on y=0
    dBC = [3 1 1 1 0;    # uz=0 on bottom of block 1
           1 4 1 1 0;    # ux=0 on x=0 face of block 1
           2 5 1 1 0]    # uy=0 on y=0 face of block 1
    bc_per_dof = dirichlet_bc_control_points(p_mat, n_mat, KV, P_ref,
                                              npd, nnp, ned, dBC)
    neq, ID = build_id(bc_per_dof, ned, ncp)
    LM = build_lm(nen, ned, npc, nel, ID, IEN, P_ref)

    mats = [LinearElastic(E, nu, :three_d), LinearElastic(E, nu, :three_d)]
    Ub0 = zeros(ncp, nsd)
    K, _ = build_stiffness_matrix(npc, nsd, npd, ned, neq,
                                   p_mat, n_mat, KV, P_ref, B_ref, Ub0,
                                   nen, nel, IEN, INC, LM, mats, NQUAD, 1.0)

    # Loading: σ_zz = -p0 (compression)
    traction_fn = (x, y, z) -> begin
        σ = zeros(3, 3); σ[3, 3] = -p0; σ
    end

    F = zeros(neq)

    # Load top of upper block (facet 6, patch 2) — full face
    F = segment_load(n_mat[2,:], p_mat[2,:], KV[2], P_ref[2], B_ref,
                     nnp[2], nen[2], nsd, npd, ned,
                     Int[], 6, ID, F, traction_fn, 1.0, NQUAD)

    # Load free rim of lower block's top face (facet 6, patch 1)
    # Skip GPs that project inside upper block's bottom face (facet 1, patch 2)
    F = segment_load_free_rim(
        n_mat[1,:], p_mat[1,:], KV[1], P_ref[1], B_ref,
        nnp[1], nen[1], nsd, npd, ned,
        6, ID, F, traction_fn, 1.0, NQUAD;
        master_p = p_mat[2,:], master_n = n_mat[2,:],
        master_KV = KV[2], master_P = P_ref[2], master_B = B_ref)

    K_bc, F_bc = enforce_dirichlet(Tuple{Int,Float64}[], K, F)

    # Mortar coupling
    if config == :overlapping_master
        # Upper = slave (smaller), Lower = master (larger)
        # Slave facet 1 (bottom of upper) ↔ Master facet 6 (top of lower)
        pair1 = InterfacePair(2, 1, 1, 6)
        pair2 = InterfacePair(1, 6, 2, 1)
    else  # :overlapping_slave
        # Lower = slave (larger), Upper = master (smaller) — T-junction
        pair1 = InterfacePair(1, 6, 2, 1)
        pair2 = InterfacePair(2, 1, 1, 6)
    end

    pairs_full = [pair1, pair2]
    pairs_sp   = [pair1]
    pairs = formulation isa SinglePassFormulation ? pairs_sp : pairs_full

    eps_use = formulation isa SinglePassFormulation ? 0.0 : epss_use
    Pc = build_interface_cps(pairs, p_mat, n_mat, KV, P_ref, npd, nnp, formulation)
    C, Z = build_mortar_coupling(Pc, pairs, p_mat, n_mat, KV, P_ref, B_ref,
                                  ID, nnp, ned, nsd, npd, neq, NQUAD_mortar, eps_use,
                                  strategy, formulation)

    U, _ = solve_mortar(K_bc, C, Z, F_bc)

    # Stress error: exact is σ_zz = -p0 everywhere
    rms_zz, max_zz = _stress_error_farah(
        U, ID, npc, nsd, npd, p_mat, n_mat, KV, P_ref, B_ref,
        nen, nel, IEN, INC, mats, NQUAD, p0)

    # Condition number
    kappa = NaN
    ndof = neq + size(C, 2)
    if ndof < 15000
        try; kappa = _compute_kappa(K_bc, C, Z); catch; end
    end

    return (rms_zz=rms_zz, max_zz=max_zz, kappa=kappa, neq=neq, n_lam=size(C, 2))
end

# ─── Stress error ─────────────────────────────────────────────────────────────

function _stress_error_farah(
    U, ID, npc, nsd, npd, p_mat, n_mat, KV, P, B,
    nen, nel, IEN, INC, mats, NQUAD, p0
)
    ncp = size(B, 1)
    Ub = zeros(ncp, nsd)
    for A in 1:ncp, i in 1:nsd
        eq = ID[i, A]; eq != 0 && (Ub[A, i] = U[eq])
    end

    err2 = 0.0; vol = 0.0
    GPW = gauss_product(NQUAD, npd)

    for pc in 1:npc
        ien = IEN[pc]; inc = INC[pc]
        D = elastic_constants(mats[pc], nsd)
        for el in 1:nel[pc]
            anchor = ien[el, 1]; n0 = inc[anchor]
            for (gp, gw) in GPW
                R_s, dR_dx, _, detJ, _ = shape_function(
                    p_mat[pc,:], n_mat[pc,:], KV[pc], B, P[pc], gp,
                    nen[pc], nsd, npd, el, n0, ien, inc)
                detJ <= 0 && continue
                gwJ = gw * detJ
                B0 = strain_displacement_matrix(nsd, nen[pc], dR_dx')
                Ue = vec(Ub[P[pc][ien[el,:]], 1:nsd]')
                σ_h = D * (B0 * Ue)  # Voigt: [xx, yy, zz, xy, yz, zx]
                σ_zz_err = σ_h[3] - (-p0)
                err2 += σ_zz_err^2 * gwJ
                vol += gwJ
            end
        end
    end
    rms = sqrt(err2 / vol)
    return rms, sqrt(err2)
end

# ─── Quick test ───────────────────────────────────────────────────────────────

function test_farah()
    println("=== Overlapping master (easy) ===")
    r = solve_farah_patch(4, 3, 2; config=:overlapping_master, epss=100.0)
    @printf("  RMS σ_zz = %.4e, κ = %.4e\n", r.rms_zz, r.kappa)

    println("=== Overlapping slave (T-junction) ===")
    r2 = solve_farah_patch(4, 3, 2; config=:overlapping_slave, epss=100.0)
    @printf("  RMS σ_zz = %.4e, κ = %.4e\n", r2.rms_zz, r2.kappa)
end
