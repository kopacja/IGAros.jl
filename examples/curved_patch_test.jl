# examples/curved_patch_test.jl
#
# 3D curved interface patch test — verifies that a constant stress state
# σ_zz = −1 is transmitted exactly across a curved, non-conforming interface.
#
# A patch test is passed if the stress error is at (or near) machine precision
# for ANY mesh, not just as mesh is refined.
#
# Reference: Song et al. (2017) Sec. 4.2 "Patch test with a curved interface"
#
# GEOMETRY
#   Box:  x ∈ [0, L_x],  y ∈ [0, L_y],  z ∈ [0, L_z]
#   Interface: z_int(x,y) = L_z/2 + arc_amp·(x/L_x)·(1−x/L_x)
#                                  + arc_amp_y·(y/L_y)·(1−y/L_y)
#     Parabolic dome with curvature in BOTH x and y directions.
#     Ref: Dittmann et al. (2019) §7.1
#   Patch 1 (bottom, slave):   z ∈ [0,          z_int(x,y)]
#   Patch 2 (top,    master):  z ∈ [z_int(x,y), L_z      ]
#
# MATERIAL   E = 1e5,  ν = 0.3  (Song et al. 2017)
#
# LOADING    t_z = −1 (Neumann) on Patch 2 top face (z = L_z)
#
# BOUNDARY CONDITIONS
#   ux = 0 on facet 4 (ξ=1, x=0),  both patches
#   uy = 0 on facet 5 (η=1, y=0),  both patches
#   uz = 0 on facet 1 (ζ=1, z=0),  Patch 1 only
#
# EXACT SOLUTION
#   σ_zz = −1,  all other σ_ij = 0
#   ux = ν/E·x,  uy = ν/E·y,  uz = −z/E
#
# PARAMETRIZATION  (ξ=x fastest, η=y middle, ζ=z slowest)
#   Facets:  1=ζ-min  2=ξ-max  3=η-max  4=ξ-min  5=η-min  6=ζ-max
#   Interface: Patch 1 facet 6 (top) ↔ Patch 2 facet 1 (bottom)

import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using IGAros
using LinearAlgebra, SparseArrays, Printf

include(joinpath(@__DIR__, "bending_beam.jl"))   # Bezier elevation helpers

# ─── Exact solution ────────────────────────────────────────────────────────────

function cpt_exact_disp(x::Real, y::Real, z::Real; E=1e5, nu=0.3)
    return nu/E * x, nu/E * y, -z/E
end

# ─── Arc interface ──────────────────────────────────────────────────────────────

"""
    arc_z_cpt(x, y; L_x, L_y, L_z, arc_amp, arc_amp_y) -> z_int

Parabolic dome interface with curvature in both x and y:
  z_int(x,y) = L_z/2 + arc_amp·(x/L_x)·(1−x/L_x) + arc_amp_y·(y/L_y)·(1−y/L_y)

Flat at domain edges (x=0, x=L_x, y=0, y=L_y); maximum at center.
Reference: Dittmann et al. (2019) §7.1 — curved interface patch test.
"""
function arc_z_cpt(x::Real, y::Real;
                   L_x=1.0, L_y=1.0, L_z=1.0,
                   arc_amp=0.3, arc_amp_y=0.3)
    ξ = x / L_x
    η = y / L_y
    return L_z/2 + arc_amp * ξ * (1.0 - ξ) + arc_amp_y * η * (1.0 - η)
end

# ─── p≥2 geometry ──────────────────────────────────────────────────────────────

"""
    cpt_geometry(p_ord; L_x, L_y, L_z, arc_amp, arc_amp_y) -> (B, P)

Two-patch NURBS geometry for the curved-interface patch test.
Interface is a quadratic Bernstein polynomial dome in BOTH x and y —
exactly representable for p_ord ≥ 2.  Bezier z-control points form a
tensor product of two parabolic arcs.

Initial coarse geometry: p=[2,2,1], n=[3,3,2] per patch.
"""
function cpt_geometry(p_ord::Int;
                       L_x=1.0, L_y=1.0, L_z=1.0,
                       arc_amp=0.3, arc_amp_y=0.3)

    # Bernstein CPs for additive parabolic arcs (relative to L_z/2)
    # ξ*(1-ξ) in degree-2 Bernstein: CPs = [0, 1/2, 0]
    z_arc_x = [0.0, arc_amp/2, 0.0]
    z_arc_y = [0.0, arc_amp_y/2, 0.0]

    x_cps = [0.0, L_x/2, L_x]
    y_cps = [0.0, L_y/2, L_y]
    n1, n2, n3 = 3, 3, 2

    B1 = zeros(n1*n2*n3, 4);  B2 = zeros(n1*n2*n3, 4)
    for k in 1:n3, j in 1:n2, i in 1:n1
        ai = (k-1)*n1*n2 + (j-1)*n1 + i
        z_int = L_z/2 + z_arc_x[i] + z_arc_y[j]
        B1[ai,:] = [x_cps[i], y_cps[j], (k==1 ? 0.0    : z_int), 1.0]
        B2[ai,:] = [x_cps[i], y_cps[j], (k==1 ? z_int   : L_z),  1.0]
    end

    B1, n1a = _elevate_x_beam(B1, n1, n2, n3, p_ord-2)
    B2, _   = _elevate_x_beam(B2, n1, n2, n3, p_ord-2)
    B1, n2a = _elevate_y_beam(B1, n1a, n2, n3, p_ord-2)
    B2, _   = _elevate_y_beam(B2, n1a, n2, n3, p_ord-2)
    B1, _   = _elevate_z_beam(B1, n1a, n2a, n3, p_ord-1)
    B2, _   = _elevate_z_beam(B2, n1a, n2a, n3, p_ord-1)

    ncp1 = size(B1,1); ncp2 = size(B2,1)
    return vcat(B1,B2), [collect(1:ncp1), collect(ncp1+1:ncp1+ncp2)]
end

# ─── p=1 geometry (direct mesh) ────────────────────────────────────────────────

"""
    cpt_geometry_p1(n_x_lower, n_x_upper, n_y, n_z; ...) -> (B, P)

Bilinear (p=1) geometry with CPs placed directly on the 2D arc surface.
Geometry error O(h²) < O(h) discretisation error.
CPs interpolate the curved interface exactly at nodal positions.
"""
function cpt_geometry_p1(n_x_lower, n_x_upper, n_y_lower, n_y_upper, n_z;
                          L_x=1.0, L_y=1.0, L_z=1.0,
                          arc_amp=0.3, arc_amp_y=0.3)
    n1_1=n_x_lower+1; n2_1=n_y_lower+1; n3_1=n_z+1
    B1 = zeros(n1_1*n2_1*n3_1, 4)
    for k in 1:n3_1, j in 1:n2_1, i in 1:n1_1
        ai = (k-1)*n1_1*n2_1 + (j-1)*n1_1 + i
        xi = (i-1)/n_x_lower*L_x
        yj = (j-1)/n_y_lower*L_y
        za = arc_z_cpt(xi, yj; L_x=L_x, L_y=L_y, L_z=L_z,
                        arc_amp=arc_amp, arc_amp_y=arc_amp_y)
        B1[ai,:] = [xi, yj, (k-1)/n_z*za, 1.0]
    end

    n1_2=n_x_upper+1; n2_2=n_y_upper+1; n3_2=n_z+1
    B2 = zeros(n1_2*n2_2*n3_2, 4)
    for k in 1:n3_2, j in 1:n2_2, i in 1:n1_2
        ai = (k-1)*n1_2*n2_2 + (j-1)*n1_2 + i
        xi = (i-1)/n_x_upper*L_x
        yj = (j-1)/n_y_upper*L_y
        za = arc_z_cpt(xi, yj; L_x=L_x, L_y=L_y, L_z=L_z,
                        arc_amp=arc_amp, arc_amp_y=arc_amp_y)
        B2[ai,:] = [xi, yj, za+(k-1)/n_z*(L_z-za), 1.0]
    end

    ncp1=size(B1,1); ncp2=size(B2,1)
    return vcat(B1,B2), [collect(1:ncp1), collect(ncp1+1:ncp1+ncp2)]
end

# ─── Stress error evaluation ────────────────────────────────────────────────────

"""
    stress_error_cpt(U, ID, npc, nsd, npd, p, n, KV, P, B, nen, nel, IEN, INC,
                     E, nu, NQUAD) -> (rms_zz, max_zz, rms_all, max_all)

Evaluate the stress error against the exact field σ_zz=−1, all others=0.
  rms_zz  — RMS of |σ_zz − (−1)| over domain (volume-weighted)
  max_zz  — pointwise maximum of |σ_zz − (−1)|
  rms_all — RMS of all 6 stress component errors combined
  max_all — pointwise maximum of combined stress error
"""
function stress_error_cpt(
    U::Vector{Float64}, ID::Matrix{Int},
    npc::Int, nsd::Int, npd::Int,
    p::Matrix{Int}, n::Matrix{Int},
    KV, P, B::Matrix{Float64},
    nen::Vector{Int}, nel::Vector{Int},
    IEN, INC, E::Float64, nu::Float64, NQUAD::Int
)
    mat = LinearElastic(E, nu, :three_d)
    D   = elastic_constants(mat, nsd)

    ncp = size(B,1)
    Ub  = zeros(ncp, nsd)
    for A in 1:ncp, i in 1:nsd
        eq = ID[i,A]; eq != 0 && (Ub[A,i] = U[eq])
    end

    err2_zz = 0.0; err2_all = 0.0; vol = 0.0
    max_zz  = 0.0; max_all  = 0.0
    GPW = gauss_product(NQUAD, npd)

    for pc in 1:npc
        ien = IEN[pc]; inc = INC[pc]
        for el in 1:nel[pc]
            n0 = inc[ien[el,1]]
            for (gp, gw) in GPW
                R, dR_dx, _, detJ, _ = shape_function(
                    p[pc,:], n[pc,:], KV[pc], B, P[pc], gp,
                    nen[pc], nsd, npd, el, n0, ien, inc)
                detJ <= 0 && continue
                gwJ = gw * detJ

                el_nodes = P[pc][ien[el,:]]
                Ue = Ub[el_nodes, :]
                u_vec = vec(Ue')         # [u1x,u1y,u1z, u2x,u2y,u2z, ...]
                dN_dX = Matrix(dR_dx')   # nsd × nen
                Bmat  = strain_displacement_matrix(nsd, nen[pc], dN_dX)
                sig   = D * (Bmat * u_vec)   # [σxx,σyy,σzz,τxy,τyz,τzx]

                # Exact: σ_zz=−1, rest=0
                e = sig - [-0.0, -0.0, -1.0, 0.0, 0.0, 0.0]
                e_zz = abs(e[3]); e_tot = norm(e)

                err2_zz += e_zz^2 * gwJ
                err2_all += e_tot^2 * gwJ
                vol      += gwJ
                max_zz    = max(max_zz, e_zz)
                max_all   = max(max_all, e_tot)
            end
        end
    end

    return sqrt(err2_zz/vol), max_zz, sqrt(err2_all/vol), max_all
end

# ─── Mesh modification (Puso 2004) ──────────────────────────────────────────────
#
# Eliminates the L² geometric gap at a non-conforming curved interface by
# projecting the master interface geometry onto the slave basis:
#   X_new^(s) = D^{-1} M X^(m)
# where D = ∫ R_s R_s^T dΓ and M = ∫ R_s R_m^T dΓ.

"""
    apply_mesh_modification!(B, pair, p, n, KV, P, nnp, nsd, npd, NQUAD;
                              strategy=ElementBasedIntegration())

Modify the slave interface control points so that D·X^(s) = M·X^(m),
eliminating the L² initial gap for the given InterfacePair.
Modifies `B` in-place and returns the CP displacement norm.
"""
function apply_mesh_modification!(
    B::Matrix{Float64}, pair::InterfacePair,
    p::Matrix{Int}, n::Matrix{Int},
    KV, P, nnp, nsd, npd, NQUAD;
    strategy::IntegrationStrategy = ElementBasedIntegration()
)
    D, M_mat, slave_ifc_cps, master_ifc_cps =
        build_mortar_mass_matrices(pair, p, n, KV, P, B, nnp, nsd, npd, NQUAD, strategy)

    # Solve D * X_new = M * X_master  for each spatial coordinate
    D_dense = Matrix(D)
    dx_norm = 0.0
    for d in 1:nsd
        X_m = B[master_ifc_cps, d]
        rhs = M_mat * X_m
        X_new = D_dense \ rhs
        for (i, cp) in enumerate(slave_ifc_cps)
            dx_norm += (X_new[i] - B[cp, d])^2
            B[cp, d] = X_new[i]
        end
    end
    return sqrt(dx_norm)
end

# ─── Internal solver (returns full mesh state for postprocessing) ───────────────
#
# dirichlet=false (default): Neumann t_z=−1 on Patch 2 top face (tests force transfer)
# dirichlet=true:  Prescribe uz=−L_z/E on Patch 2 top face instead (standard patch test)
#   With dirichlet=true, no force transfer is needed → λ=0 is consistent → machine precision.

function _cpt_solve(p_ord, exp_level;
                    conforming=false, L_x=1.0, L_y=1.0, L_z=1.0,
                    arc_amp=0.3, arc_amp_y=0.3,
                    E=1e5, nu=0.3, epss=0.0, NQUAD=p_ord+1, NQUAD_mortar=p_ord+4,
                    strategy=ElementBasedIntegration(), formulation=TwinMortarFormulation(),
                    normal_strategy=SlaveNormal(),
                    n_x_lower_base=3, n_x_upper_base=2,
                    n_y_lower_base=3, n_y_upper_base=2,
                    dirichlet::Bool=false,
                    mesh_modification::Bool=false)

    nsd=3; npd=3; ned=3; npc=2

    if p_ord == 1
        n_x_u = n_x_upper_base * 2^exp_level
        n_x_l = conforming ? n_x_u : n_x_lower_base * 2^exp_level
        n_y_u = n_y_upper_base * 2^exp_level
        n_y_l = conforming ? n_y_u : n_y_lower_base * 2^exp_level
        n_z   = max(1, 2^exp_level)

        B_ref, P_ref = cpt_geometry_p1(n_x_l, n_x_u, n_y_l, n_y_u, n_z;
                                        L_x=L_x, L_y=L_y, L_z=L_z,
                                        arc_amp=arc_amp, arc_amp_y=arc_amp_y)
        p_mat     = fill(1, npc, npd)
        n_mat_ref = [n_x_l+1 n_y_l+1 n_z+1; n_x_u+1 n_y_u+1 n_z+1]
        KV_ref    = [[open_uniform_kv(n_x_l,1), open_uniform_kv(n_y_l,1), open_uniform_kv(n_z,1)],
                     [open_uniform_kv(n_x_u,1), open_uniform_kv(n_y_u,1), open_uniform_kv(n_z,1)]]
        NQUAD_use = (NQUAD == p_ord+1) ? 2 : NQUAD
        NQUAD_m   = (NQUAD_mortar == p_ord+2) ? 3 : NQUAD_mortar
    else
        B0, P  = cpt_geometry(p_ord; L_x=L_x, L_y=L_y, L_z=L_z,
                               arc_amp=arc_amp, arc_amp_y=arc_amp_y)
        p_mat  = fill(p_ord, npc, npd)
        n_mat  = fill(p_ord+1, npc, npd)
        KV     = generate_knot_vectors(npc, npd, p_mat, n_mat)

        # z-offset hack: prevent interface CP merging during k-refinement
        B0_hack = copy(B0);  B0_hack[P[1], 3] .+= 1000.0

        n_x  = n_x_upper_base * 2^exp_level
        n_xl = conforming ? n_x : n_x_lower_base * 2^exp_level
        n_y  = n_y_upper_base * 2^exp_level
        n_yl = conforming ? n_y : n_y_lower_base * 2^exp_level
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
        for i in axes(B_ref,1)
            B_ref[i,3] > 100.0 && (B_ref[i,3] -= 1000.0)
        end

        NQUAD_use = NQUAD; NQUAD_m = NQUAD_mortar
    end

    epss_use = epss > 0.0 ? epss : 1.0e6
    ncp = size(B_ref, 1)

    nel, nnp, nen = patch_metrics(npc, npd, p_mat, n_mat_ref)
    IEN = build_ien(nsd, npd, npc, p_mat, n_mat_ref, nel, nnp, nen)
    INC = [build_inc(n_mat_ref[pc,:]) for pc in 1:npc]

    # ── Mesh modification (Puso 2004): eliminate L² gap ────────────────────
    if mesh_modification
        pair1 = InterfacePair(1, 6, 2, 1)
        max_iter = 20; tol_gap = 1e-12

        # Iterate: modify interface CPs only (no interior propagation yet)
        for it in 1:max_iter
            dx = apply_mesh_modification!(B_ref, pair1, p_mat, n_mat_ref,
                                           KV_ref, P_ref, nnp, nsd, npd, NQUAD_m,
                                           strategy=strategy)

            D_chk, M_chk, s_chk, m_chk = build_mortar_mass_matrices(
                pair1, p_mat, n_mat_ref, KV_ref, P_ref, B_ref, nnp, nsd, npd, NQUAD_m,
                strategy)
            gap = norm(D_chk * B_ref[s_chk, 1:nsd] - M_chk * B_ref[m_chk, 1:nsd])

            @printf "  mesh mod iter %2d: ||ΔX||=%.3e  gap=%.3e\n" it dx gap
            gap < tol_gap && break
        end

        # Propagate final interface positions to interior CPs of Patch 1
        n1_1 = n_mat_ref[1,1]; n2_1 = n_mat_ref[1,2]; n3_1 = n_mat_ref[1,3]
        for j in 1:n2_1, i in 1:n1_1
            top_cp  = P_ref[1][(n3_1-1)*n1_1*n2_1 + (j-1)*n1_1 + i]
            z_new_top = B_ref[top_cp, 3]
            for k in 2:n3_1-1
                cp = P_ref[1][(k-1)*n1_1*n2_1 + (j-1)*n1_1 + i]
                B_ref[cp, 3] = (k-1) / (n3_1-1) * z_new_top
            end
        end
    end

    # dBC: [dof, facet, n_patches, pc1, ...]
    # ux=0 on facet4 (ξ=1,x=0), uy=0 on facet5 (η=1,y=0), uz=0 on facet1 (Patch1 only)
    dBC = [1 4 2 1 2;
           2 5 2 1 2;
           3 1 1 1 0]

    bc_per_dof = dirichlet_bc_control_points(p_mat, n_mat_ref, KV_ref, P_ref,
                                              npd, nnp, ned, dBC)
    neq, ID = build_id(bc_per_dof, ned, ncp)
    LM      = build_lm(nen, ned, npc, nel, ID, IEN, P_ref)

    mats = [LinearElastic(E, nu, :three_d), LinearElastic(E, nu, :three_d)]
    Ub0  = zeros(ncp, nsd)
    K, _ = build_stiffness_matrix(npc, nsd, npd, ned, neq,
                                   p_mat, n_mat_ref, KV_ref, P_ref, B_ref, Ub0,
                                   nen, nel, IEN, INC, LM, mats, NQUAD_use, 1.0)

    F = zeros(neq)
    IND = Tuple{Int,Float64}[]

    if dirichlet
        # ── Dirichlet patch test: prescribe uz = −z/E on Patch 2 top face ────
        # σ_zz = −1 requires  uz = −L_z/E  on z = L_z.
        # No Neumann load needed — λ = 0 is consistent → machine-precision stress.
        tol_geom = 1e-8 * L_z
        for loc_A in 1:nnp[2]
            cp = P_ref[2][loc_A]
            z  = B_ref[cp, 3]
            if abs(z - L_z) < tol_geom
                eq = ID[3, cp]
                eq != 0 && push!(IND, (eq, -L_z / E))
            end
        end
        unique!(IND)
    else
        # ── Neumann patch test: traction t_z = −1 on Patch 2 top face ────────
        traction_fn = (x,y,z) -> (σ=zeros(3,3); σ[3,3]=-1.0; σ)
        F = segment_load(n_mat_ref[2,:], p_mat[2,:], KV_ref[2], P_ref[2], B_ref,
                          nnp[2], nen[2], nsd, npd, ned,
                          Int[], 6, ID, F, traction_fn, 1.0, NQUAD_use)
    end

    K_bc, F_bc = enforce_dirichlet(IND, K, F)

    pairs_full = [InterfacePair(1,6,2,1), InterfacePair(2,1,1,6)]
    pairs_sp   = [InterfacePair(1,6,2,1)]
    pairs = formulation isa SinglePassFormulation ? pairs_sp : pairs_full

    Pc = build_interface_cps(pairs, p_mat, n_mat_ref, KV_ref, P_ref, npd, nnp, formulation)
    C, Z = build_mortar_coupling(Pc, pairs, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
                                  ID, nnp, ned, nsd, npd, neq, NQUAD_m, epss_use,
                                  strategy, formulation, normal_strategy)

    rows_C, cols_C, vals_C = findnz(C)
    C_bc = sparse(rows_C, cols_C, vals_C, size(C,1), size(C,2))
    _, cols_nz, _ = findnz(C_bc)
    active_lm = sort(unique(cols_nz))
    if length(active_lm) < size(C_bc,2)
        C_bc = C_bc[:, active_lm]; Z = Z[active_lm, active_lm]
    end

    U, _ = solve_mortar(K_bc, C_bc, Z, F_bc)

    return U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref, nen, nel, IEN, INC,
           E, nu, NQUAD_use, n_mat_ref
end

# ─── Diagnostic solver (exposes C, Z, K for residual checks) ──────────────────

function _cpt_solve_diag(p_ord, exp_level; kwargs...)
    # Re-runs _cpt_solve but also returns C, Z, K, F, IND, Pc for diagnostics
    nsd=3; npd=3; ned=3; npc=2

    conforming   = get(kwargs, :conforming,   false)
    L_x          = get(kwargs, :L_x,          1.0)
    L_y          = get(kwargs, :L_y,          1.0)
    L_z          = get(kwargs, :L_z,          1.0)
    arc_amp      = get(kwargs, :arc_amp,       0.3)
    arc_amp_y    = get(kwargs, :arc_amp_y,     0.3)
    E            = get(kwargs, :E,             1e5)
    nu           = get(kwargs, :nu,            0.3)
    epss         = get(kwargs, :epss,          0.0)
    NQUAD        = get(kwargs, :NQUAD,         p_ord+1)
    NQUAD_mortar = get(kwargs, :NQUAD_mortar,  p_ord+2)
    strategy     = get(kwargs, :strategy,      ElementBasedIntegration())
    formulation  = get(kwargs, :formulation,   TwinMortarFormulation())
    normal_strat = get(kwargs, :normal_strategy, SlaveNormal())
    n_x_lower_base = get(kwargs, :n_x_lower_base, 3)
    n_x_upper_base = get(kwargs, :n_x_upper_base, 2)
    n_y_lower_base = get(kwargs, :n_y_lower_base, 3)
    n_y_upper_base = get(kwargs, :n_y_upper_base, 2)

    B0, P  = cpt_geometry(p_ord; L_x=L_x, L_y=L_y, L_z=L_z,
                           arc_amp=arc_amp, arc_amp_y=arc_amp_y)
    p_mat  = fill(p_ord, npc, npd)
    n_mat  = fill(p_ord+1, npc, npd)
    KV     = generate_knot_vectors(npc, npd, p_mat, n_mat)

    B0_hack = copy(B0);  B0_hack[P[1], 3] .+= 1000.0

    n_x  = n_x_upper_base * 2^exp_level
    n_xl = conforming ? n_x : n_x_lower_base * 2^exp_level
    n_y  = n_y_upper_base * 2^exp_level
    n_yl = conforming ? n_y : n_y_lower_base * 2^exp_level
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
    for i in axes(B_ref,1)
        B_ref[i,3] > 100.0 && (B_ref[i,3] -= 1000.0)
    end

    epss_use = epss > 0.0 ? epss : 1.0e6
    ncp = size(B_ref, 1)

    nel, nnp, nen = patch_metrics(npc, npd, p_mat, n_mat_ref)
    IEN = build_ien(nsd, npd, npc, p_mat, n_mat_ref, nel, nnp, nen)
    INC = [build_inc(n_mat_ref[pc,:]) for pc in 1:npc]

    dBC = [1 4 2 1 2;
           2 5 2 1 2;
           3 1 1 1 0]

    bc_per_dof = dirichlet_bc_control_points(p_mat, n_mat_ref, KV_ref, P_ref, npd, nnp, ned, dBC)
    neq, ID = build_id(bc_per_dof, ned, ncp)
    LM      = build_lm(nen, ned, npc, nel, ID, IEN, P_ref)

    mats = [LinearElastic(E, nu, :three_d), LinearElastic(E, nu, :three_d)]
    Ub0  = zeros(ncp, nsd)
    K, _ = build_stiffness_matrix(npc, nsd, npd, ned, neq,
                                   p_mat, n_mat_ref, KV_ref, P_ref, B_ref, Ub0,
                                   nen, nel, IEN, INC, LM, mats, NQUAD, 1.0)

    F   = zeros(neq)
    IND = Tuple{Int,Float64}[]
    tol_geom = 1e-8 * L_z
    for loc_A in 1:nnp[2]
        cp = P_ref[2][loc_A]; z = B_ref[cp, 3]
        if abs(z - L_z) < tol_geom
            eq = ID[3, cp]; eq != 0 && push!(IND, (eq, -L_z / E))
        end
    end
    unique!(IND)

    K_bc, F_bc = enforce_dirichlet(IND, K, F)

    pairs_full = [InterfacePair(1,6,2,1), InterfacePair(2,1,1,6)]
    Pc = build_interface_cps(pairs_full, p_mat, n_mat_ref, KV_ref, P_ref, npd, nnp, formulation)
    C, Z = build_mortar_coupling(Pc, pairs_full, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
                                  ID, nnp, ned, nsd, npd, neq, NQUAD_mortar, epss_use,
                                  strategy, formulation, normal_strat)

    rows_C, cols_C, vals_C = findnz(C)
    C_bc = sparse(rows_C, cols_C, vals_C, size(C,1), size(C,2))
    _, cols_nz, _ = findnz(C_bc)
    active_lm = sort(unique(cols_nz))
    if length(active_lm) < size(C_bc,2)
        C_bc = C_bc[:, active_lm]; Z = Z[active_lm, active_lm]
    end

    U, lam = solve_mortar(K_bc, C_bc, Z, F_bc)

    return (U=U, lam=lam, K=K_bc, C=C_bc, Z=Z, F=F_bc,
            IND=IND, Pc=Pc,
            ID=ID, B_ref=B_ref, P_ref=P_ref, ncp=ncp, neq=neq,
            p_mat=p_mat, n_mat_ref=n_mat_ref, KV_ref=KV_ref,
            nen=nen, nel=nel, IEN=IEN, INC=INC,
            E=E, nu=nu, L_z=L_z, NQUAD_use=NQUAD)
end

"""
    diagnose_cpt(p_ord, exp_level; kwargs...)

Diagnose patch test failure by checking individual residuals:
  1. ||K * U_exact||          — stiffness residual (should be ~0)
  2. ||C^T * U_exact||        — mortar constraint residual (should be ~0)
  3. ||K * U_FE - F||         — FE system residual (solver accuracy)
  4. ||C^T * U_FE - Z * λ||  — constraint satisfaction
"""
function diagnose_cpt(p_ord::Int, exp_level::Int; kwargs...)
    d = _cpt_solve_diag(p_ord, exp_level; kwargs...)
    E = d.E; nu = d.nu; L_z = d.L_z
    ID = d.ID; B = d.B_ref; ncp = d.ncp; neq = d.neq

    # Build exact displacement vector in equation space
    U_exact = zeros(neq)
    for cp in 1:ncp
        x, y, z = B[cp,1], B[cp,2], B[cp,3]
        ux_ex = nu/E * x
        uy_ex = nu/E * y
        uz_ex = -z/E
        for (i, val) in ((1, ux_ex), (2, uy_ex), (3, uz_ex))
            eq = ID[i, cp]; eq != 0 && (U_exact[eq] = val)
        end
    end

    # Enforce Dirichlet values in U_exact (some DOFs may be prescribed)
    for (eq, val) in d.IND
        U_exact[eq] = val
    end

    rK   = norm(d.K * U_exact)           # stiffness residual
    rC   = norm(d.C' * U_exact)          # mortar constraint residual
    rFE  = norm(d.K * d.U - d.F)        # FE system residual
    rCon = norm(d.C' * d.U - d.Z * d.lam)  # KKT row 2 residual

    @printf "\n=== diagnose_cpt: p=%d, exp=%d ===\n" p_ord exp_level
    @printf "  ||K * U_exact||        = %.3e  (stiffness residual, expect ~0)\n" rK
    @printf "  ||C^T * U_exact||      = %.3e  (mortar constraint,  expect ~0)\n" rC
    @printf "  ||K * U_FE - F||       = %.3e  (FE system residual, expect ~0)\n" rFE
    @printf "  ||C^T*U_FE - Z*λ||    = %.3e  (KKT row 2,         expect ~0)\n" rCon
    @printf "  ||U_FE - U_exact||     = %.3e\n" norm(d.U - U_exact)
    @printf "  ||λ||                  = %.3e\n" norm(d.lam)
    @printf "  neq=%d, nlm=%d, |Pc|=%d\n" neq size(d.Z,1) length(d.Pc)
end

# ─── Patch test check ──────────────────────────────────────────────────────────

"""
    check_cpt(p_ord, exp_level; ...) -> (rms_zz, max_zz)

Run the 3D curved patch test and report stress uniformity.
The test is PASSED if σ_zz ≈ −1 and all other components ≈ 0 throughout both
patches, to near machine precision — regardless of mesh size.

Prints a summary table and returns (rms_zz, max_zz).
"""
function check_cpt(p_ord::Int, exp_level::Int;
                   conforming::Bool = false,
                   dirichlet::Bool  = false,  # false = Neumann (traction loading); true = Dirichlet
                   formulation = TwinMortarFormulation(),
                   strategy    = ElementBasedIntegration(),
                   tol::Float64 = 1e-3,       # RMS < 0.1% is PASS for ε-regularised mortar
                   kwargs...)

    U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
        nen, nel, IEN, INC, E, nu, NQUAD, _ =
        _cpt_solve(p_ord, exp_level; conforming=conforming, dirichlet=dirichlet,
                   formulation=formulation, strategy=strategy, kwargs...)

    rms_zz, max_zz, rms_all, max_all =
        stress_error_cpt(U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref,
                         B_ref, nen, nel, IEN, INC, E, nu, NQUAD)

    fmt_s   = replace(split(string(typeof(formulation)),'.')[end], "Formulation" => "")
    strat_s = replace(split(string(typeof(strategy)),   '.')[end], "Integration"  => "")
    bc_s    = dirichlet ? "Dirichlet" : "Neumann"
    conf_s  = conforming ? "conforming" : "non-conforming"
    # Element counts from n_mat_ref (CPs - p per direction)
    nel1 = (n_mat_ref[1,1]-1, n_mat_ref[1,2]-1, n_mat_ref[1,3]-1)
    nel2 = (n_mat_ref[2,1]-1, n_mat_ref[2,2]-1, n_mat_ref[2,3]-1)

    @printf "\n=== 3D Curved Patch Test: p=%d, %s, %s, %s, %s ===\n" p_ord fmt_s strat_s conf_s bc_s
    @printf "  Mesh:  Patch1 = %d×%d×%d,  Patch2 = %d×%d×%d\n" nel1... nel2...
    @printf "  σ_zz error:    RMS = %.2e,  max = %.2e\n" rms_zz max_zz
    @printf "  All σ error:   RMS = %.2e,  max = %.2e\n" rms_all max_all
    @printf "  Patch test:    %s  (tol = %.0e)\n" (max_zz < tol ? "PASS ✓" : "FAIL ✗") tol

    return rms_zz, max_zz
end

"""
    run_patch_test_cpt(; degrees, exp_levels, formulations, kwargs...)

Run the curved patch test for a matrix of polynomial degrees × formulations.
Prints a summary table of max|σ_zz+1| for each combination.
"""
function run_patch_test_cpt(;
    degrees      = [1, 2, 3],
    exp_level    = 1,
    conforming   = false,
    formulations = [("TM-Elm", TwinMortarFormulation(), ElementBasedIntegration()),
                    ("SP-Elm", SinglePassFormulation(), ElementBasedIntegration())],
    tol          = 1e-3,   # patch test: RMS < 0.1% is PASS
    kwargs...
)
    conf_s = conforming ? "conforming" : "non-conforming"
    @printf "\n=== Curved Patch Test (3D): %s, exp=%d ===\n" conf_s exp_level
    @printf "  Neumann loading: t_z=−1 on top face.  PASS if RMS|σ_zz+1| < %.0e\n\n" tol

    # Header
    @printf "  %-3s |" "p"
    for (lb, _, _) in formulations; @printf "  %-10s |" lb; end; println()
    @printf "  %s-|" "---"
    for _ in formulations; @printf "  %s-|" "----------"; end; println()

    for p in degrees
        @printf "  %-3d |" p
        for (_, form, strat) in formulations
            try
                U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
                    nen, nel, IEN, INC, E, nu, NQUAD, _ =
                    _cpt_solve(p, exp_level; conforming=conforming, dirichlet=false,
                               formulation=form, strategy=strat, kwargs...)
                rms_zz, max_zz, _, _ =
                    stress_error_cpt(U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref,
                                     B_ref, nen, nel, IEN, INC, E, nu, NQUAD)
                pass = rms_zz < tol
                @printf "  rms=%.1e %s |" rms_zz (pass ? "✓" : "✗")
            catch e
                @printf "  %-12s |" "ERROR"
            end
        end
        println()
    end
end

# ─── VTK export ────────────────────────────────────────────────────────────────

"""
    write_vtk_cpt(prefix, U, ID, npc, nsd, npd, p_mat, n_mat, KV, P, B,
                  nen_vec, IEN, INC, E, nu; n_vis=4)

Write one VTK STRUCTURED_GRID file per patch.  Fields written per point:
  displacement (vector), stress_xx/yy/zz/xy/yz/zx (scalars),
  von_mises (scalar), stress_error_zz = |σ_zz − (−1)| (scalar).
The last field is the pointwise patch-test error for the CPT exact solution
σ_zz = −1.

`n_vis` controls sampling density: `n_vis` equally spaced points are placed
within each knot span in each parametric direction.
"""
function write_vtk_cpt(
    prefix::String,
    U::Vector{Float64},
    ID::Matrix{Int},
    npc::Int, nsd::Int, npd::Int,
    p_mat::Matrix{Int}, n_mat::Matrix{Int},
    KV::Vector{<:AbstractVector{<:AbstractVector{Float64}}},
    P::Vector{Vector{Int}},
    B::Matrix{Float64},
    nen_vec::Vector{Int},
    IEN::Vector{Matrix{Int}},
    INC::Vector{<:AbstractVector{<:AbstractVector{Int}}},
    E::Float64, nu::Float64;
    n_vis::Int = 4
)
    ncp = size(B, 1)

    Ub = zeros(ncp, nsd)
    for A in 1:ncp, i in 1:nsd
        eq = ID[i, A]; eq != 0 && (Ub[A, i] = U[eq])
    end

    mat = LinearElastic(E, nu, :three_d)
    D   = elastic_constants(mat, nsd)

    for pc in 1:npc
        pv  = p_mat[pc, :]
        nv  = n_mat[pc, :]
        kv  = KV[pc]
        Ppc = P[pc]
        ien = IEN[pc]
        inc = INC[pc]
        nen = nen_vec[pc]

        s1 = _kv_sample(collect(kv[1]), n_vis)
        s2 = _kv_sample(collect(kv[2]), n_vis)
        s3 = _kv_sample(collect(kv[3]), n_vis)
        nd1, nd2, nd3 = length(s1), length(s2), length(s3)
        n_pts = nd1 * nd2 * nd3

        pts   = zeros(3, n_pts)
        disp  = zeros(3, n_pts)
        sxx   = zeros(n_pts); syy = zeros(n_pts); szz = zeros(n_pts)
        sxy   = zeros(n_pts); syz = zeros(n_pts); szx = zeros(n_pts)
        svm   = zeros(n_pts)
        serr  = zeros(n_pts)   # pointwise |σ_zz − (−1)|

        n_elem = [nv[d] - pv[d] for d in 1:npd]

        idx = 0
        for xi3 in s3, xi2 in s2, xi1 in s1
            idx += 1
            Xi = [xi1, xi2, xi3]

            n0 = [find_span(nv[d]-1, pv[d], Float64(Xi[d]), collect(kv[d])) for d in 1:npd]
            e  = [n0[d] - pv[d] for d in 1:npd]
            el = (e[3]-1)*n_elem[1]*n_elem[2] + (e[2]-1)*n_elem[1] + e[1]

            xi_tilde = zeros(npd)
            for d in 1:npd
                kv_d = collect(kv[d])
                a, b = kv_d[n0[d]], kv_d[n0[d]+1]
                xi_tilde[d] = (b > a) ? clamp((2*Xi[d] - a - b) / (b - a), -1.0, 1.0) : 0.0
            end

            R, dR_dx, _, detJ, _ = shape_function(
                pv, nv, kv, B, Ppc, xi_tilde, nen, nsd, npd, el, n0, ien, inc)

            el_nodes = Ppc[ien[el, :]]
            Xe = B[el_nodes, 1:nsd]
            X  = Xe' * R
            pts[1:nsd, idx] = X

            Ue = Ub[el_nodes, :]
            u  = Ue' * R
            disp[1:nsd, idx] = u

            if detJ > 0.0
                dN_dX = Matrix(dR_dx')
                B_mat = strain_displacement_matrix(nsd, nen, dN_dX)
                u_vec = Vector{Float64}(undef, nsd * nen)
                for a in 1:nen, d in 1:nsd; u_vec[(a-1)*nsd + d] = Ue[a, d]; end
                sig = D * (B_mat * u_vec)  # [σxx, σyy, σzz, τxy, τyz, τzx]
                sxx[idx] = sig[1]; syy[idx] = sig[2]; szz[idx] = sig[3]
                sxy[idx] = sig[4]; syz[idx] = sig[5]; szx[idx] = sig[6]
                svm[idx] = sqrt(max(0.0,
                    0.5*((sig[1]-sig[2])^2 + (sig[2]-sig[3])^2 + (sig[3]-sig[1])^2)
                    + 3*(sig[4]^2 + sig[5]^2 + sig[6]^2)))
                serr[idx] = abs(sig[3] + 1.0)   # |σ_zz − (−1)|
            end
        end

        fname = "$(prefix)_patch$(pc).vtk"
        open(fname, "w") do f
            println(f, "# vtk DataFile Version 2.0")
            println(f, "Curved patch test patch $pc")
            println(f, "ASCII")
            println(f, "DATASET STRUCTURED_GRID")
            println(f, "DIMENSIONS $nd1 $nd2 $nd3")
            println(f, "POINTS $n_pts float")
            for i in 1:n_pts
                @printf f "%e\t%e\t%e\n" pts[1,i] pts[2,i] pts[3,i]
            end
            println(f, "POINT_DATA $n_pts")
            println(f, "VECTORS displacement float")
            for i in 1:n_pts
                @printf f "%e\t%e\t%e\n" disp[1,i] disp[2,i] disp[3,i]
            end
            for (name, arr) in [("stress_xx",    sxx), ("stress_yy",    syy),
                                 ("stress_zz",    szz), ("stress_xy",    sxy),
                                 ("stress_yz",    syz), ("stress_zx",    szx),
                                 ("von_mises",    svm), ("stress_error_zz", serr)]
                println(f, "SCALARS $name float 1")
                println(f, "LOOKUP_TABLE default")
                for i in 1:n_pts; @printf f "%e\n" arr[i]; end
            end
        end
        @printf "  Wrote %s  (%d×%d×%d grid)\n" fname nd1 nd2 nd3
    end
end

"""
    energy_norm_error_cpt(U, ID, npc, nsd, npd, p, n, KV, P, B,
                          nen, nel, IEN, INC, E, nu, NQUAD) -> (rel_energy_err,)

Compute the relative energy-norm error ||u_h - u_ex||_E / ||u_ex||_E
for the curved patch test (exact: σ_zz=-1, rest=0).
"""
function energy_norm_error_cpt(
    U::Vector{Float64}, ID::Matrix{Int},
    npc::Int, nsd::Int, npd::Int,
    p::Matrix{Int}, n::Matrix{Int},
    KV, P, B::Matrix{Float64},
    nen::Vector{Int}, nel::Vector{Int},
    IEN, INC, E_mod::Float64, nu::Float64, NQUAD::Int
)
    mat = LinearElastic(E_mod, nu, :three_d)
    D   = elastic_constants(mat, nsd)

    # Exact strain: σ_exact = [0,0,-1,0,0,0]
    # ε_xx = -ν·σ_zz/E = ν/E, ε_yy = ν/E, ε_zz = σ_zz/E = -1/E
    eps_exact = [nu/E_mod, nu/E_mod, -1.0/E_mod, 0.0, 0.0, 0.0]

    ncp = size(B,1)
    Ub  = zeros(ncp, nsd)
    for A in 1:ncp, i in 1:nsd
        eq = ID[i,A]; eq != 0 && (Ub[A,i] = U[eq])
    end

    err2 = 0.0; ref2 = 0.0
    GPW = gauss_product(NQUAD, npd)

    for pc in 1:npc
        ien = IEN[pc]; inc = INC[pc]
        for el in 1:nel[pc]
            n0 = inc[ien[el,1]]
            for (gp, gw) in GPW
                R, dR_dx, _, detJ, _ = shape_function(
                    p[pc,:], n[pc,:], KV[pc], B, P[pc], gp,
                    nen[pc], nsd, npd, el, n0, ien, inc)
                detJ <= 0 && continue
                gwJ = gw * detJ

                el_nodes = P[pc][ien[el,:]]
                Ue = Ub[el_nodes, :]
                u_vec = vec(Ue')
                dN_dX = Matrix(dR_dx')
                Bmat  = strain_displacement_matrix(nsd, nen[pc], dN_dX)
                eps_h = Bmat * u_vec

                de = eps_h - eps_exact
                err2 += dot(de, D * de) * gwJ
                ref2 += dot(eps_exact, D * eps_exact) * gwJ
            end
        end
    end

    return sqrt(err2 / ref2)
end

"""
    run_convergence_cpt(; p_range=1:4, exp_range_per_p=Dict(), epss_per_p=Dict(),
                          n_x_lower_base=2, kwargs...)

Run the curved patch test convergence study and print results.
Returns a Dict(p => (hs, errs)) for plotting.
"""
function run_convergence_cpt(;
        p_range = 1:4,
        exp_range_per_p = Dict(1 => 0:4, 2 => 0:3, 3 => 0:3, 4 => 0:2),
        epss_per_p = Dict(1 => 1e3, 2 => 1e2, 3 => 1e2, 4 => 1e2),
        n_x_lower_base = 2,
        kwargs...)

    results = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}}}()

    for p_ord in p_range
        exp_range = get(exp_range_per_p, p_ord, 0:3)
        epss = get(epss_per_p, p_ord, 1e3)
        hs   = Float64[]
        errs = Float64[]
        @printf "\n=== p = %d, ε = %.0e ===\n" p_ord epss
        for e in exp_range
            h = 1.0 / max(1, 2^e)
            U, rest... = _cpt_solve(p_ord, e; epss=epss,
                                     n_x_lower_base=n_x_lower_base, kwargs...)
            en = energy_norm_error_cpt(U, rest[1:end-1]...)
            push!(hs, h); push!(errs, en)
            @printf "  exp=%d  h=%.4f  ||e||_E/||u||_E = %.4e\n" e h en
        end

        # rates
        @printf "  rates:"
        for i in 2:length(hs)
            r = log(errs[i-1]/errs[i]) / log(hs[i-1]/hs[i])
            @printf " %.2f" r
        end
        println()
        results[p_ord] = (hs, errs)
    end

    return results
end

"""
    solve_cpt_vtk(p_ord, exp_level; vtk_prefix="cpt", n_vis=4, kwargs...)

Run the curved patch test and write VTK output for both patches.
All `_cpt_solve` keyword arguments are forwarded (epss, arc_amp, formulation, etc.).
Files written: `<vtk_prefix>_patch1.vtk`, `<vtk_prefix>_patch2.vtk`.
"""
function solve_cpt_vtk(p_ord::Int, exp_level::Int;
                       vtk_prefix::String = "cpt",
                       n_vis::Int = 4,
                       kwargs...)
    U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
        nen, nel, IEN, INC, E, nu, NQUAD, _ =
        _cpt_solve(p_ord, exp_level; kwargs...)

    rms_zz, max_zz, _, _ = stress_error_cpt(
        U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref,
        B_ref, nen, nel, IEN, INC, E, nu, NQUAD)

    @printf "\np=%d  exp=%d  RMS|σ_zz+1|=%.3e  MAX|σ_zz+1|=%.3e\n" p_ord exp_level rms_zz max_zz
    write_vtk_cpt(vtk_prefix, U, ID, npc, nsd, npd,
                  p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
                  nen, IEN, INC, E, nu; n_vis=n_vis)
end

# ─── All 4 methods patch test (§7.4) ─────────────────────────────────────────

"""
    run_patch_test_cpt_4methods(; degrees, exp_level, kwargs...)

Run 3D curved patch test for all 4 methods: SPMS, SPME, DPM, TM.
Reports RMS |σ_zz + 1| for each (method, p) combination.
"""
function run_patch_test_cpt_4methods(;
    degrees    = [1, 2],
    exp_level  = 1,
    conforming = false,
    epss       = 1e6,
    tol        = 1e-3,
    kwargs...
)
    formulations = [
        ("SPMS",  SinglePassFormulation(),  SegmentBasedIntegration()),
        ("SPME",  SinglePassFormulation(),  ElementBasedIntegration()),
        ("DPM",   DualPassFormulation(),    SegmentBasedIntegration()),
        ("TM",    TwinMortarFormulation(),  ElementBasedIntegration()),
    ]

    conf_s = conforming ? "conforming" : "non-conforming"
    @printf "\n=== Curved Patch Test (3D): %s, exp=%d, ε=%.0e ===\n" conf_s exp_level epss
    @printf "  PASS if RMS|σ_zz+1| < %.0e\n\n" tol

    @printf "  %-3s |" "p"
    for (lb, _, _) in formulations; @printf "  %-14s |" lb; end; println()
    @printf "  %s-|" "---"
    for _ in formulations; @printf "  %s-|" "--------------"; end; println()

    for p in degrees
        @printf "  %-3d |" p
        for (label, form, strat) in formulations
            try
                U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
                    nen, nel, IEN, INC, E, nu, NQUAD, _ =
                    _cpt_solve(p, exp_level; conforming=conforming, epss=epss,
                               formulation=form, strategy=strat, kwargs...)
                rms_zz, _, _, _ =
                    stress_error_cpt(U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref,
                                     B_ref, nen, nel, IEN, INC, E, nu, NQUAD)
                pass = rms_zz < tol
                @printf "  rms=%.2e %s |" rms_zz (pass ? "✓" : "✗")
            catch e
                @printf "  %-16s |" "ERROR"
            end
        end
        println()
    end
end

# ─── ε-sweep for curved patch test ───────────────────────────────────────────

"""
    run_cpt_eps_sweep(; degrees, exp_level, eps_range, kwargs...)

Sweep ε for TM on the 3D curved patch test.  Shows error declining with ε
(conditional patch test pass).
"""
function run_cpt_eps_sweep(;
    degrees    = [1, 2, 3, 4],
    exp_level  = 1,
    eps_range  = 10.0 .^ (0:8),
    conforming = false,
    kwargs...
)
    conf_s = conforming ? "conforming" : "non-conforming"
    @printf "\n=== CPT ε-sweep (TM, %s, exp=%d) ===\n\n" conf_s exp_level

    @printf "%-12s" "ε"
    for p in degrees; @printf "  %12s" "p=$p"; end
    println()
    @printf "%s\n" "─"^(12 + 14*length(degrees))

    for eps in eps_range
        @printf "%-12.1e" eps
        for p in degrees
            try
                U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
                    nen, nel, IEN, INC, E, nu, NQUAD, _ =
                    _cpt_solve(p, exp_level; conforming=conforming, epss=eps,
                               formulation=TwinMortarFormulation(),
                               strategy=ElementBasedIntegration(), kwargs...)
                rms_zz, _, _, _ =
                    stress_error_cpt(U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref,
                                     B_ref, nen, nel, IEN, INC, E, nu, NQUAD)
                @printf "  %12.3e" rms_zz
            catch e
                @printf "  %12s" "ERROR"
            end
        end
        println()
    end
end
