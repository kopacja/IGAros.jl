# run_flat_patchtest_3d.jl — 3D Farah-style flat patch test (full factorial)
#
# Farah et al. (2015) Fig. 10:
#   Lower block: 6×6×2.5, 5×5×3 elements (6×6×4 nodes for p=1)
#   Upper block: 3×3×2.5, 3×3×3 elements (4×4×4 nodes for p=1), centered
#   Interface: P1 F6 (ζ=max) ↔ P2 F1 (ζ=min)
#   Loading: σ_zz = −1 on top of both blocks
#   BCs: uz=0 on P1 bottom, ux=0 on P1 x=0, uy=0 on P1 y=0
#
# Segment-based: 3 Gauss points per triangular cell for p=1 (Farah convention)
#   tri_gauss_rule(NQUAD) with NQUAD=p+2: p=1→3pts, p=2→7pts, p≥3→7pts
#
# Saves: factorial.csv, eps_sweep.csv, nquad_sweep.csv, moments.csv, meta.toml

import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using IGAros, LinearAlgebra, SparseArrays, Printf, Dates

function open_kv(n_elem::Int, p::Int)
    n_cp = n_elem + p
    kv = zeros(n_cp + p + 1)
    for i in 1:n_elem-1; kv[p+1+i] = i/n_elem; end
    kv[end-p:end] .= 1.0
    return kv
end

"""
    _subtract_overlap_load!(F, p, n, KV, P, B, nnp, nsd, npd, ned, ID,
                            σ_app, NQUAD_tri, xy_min, xy_max, z_face)

Subtract the traction σ_zz = −σ_app integrated over the overlap region
[xy_min[1], xy_max[1]] × [xy_min[2], xy_max[2]] on facet 6 (z = z_face)
from the force vector F.

For each face element, clips its physical quad against the overlap rectangle
using Sutherland-Hodgman, triangulates the intersection, and integrates
the NURBS-weighted traction over each triangle.  Physical-to-parametric
mapping assumes flat geometry (bilinear inversion).
"""
function _subtract_overlap_load!(
    F::Vector{Float64},
    p::AbstractVector{Int}, n::AbstractVector{Int},
    KV::AbstractVector{<:AbstractVector{Float64}},
    P::AbstractVector{Int}, B::AbstractMatrix{Float64},
    nnp::Int, nsd::Int, npd::Int, ned::Int,
    ID::Matrix{Int}, σ_app::Float64, NQUAD_tri::Int,
    xy_min::Vector{Float64}, xy_max::Vector{Float64}, z_face::Float64
)
    ps, ns, KVs, Ps, nsn, nsen, nsel, norm_sign, _, _, _ =
        get_segment_patch(p, n, KV, P, npd, nnp, 6)
    ien_s = build_ien(nsd, npd-1, 1, reshape(ps,1,:), reshape(ns,1,:),
                      [nsel], [nsn], [nsen])
    ien = ien_s[1]
    inc = build_inc(ns)

    clip = [
        [xy_min[1], xy_min[2], z_face],
        [xy_max[1], xy_min[2], z_face],
        [xy_max[1], xy_max[2], z_face],
        [xy_min[1], xy_max[2], z_face],
    ]
    n_up = [0.0, 0.0, 1.0]

    tri_pts, tri_wts = tri_gauss_rule(NQUAD_tri)
    nqp = length(tri_wts)

    for el in 1:nsel
        anchor = ien[el, 1]
        n0     = inc[anchor]

        # Element parametric bounds
        kv1, kv2 = KVs[1], KVs[2]
        ξ_lo = kv1[n0[1]]; ξ_hi = kv1[n0[1]+1]
        η_lo = kv2[n0[2]]; η_hi = kv2[n0[2]+1]

        # Physical corners of element (evaluate at parametric corners via gp in [-1,1])
        corners_gp = [[-1.0,-1.0], [1.0,-1.0], [1.0,1.0], [-1.0,1.0]]
        corners_phys = Vector{Float64}[]
        for gp in corners_gp
            R, _, _, _, _ = shape_function(ps, ns, KVs, B, Ps, gp,
                                           nsen, nsd, npd-1, el, n0, ien, inc)
            x = zeros(nsd)
            for a in 1:nsen; x .+= R[a] .* B[Ps[ien[el,a]], 1:nsd]; end
            push!(corners_phys, x)
        end

        # Clip element quad against overlap rectangle
        poly = sutherland_hodgman_clip(corners_phys, clip, n_up)
        length(poly) < 3 && continue

        # x-range and y-range of element (for flat geometry inverse mapping)
        x_lo = corners_phys[1][1]; x_hi = corners_phys[2][1]
        y_lo = corners_phys[1][2]; y_hi = corners_phys[4][2]
        dx = x_hi - x_lo; dy = y_hi - y_lo

        for (v1, v2, v3) in triangulate_polygon(poly)
            e1 = v2 .- v1; e2 = v3 .- v1
            area2 = norm(cross(e1, e2))  # 2 × triangle area
            area2 < 1e-30 && continue

            Fs = zeros(ned, nsen)
            for q in 1:nqp
                L1 = tri_pts[1, q]; L2 = tri_pts[2, q]; L0 = 1 - L1 - L2
                x_q = L0 .* v1 .+ L1 .* v2 .+ L2 .* v3

                # Inverse map: physical → parent element gp ∈ [-1,1]²
                gp = [2*(x_q[1] - x_lo)/dx - 1,
                      2*(x_q[2] - y_lo)/dy - 1]

                R, _, _, _, _ = shape_function(ps, ns, KVs, B, Ps, gp,
                                               nsen, nsd, npd-1, el, n0, ien, inc)

                # Traction on F6: σ·n = [0,0,−σ_app]·[0,0,norm_sign] → Fp_z = −σ_app*norm_sign
                gwJ = tri_wts[q] * area2
                for a in 1:nsen
                    Fs[3, a] += (-σ_app * norm_sign) * R[a] * gwJ
                end
            end

            # Scatter (subtract from F)
            for a in 1:nsen
                cp = Ps[ien[el, a]]
                for i in 1:ned
                    eq = ID[i, cp]; eq != 0 && (F[eq] -= Fs[i, a])
                end
            end
        end
    end
    return F
end

"""
    _subtract_overlap_load_elem!(F, ...)

Element-based variant of `_subtract_overlap_load!`.  Uses standard tensor-
product Gauss quadrature on each face element and skips GPs whose physical
position falls outside the overlap rectangle [xy_min, xy_max].  This matches
the element-based mortar assembly so that errors in the overlap area
approximation cancel between the rim subtraction and the mortar reaction.
"""
function _subtract_overlap_load_elem!(
    F::Vector{Float64},
    p::AbstractVector{Int}, n::AbstractVector{Int},
    KV::AbstractVector{<:AbstractVector{Float64}},
    P::AbstractVector{Int}, B::AbstractMatrix{Float64},
    nnp::Int, nsd::Int, npd::Int, ned::Int,
    ID::Matrix{Int}, σ_app::Float64, NQUAD::Int,
    xy_min::Vector{Float64}, xy_max::Vector{Float64}, z_face::Float64
)
    ps, ns, KVs, Ps, nsn, nsen, nsel, norm_sign, _, _, _ =
        get_segment_patch(p, n, KV, P, npd, nnp, 6)
    ien_s = build_ien(nsd, npd-1, 1, reshape(ps,1,:), reshape(ns,1,:),
                      [nsel], [nsn], [nsen])
    ien = ien_s[1]
    inc = build_inc(ns)

    GPW = gauss_product(NQUAD, npd - 1)
    tol = 1e-10

    for el in 1:nsel
        anchor = ien[el, 1]
        n0     = inc[anchor]

        Fs = zeros(ned, nsen)
        for (gp, gw) in GPW
            R, _, _, detJ, _ = shape_function(
                ps, ns, KVs, B, Ps, gp, nsen, nsd, npd-1, el, n0, ien, inc)
            detJ <= 0 && continue
            # Physical position of this GP
            x_phys = zeros(nsd)
            for a in 1:nsen
                x_phys .+= R[a] .* B[Ps[ien[el,a]], 1:nsd]
            end
            # Skip GPs outside the overlap rectangle
            (x_phys[1] < xy_min[1] - tol || x_phys[1] > xy_max[1] + tol ||
             x_phys[2] < xy_min[2] - tol || x_phys[2] > xy_max[2] + tol) && continue

            gwJ = gw * detJ
            for a in 1:nsen
                Fs[3, a] += (-σ_app * norm_sign) * R[a] * gwJ
            end
        end

        for a in 1:nsen
            cp = Ps[ien[el, a]]
            for i in 1:ned
                eq = ID[i, cp]; eq != 0 && (F[eq] -= Fs[i, a])
            end
        end
    end
    return F
end

function solve_farah3d(p_ord; n_lower=5, n_upper=3, n_z=3,
    E=100.0, nu=0.0, epss=1e8,
    L_lower=10.0, H_lower=4.0, L_upper=5.0, H_upper=4.0,
    NQUAD=p_ord+1, NQUAD_mortar=p_ord+2,
    formulation::FormulationStrategy=TwinMortarFormulation(),
    strategy::IntegrationStrategy=ElementBasedIntegration(),
    slave_first::Symbol=:lower)

    nsd=3; npd=3; ned=3; npc=2
    δ = (L_lower - L_upper) / 2

    nc = (n, p) -> n + p
    nc_x1=nc(n_lower,p_ord); nc_y1=nc(n_lower,p_ord); nc_z1=nc(n_z,p_ord)
    nc_x2=nc(n_upper,p_ord); nc_y2=nc(n_upper,p_ord); nc_z2=nc(n_z,p_ord)

    B1 = zeros(nc_x1*nc_y1*nc_z1, 4)
    for k in 1:nc_z1, j in 1:nc_y1, i in 1:nc_x1
        A = (k-1)*nc_x1*nc_y1 + (j-1)*nc_x1 + i
        B1[A,:] = [(i-1)/(nc_x1-1)*L_lower, (j-1)/(nc_y1-1)*L_lower,
                   (k-1)/(nc_z1-1)*H_lower, 1.0]
    end

    B2 = zeros(nc_x2*nc_y2*nc_z2, 4)
    for k in 1:nc_z2, j in 1:nc_y2, i in 1:nc_x2
        A = (k-1)*nc_x2*nc_y2 + (j-1)*nc_x2 + i
        B2[A,:] = [δ+(i-1)/(nc_x2-1)*L_upper, δ+(j-1)/(nc_y2-1)*L_upper,
                   H_lower+(k-1)/(nc_z2-1)*H_upper, 1.0]
    end

    ncp1=size(B1,1); ncp2=size(B2,1); ncp=ncp1+ncp2
    B = vcat(B1, B2)
    P = [collect(1:ncp1), collect(ncp1+1:ncp1+ncp2)]
    p_mat = fill(p_ord, npc, npd)
    n_mat = [nc_x1 nc_y1 nc_z1; nc_x2 nc_y2 nc_z2]
    KV = [[open_kv(n_lower,p_ord), open_kv(n_lower,p_ord), open_kv(n_z,p_ord)],
          [open_kv(n_upper,p_ord), open_kv(n_upper,p_ord), open_kv(n_z,p_ord)]]

    nel, nnp, nen = patch_metrics(npc, npd, p_mat, n_mat)
    IEN = build_ien(nsd, npd, npc, p_mat, n_mat, nel, nnp, nen)
    INC = [build_inc(n_mat[pc,:]) for pc in 1:npc]

    # BCs: Farah-style patch test.
    #   Dirichlet: P1 bottom (z=0) fully fixed (ux=uy=uz=0).
    #   Rigid-body pins on P2 top face (z=H1+H2) corners:
    #     corner 1 (δ,δ,H1+H2):         ux=0, uy=0
    #     corner 2 (δ+Lu,δ,H1+H2):      ux=0
    #     corner 3 (δ,δ+Lu,H1+H2):      uy=0
    #     corner 4 (δ+Lu,δ+Lu,H1+H2):   free
    #   Neumann: σ_zz = −σ_app on P2 top face + P1 F6 rim (outside overlap).
    tol = 1e-10
    σ_app = 0.5
    bc = [Int[] for _ in 1:ned]

    # P1 bottom face (z=0): fix all DOFs
    for A in P[1]
        if B[A,3] < tol
            push!(bc[1], A); push!(bc[2], A); push!(bc[3], A)
        end
    end

    # P2 top face corners: rigid-body pins
    z_top = H_lower + H_upper
    corners = [  # (x, y, fix_ux, fix_uy)
        (δ,          δ,          true, true),
        (δ+L_upper,  δ,          true, false),
        (δ,          δ+L_upper,  false, true),
    ]
    for (cx, cy, fix_ux, fix_uy) in corners
        for A in P[2]
            x,y,z = B[A,1:3]
            if abs(x-cx)<tol && abs(y-cy)<tol && abs(z-z_top)<tol
                fix_ux && push!(bc[1], A)
                fix_uy && push!(bc[2], A)
            end
        end
    end

    for d in 1:ned; unique!(bc[d]); end
    neq, ID = build_id(bc, ned, ncp)
    LM = build_lm(nen, ned, npc, nel, ID, IEN, P)

    mats = [LinearElastic(E, nu, :three_d) for _ in 1:npc]
    t0 = time()
    K, _ = build_stiffness_matrix(npc, nsd, npd, ned, neq, p_mat, n_mat, KV, P, B,
        zeros(ncp,nsd), nen, nel, IEN, INC, LM, mats, NQUAD, 1.0)

    # Neumann: σ_zz = −σ_app on P2 top face (F6) + P1 F6 rim (outside overlap).
    # The overlap area of P1 F6 is mortar-coupled — traction there comes from λ.
    # Rim loading uses segment-based integration: for each P1 F6 element, clip
    # against the overlap footprint and integrate pressure only on the complement.
    # Strategy: F_rim = F_full_P1_F6 − F_overlap_P1_F6
    stress_fn = (x,y,z) -> begin; σ=zeros(3,3); σ[3,3]=-σ_app; σ; end
    F = zeros(neq)
    # The full-face Neumann integrals are smooth (constant pressure) and can
    # be integrated exactly with NQUAD = p+1 GPs (volume quadrature setting).
    # P2 top face: full traction
    F = segment_load(n_mat[2,:], p_mat[2,:], KV[2], P[2], B,
                     nnp[2], nen[2], nsd, npd, ned, Int[], 6, ID, F, stress_fn, 1.0, NQUAD)
    # P1 F6: full face (overlap subtracted in the next call)
    F = segment_load(n_mat[1,:], p_mat[1,:], KV[1], P[1], B,
                     nnp[1], nen[1], nsd, npd, ned, Int[], 6, ID, F, stress_fn, 1.0, NQUAD)
    # Subtract overlap contribution.  For element-based mortar, use the SAME
    # element-based skip-quadrature so that the rim subtraction and the mortar
    # reaction approximate the overlap region IDENTICALLY — errors cancel and
    # the sum F_rim + Cλ equals the constant total pressure at every CP.
    # For segment-based mortar, use exact polygon clipping (also exact).
    if strategy isa SegmentBasedIntegration
        F = _subtract_overlap_load!(F, p_mat[1,:], n_mat[1,:], KV[1], P[1], B,
            nnp[1], nsd, npd, ned, ID, σ_app, max(NQUAD_mortar, 7),
            [δ, δ], [δ+L_upper, δ+L_upper], H_lower)
    else
        F = _subtract_overlap_load_elem!(F, p_mat[1,:], n_mat[1,:], KV[1], P[1], B,
            nnp[1], nsd, npd, ned, ID, σ_app, NQUAD_mortar,
            [δ, δ], [δ+L_upper, δ+L_upper], H_lower)
    end
    IND = Tuple{Int,Float64}[]
    K_bc, F_bc = enforce_dirichlet(IND, K, F)

    # slave_first: which patch is slave in the first (or only) pass
    #   :lower → P1 F6 slave, P2 F1 master (Farah Case 2: slave bigger)
    #   :upper → P2 F1 slave, P1 F6 master (Farah Case 1: slave smaller, all covered)
    if slave_first == :upper
        pair_fwd = InterfacePair(2,1,1,6)
        pair_rev = InterfacePair(1,6,2,1)
    else
        pair_fwd = InterfacePair(1,6,2,1)
        pair_rev = InterfacePair(2,1,1,6)
    end
    if formulation isa SinglePassFormulation
        pairs = [pair_fwd]
    else
        pairs = [pair_fwd, pair_rev]
    end
    Pc = build_interface_cps(pairs, p_mat, n_mat, KV, P, npd, nnp, formulation)
    C, Z = build_mortar_coupling(Pc, pairs, p_mat, n_mat, KV, P, B,
        ID, nnp, ned, nsd, npd, neq, NQUAD_mortar, epss, strategy, formulation)

    # Mortar moments (scalar mass matrices — element-based only)
    d0_p1=NaN; d1_p1=NaN; d2_p1=NaN; d0_p2=NaN; d1_p2=NaN; d2_p2=NaN
    if strategy isa ElementBasedIntegration
        try
            pair1 = pair_fwd
            D, M, s_cps, m_cps = build_mortar_mass_matrices(
                pair1, p_mat, n_mat, KV, P, B, nnp, nsd, npd, NQUAD_mortar, strategy)
            Dd = Matrix(D); Md = Matrix(M)
            d0_p1 = sum(Dd); d0_p2 = sum(Md)
            x_s = [B[cp,1] for cp in s_cps]; z_s = [B[cp,3] for cp in s_cps]
            x_m = [B[cp,1] for cp in m_cps]; z_m = [B[cp,3] for cp in m_cps]
            d1_p1 = sum(Dd * x_s); d2_p1 = sum(Dd * z_s)
            d1_p2 = sum(Md * x_m); d2_p2 = sum(Md * z_m)
        catch; end
    end

    # Constraint RHS correction for non-homogeneous Dirichlet DOFs.
    # KKT constraint row: C'*U - Z*λ = 0.  When Dirichlet DOFs are in C,
    # their prescribed values must be moved to the RHS before zeroing C rows:
    #   g = -C_fixed' * u_prescribed
    nlm2 = size(C, 2)
    g = zeros(nlm2)
    fixed_eqs = Set{Int}()
    for (eq, val) in IND
        push!(fixed_eqs, eq)
        g .-= C[eq, :] .* val
    end
    if !isempty(fixed_eqs)
        rows_C, cols_C, vals_C = findnz(C)
        keep = [i for i in eachindex(rows_C) if !(rows_C[i] in fixed_eqs)]
        C = sparse(rows_C[keep], cols_C[keep], vals_C[keep], size(C,1), size(C,2))
    end

    # Remove multiplier DOFs with negligible C column AND Z row/col.
    # Must happen AFTER C-row zeroing (Dirichlet may create new zero columns).
    # Use tolerance: for p≥2, rim CPs whose basis barely grazes the overlap
    # produce floating-point dust (~1e-48) that is not exactly zero.
    drop_tol = 1e-12
    active = trues(nlm2)
    for j in 1:nlm2
        c_norm = norm(C[:, j])
        z_norm = norm(Z[:, j]) + norm(Z[j, :])
        if c_norm < drop_tol && z_norm < drop_tol
            active[j] = false
        end
    end
    if !all(active)
        idx = findall(active)
        C = C[:, idx]
        Z = Z[idx, idx]
        g = g[idx]
    end
    # Also drop near-zero entries from C and Z (floating-point dust from
    # rim CPs with p≥2 whose basis barely grazes the overlap)
    droptol!(C, drop_tol)
    droptol!(Z, drop_tol)

    # Condition numbers (only for small systems)
    kappa = NaN; kappa_kkt = NaN
    n_total = neq + size(C, 2)
    if n_total < 5000
        try
            Kd = Matrix(K_bc); Cd = Matrix(C); Zd = Matrix(Z)
            kappa = cond(Kd)          # stiffness only
            A_kkt = [Kd Cd; Cd' Zd]
            kappa_kkt = cond(A_kkt)   # full KKT
        catch; end
    end

    U, lam = solve_mortar(K_bc, C, Z, F_bc; g=g)
    wall = time() - t0

    # Displacement error: exact (ν=0) uz = −σ_app·z/E, ux=uy=0
    # (a) Pointwise max at control points
    max_err = 0.0
    for pc in 1:npc, A in P[pc]
        x,y,z = B[A,1:3]
        u_ex = [nu/E*σ_app*x, nu/E*σ_app*y, -σ_app*z/E]
        for d in 1:ned
            eq = ID[d,A]; u_h = eq==0 ? 0.0 : U[eq]
            max_err = max(max_err, abs(u_h - u_ex[d]))
        end
    end
    # (b) Continuous L² norm over the volume
    ncp = size(B, 1)
    Ub = zeros(ncp, nsd)
    for A in 1:ncp, i in 1:nsd
        eq = ID[i, A]; eq != 0 && (Ub[A, i] = U[eq])
    end
    l2_err2 = 0.0; l2_ref2 = 0.0
    NQUAD_vol = max(p_ord + 1, 2)
    GPW_vol = gauss_product(NQUAD_vol, npd)
    for pc in 1:npc
        ien_pc = IEN[pc]; inc_pc = INC[pc]
        for el in 1:nel[pc]
            anchor = ien_pc[el, 1]; n0 = inc_pc[anchor]
            for (gp, gw) in GPW_vol
                R, _, _, detJ, _ = shape_function(
                    p_mat[pc,:], n_mat[pc,:], KV[pc], B, P[pc], gp,
                    nen[pc], nsd, npd, el, n0, ien_pc, inc_pc)
                detJ <= 0 && continue
                gwJ = gw * detJ
                u_h  = Ub[P[pc][ien_pc[el,:]], 1:nsd]' * R
                X    = B[P[pc][ien_pc[el,:]], :]' * R
                u_ex = [nu/E*σ_app*X[1], nu/E*σ_app*X[2], -σ_app*X[3]/E]
                diff = u_h[1:nsd] - u_ex
                l2_err2 += dot(diff, diff) * gwJ
                l2_ref2 += dot(u_ex, u_ex) * gwJ
            end
        end
    end
    l2_disp = sqrt(l2_err2)

    # Lagrange multiplier error (overlap CPs only, rim excluded).
    # SP: single multiplier field, sign depends on slave normal via dir_vecs.
    #   slave_first=:upper (P2 slave, normal=-z) → λ_z = -σ_app
    #   slave_first=:lower (P1 slave, normal=+z) → λ_z = -σ_app (same!)
    #   because the SP kernel multiplies by norm_sign, so λ always = -σ_app.
    # TM/DP: two multiplier fields, assembly is symmetric (slave_first only
    #   affects pair ordering, not the result). P1 F6 normal = +z → λ_z(P1) = +σ_app
    #   (interface traction aligns with outward normal); P2 F1 normal = -z →
    #   λ_z(P2) = -σ_app. This is independent of slave_first.
    if formulation isa SinglePassFormulation
        lam_ref_fn = cp -> -σ_app   # always -σ_app for SP
    else
        lam_ref_fn = cp -> (cp <= ncp1 ? +σ_app : -σ_app)  # P1=+, P2=-
    end
    nlm_orig = length(Pc)
    nlm_active = size(Z, 1) ÷ nsd
    max_lam_err = NaN
    if nlm_active > 0
        active_z = findall(active[(nsd-1)*nlm_orig+1 : nsd*nlm_orig])
        lam_z = lam[(nsd-1)*nlm_active+1 : nsd*nlm_active]
        errs = Float64[]
        for (k, ic) in enumerate(active_z)
            cp = Pc[ic]
            # Skip rim CPs (P1 face outside overlap)
            if cp <= ncp1
                x, y = B[cp, 1], B[cp, 2]
                in_overlap = (x > δ - 1e-6 && x < δ + L_upper + 1e-6 &&
                              y > δ - 1e-6 && y < δ + L_upper + 1e-6)
                in_overlap || continue
            end
            push!(errs, abs(lam_z[k] - lam_ref_fn(cp)))
        end
        !isempty(errs) && (max_lam_err = maximum(errs))
    end

    h = L_lower / n_lower

    return (l2_disp=l2_disp, linf_disp=max_err, lam_err=max_lam_err, ndof=neq, n_lam=size(Z,1),
            h=h, wall_s=wall, kappa=kappa, kappa_kkt=kappa_kkt,
            d0_p1=d0_p1, d1_p1=d1_p1, d2_p1=d2_p1,
            d0_p2=d0_p2, d1_p2=d1_p2, d2_p2=d2_p2)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Study section (skip when included from another script)
# ═══════════════════════════════════════════════════════════════════════════════

if get(ENV, "TM_SKIP_STUDY", "") != "1"

results_base = get(ENV, "TM_RESULTS_DIR",
                  joinpath(@__DIR__, "..", "..", "results"))
results_dir = joinpath(results_base, "flat_patch_test_3d",
                       Dates.format(now(), "yyyy-mm-dd") * "_benchmark")
mkpath(results_dir)
t_start = time()

all_methods = [
    ("TME",  TwinMortarFormulation(),  ElementBasedIntegration()),
    ("TMS",  TwinMortarFormulation(),  SegmentBasedIntegration()),
    ("DPME", DualPassFormulation(),    ElementBasedIntegration()),
    ("DPMS", DualPassFormulation(),    SegmentBasedIntegration()),
    ("SPME", SinglePassFormulation(),  ElementBasedIntegration()),
    ("SPMS", SinglePassFormulation(),  SegmentBasedIntegration()),
]

slave_labels = Dict(:lower => "sL", :upper => "sU")
slave_choices = [:lower, :upper]
p_range = [1, 2, 3, 4]

# ── 1. Factorial: all 6 methods × p=1,2,3,4 × both slave orientations ───────
println("=== Study 1: Factorial ===")
open(joinpath(results_dir, "factorial.csv"), "w") do io
    println(io, "method,slave,p,l2_disp,lam_err,ndof,n_lam,kappa,kappa_kkt,wall_s")
    for sf in slave_choices, p in p_range, (mname, form, strat) in all_methods
        tag = mname * "_" * slave_labels[sf]
        @printf("  %s p=%d ...", tag, p); flush(stdout)
        try
            r = solve_farah3d(p; formulation=form, strategy=strat, slave_first=sf)
            @printf(" disp=%.2e  lam=%.2e  κ=%.2e  κ_kkt=%.2e  t=%.1fs\n",
                    r.l2_disp, r.lam_err, r.kappa, r.kappa_kkt, r.wall_s)
            @printf(io, "%s,%s,%d,%.6e,%.6e,%d,%d,%.6e,%.6e,%.6e\n",
                    mname, slave_labels[sf], p, r.l2_disp, r.lam_err,
                    r.ndof, r.n_lam, r.kappa, r.kappa_kkt, r.wall_s)
        catch e
            @printf(" ERROR: %s\n", sprint(showerror, e)[1:min(80,end)])
            @printf(io, "%s,%s,%d,NaN,NaN,,,NaN,NaN,NaN\n", mname, slave_labels[sf], p)
        end
    end
end

# ── 2. ε sweep: TME,TMS,DPME,DPMS × p=1,2,3,4 × both orientations ─────────
println("\n=== Study 2: ε sweep ===")
eps_methods = [
    ("TME",  TwinMortarFormulation(),  ElementBasedIntegration()),
    ("TMS",  TwinMortarFormulation(),  SegmentBasedIntegration()),
    ("DPME", DualPassFormulation(),    ElementBasedIntegration()),
    ("DPMS", DualPassFormulation(),    SegmentBasedIntegration()),
]
epss_range = 10.0 .^ (-2:0.5:6)

open(joinpath(results_dir, "eps_sweep.csv"), "w") do io
    println(io, "method,slave,p,eps,l2_disp,lam_err,kappa,kappa_kkt,wall_s")
    for sf in slave_choices, p in p_range, (mname, form, strat) in eps_methods
        tag = mname * "_" * slave_labels[sf]
        @printf("  %s p=%d ε sweep ...", tag, p); flush(stdout)
        for eps in epss_range
            try
                r = solve_farah3d(p; epss=eps, formulation=form, strategy=strat, slave_first=sf)
                @printf(io, "%s,%s,%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                        mname, slave_labels[sf], p, eps,
                        r.l2_disp, r.lam_err, r.kappa, r.kappa_kkt, r.wall_s)
            catch
                @printf(io, "%s,%s,%d,%.6e,NaN,NaN,NaN,NaN,NaN\n",
                        mname, slave_labels[sf], p, eps)
            end
        end
        println(" done")
    end
end

# ── 3. NQUAD sweep: TME,DPME,SPME × p=1,2,3,4 × both orientations ─────────
println("\n=== Study 3: NQUAD sweep ===")
nquad_methods = [
    ("TME",  TwinMortarFormulation(),  ElementBasedIntegration()),
    ("DPME", DualPassFormulation(),    ElementBasedIntegration()),
    ("SPME", SinglePassFormulation(),  ElementBasedIntegration()),
]
nquad_range = 2:20

open(joinpath(results_dir, "nquad_sweep.csv"), "w") do io
    println(io, "method,slave,p,nquad,l2_disp,lam_err,wall_s")
    for sf in slave_choices, p in p_range, (mname, form, strat) in nquad_methods
        tag = mname * "_" * slave_labels[sf]
        @printf("  %s p=%d NQUAD sweep ...", tag, p); flush(stdout)
        for nq in nquad_range
            try
                r = solve_farah3d(p; NQUAD_mortar=nq, formulation=form, strategy=strat, slave_first=sf)
                @printf(io, "%s,%s,%d,%d,%.6e,%.6e,%.6e\n",
                        mname, slave_labels[sf], p, nq,
                        r.l2_disp, r.lam_err, r.wall_s)
            catch
                @printf(io, "%s,%s,%d,%d,NaN,NaN,NaN\n",
                        mname, slave_labels[sf], p, nq)
            end
        end
        println(" done")
    end
end

# ── 4. Moments: all 6 methods × p=1,2,3,4 × both orientations ──────────────
println("\n=== Study 4: Moments ===")
open(joinpath(results_dir, "moments.csv"), "w") do io
    println(io, "method,slave,p,d0_p1,d1_p1,d2_p1,d0_p2,d1_p2,d2_p2,d0_sum,d1_sum,d2_sum")
    for sf in slave_choices, p in p_range, (mname, form, strat) in all_methods
        tag = mname * "_" * slave_labels[sf]
        @printf("  %s p=%d moments ...", tag, p); flush(stdout)
        try
            r = solve_farah3d(p; formulation=form, strategy=strat, slave_first=sf)
            d0s = r.d0_p1 + r.d0_p2
            d1s = r.d1_p1 + r.d1_p2
            d2s = r.d2_p1 + r.d2_p2
            @printf(" d0=%.4e\n", d0s)
            @printf(io, "%s,%s,%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                    mname, slave_labels[sf], p,
                    r.d0_p1, r.d1_p1, r.d2_p1,
                    r.d0_p2, r.d1_p2, r.d2_p2,
                    d0s, d1s, d2s)
        catch e
            @printf(" ERROR: %s\n", sprint(showerror, e)[1:min(80,end)])
            @printf(io, "%s,%s,%d,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN\n",
                    mname, slave_labels[sf], p)
        end
    end
end

# ── 5. Meta (TOML) ──────────────────────────────────────────────────────────
write_meta_toml(
    results_dir;
    benchmark = "flat_patch_test_3d",
    description = "3D flat patch test benchmark: factorial + eps sweep + NQUAD sweep + moments",
    parameters = Dict(
        "epss_default"       => 10.0,
        "NQUAD"              => "p+1",
        "NQUAD_mortar"       => "p+2",
        "degrees"            => [1, 2, 3, 4],
        "slave_orientations" => ["sL (lower=slave)", "sU (upper=slave)"],
        "eps_range"          => [10.0^e for e in -2:0.5:6],
        "nquad_range"        => collect(2:20),
        "methods"            => ["TME", "TMS", "DPME", "DPMS", "SPME", "SPMS"],
    ),
    outputs = ["factorial.csv", "eps_sweep.csv", "nquad_sweep.csv", "moments.csv"],
    wallclock_seconds = time() - t_start,
    extras = Dict(
        "geometry" => Dict(
            "L_lower"   => 10.0,  "H_lower" => 4.0,
            "L_upper"   => 5.0,   "H_upper" => 4.0,
            "n_lower"   => 5,     "n_upper" => 3,
            "n_z"       => 3,
            "interface" => "F6-F1 (overhanging, Farah et al. 2015 Fig. 10)",
        ),
        "material" => Dict("E" => 100.0, "nu" => 0.0),
        "loading"  => Dict(
            "sigma_app"   => 0.5,
            "description" => "sigma_zz = -0.5 Neumann on P2 top + P1 rim",
        ),
        "boundary_conditions" => Dict(
            "P1_bottom"   => "fully fixed (ux=uy=uz=0 on z=0)",
            "P2_top_pins" => "corner pins for rigid body (see source)",
        ),
        "studies" => Dict(
            "factorial"   => "6 methods x 4 degrees x 2 orientations",
            "eps_sweep"   => "4 methods (TME,TMS,DPME,DPMS) x 4 degrees x 2 orientations, eps=10^(-2:0.5:6)",
            "nquad_sweep" => "3 methods (TME,DPME,SPME) x 4 degrees x 2 orientations, NQUAD=2:20",
            "moments"     => "6 methods x 4 degrees x 2 orientations",
        ),
        "reference" => Dict(
            "paper"               => "Farah et al. (2015) Sec. 5.1, Fig. 10",
            "seg_based_tri_gauss" => "NQUAD_mortar=p+2: p=1->3pts, p>=2->7pts (Cowper)",
        ),
    ),
)

println("\nAll studies complete. Results saved to:\n  ", results_dir)

end  # if TM_SKIP_STUDY
