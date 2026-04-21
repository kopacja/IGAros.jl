# run_nquad_sweep_3d.jl — 3D Farah-style flat patch test: NQUAD sweep
#
# Sweeps NQUAD_mortar = 2:20 for TME, DPME, SPME × p=1..4 × sL/sU.
#
# Output directory: TM_RESULTS_DIR env var (required).
# CSV format:
#   nquad_sweep.csv: method,slave,p,nquad,l2_disp,lam_err,wall_s

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

        kv1, kv2 = KVs[1], KVs[2]
        ξ_lo = kv1[n0[1]]; ξ_hi = kv1[n0[1]+1]
        η_lo = kv2[n0[2]]; η_hi = kv2[n0[2]+1]

        corners_gp = [[-1.0,-1.0], [1.0,-1.0], [1.0,1.0], [-1.0,1.0]]
        corners_phys = Vector{Float64}[]
        for gp in corners_gp
            R, _, _, _, _ = shape_function(ps, ns, KVs, B, Ps, gp,
                                           nsen, nsd, npd-1, el, n0, ien, inc)
            x = zeros(nsd)
            for a in 1:nsen; x .+= R[a] .* B[Ps[ien[el,a]], 1:nsd]; end
            push!(corners_phys, x)
        end

        poly = sutherland_hodgman_clip(corners_phys, clip, n_up)
        length(poly) < 3 && continue

        x_lo = corners_phys[1][1]; x_hi = corners_phys[2][1]
        y_lo = corners_phys[1][2]; y_hi = corners_phys[4][2]
        dx = x_hi - x_lo; dy = y_hi - y_lo

        for (v1, v2, v3) in triangulate_polygon(poly)
            e1 = v2 .- v1; e2 = v3 .- v1
            area2 = norm(cross(e1, e2))
            area2 < 1e-30 && continue

            Fs = zeros(ned, nsen)
            for q in 1:nqp
                L1 = tri_pts[1, q]; L2 = tri_pts[2, q]; L0 = 1 - L1 - L2
                x_q = L0 .* v1 .+ L1 .* v2 .+ L2 .* v3

                gp = [2*(x_q[1] - x_lo)/dx - 1,
                      2*(x_q[2] - y_lo)/dy - 1]

                R, _, _, _, _ = shape_function(ps, ns, KVs, B, Ps, gp,
                                               nsen, nsd, npd-1, el, n0, ien, inc)

                gwJ = tri_wts[q] * area2
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

    tol = 1e-10
    σ_app = 0.5
    bc = [Int[] for _ in 1:ned]

    for A in P[1]
        if B[A,3] < tol
            push!(bc[1], A); push!(bc[2], A); push!(bc[3], A)
        end
    end

    z_top = H_lower + H_upper
    corners = [
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

    stress_fn = (x,y,z) -> begin; σ=zeros(3,3); σ[3,3]=-σ_app; σ; end
    F = zeros(neq)
    F = segment_load(n_mat[2,:], p_mat[2,:], KV[2], P[2], B,
                     nnp[2], nen[2], nsd, npd, ned, Int[], 6, ID, F, stress_fn, 1.0, NQUAD)
    F = segment_load(n_mat[1,:], p_mat[1,:], KV[1], P[1], B,
                     nnp[1], nen[1], nsd, npd, ned, Int[], 6, ID, F, stress_fn, 1.0, NQUAD)
    F = _subtract_overlap_load!(F, p_mat[1,:], n_mat[1,:], KV[1], P[1], B,
        nnp[1], nsd, npd, ned, ID, σ_app, NQUAD_mortar,
        [δ, δ], [δ+L_upper, δ+L_upper], H_lower)
    IND = Tuple{Int,Float64}[]
    K_bc, F_bc = enforce_dirichlet(IND, K, F)

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
    droptol!(C, drop_tol)
    droptol!(Z, drop_tol)

    U, lam = solve_mortar(K_bc, C, Z, F_bc; g=g)
    wall = time() - t0

    max_err = 0.0
    for pc in 1:npc, A in P[pc]
        x,y,z = B[A,1:3]
        u_ex = [nu/E*σ_app*x, nu/E*σ_app*y, -σ_app*z/E]
        for d in 1:ned
            eq = ID[d,A]; u_h = eq==0 ? 0.0 : U[eq]
            max_err = max(max_err, abs(u_h - u_ex[d]))
        end
    end

    if formulation isa SinglePassFormulation
        lam_ref_fn = cp -> -σ_app
    else
        lam_ref_fn = cp -> (cp <= ncp1 ? +σ_app : -σ_app)
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

    return (l2_disp=max_err, lam_err=max_lam_err, ndof=neq, n_lam=size(Z,1),
            h=h, wall_s=wall)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Study: NQUAD sweep
# ═══════════════════════════════════════════════════════════════════════════════

results_dir = get(ENV, "TM_RESULTS_DIR",
    joinpath(@__DIR__, "..", "..", "results", "flat_patch_test_3d",
             Dates.format(now(), "yyyy-mm-dd") * "_benchmark"))
mkpath(results_dir)

nquad_methods = [
    ("TME",  TwinMortarFormulation(),  ElementBasedIntegration()),
    ("DPME", DualPassFormulation(),    ElementBasedIntegration()),
    ("SPME", SinglePassFormulation(),  ElementBasedIntegration()),
]

slave_labels = Dict(:lower => "sL", :upper => "sU")
slave_choices = [:lower, :upper]
p_range = [1, 2, 3, 4]
nquad_range = 1:100

println("=== NQUAD sweep ===")
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

println("\nNQUAD sweep complete. Results saved to:\n  ", results_dir)
