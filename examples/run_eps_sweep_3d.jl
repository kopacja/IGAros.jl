# run_eps_sweep_3d.jl — ε sweep for 3D flat patch test with Schur complement conditioning
#
# Self-contained script: performs the Farah-style 3D flat patch test
# (same geometry as run_flat_patchtest_3d.jl) but sweeps ε via
# Schur complement factorization to separate the ε-dependent cost
# from the expensive K-factorization.
#
# For each (method, slave, p) combination:
#   1. Build geometry, BCs, stiffness, loading  (setup_farah3d)
#   2. Build C and Z_ref at eps=1             (build_and_clean_mortar)
#   3. Factorize K and precompute C'K^{-1}C   (expensive, ONCE)
#   4. Sweep eps: S = C'K^{-1}C + eps*Z_ref   (cheap dense solve)
#
# KKT sign convention:
#   [K  C ] [U  ]   [F]
#   [C' -Z] [lam] = [g]
#
#   Row 1: K*U + C*lam = F  →  U = K^{-1}(F - C*lam)
#   Row 2: C'*U - Z*lam = g →  (C'K^{-1}C + Z_s)*lam = C'K^{-1}F - g
#
#   where Z_s is the STORED matrix (with negative diagonal from assembly).
#   So S_lam = C'K^{-1}C + Z_s, and for the sweep: S_lam = CtKinvC + eps*Z_ref.
#
# Two Schur complements (both computed):
#   S_lam = C'K^{-1}C + Z       (condense U, multiplier-only, n_lam × n_lam)
#   S_u   = K - C'Z^{-1}C       (condense λ, displacement-only, neq × neq, Remark 3.1)
#         = K - (1/eps)*CtZinvC  (precompute CtZinvC = C'*Z_ref^{-1}*C once)
#
# S_u is directly comparable to single-pass condensed system (same size neq × neq).
#
# Output: eps_sweep.csv

import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using IGAros, LinearAlgebra, SparseArrays, Printf, Dates

# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions (verbatim from run_flat_patchtest_3d.jl)
# ═══════════════════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════════════════
# setup_farah3d — build everything EXCEPT mortar coupling and solve
# ═══════════════════════════════════════════════════════════════════════════════

function setup_farah3d(p_ord; n_lower=5, n_upper=3, n_z=3,
    E=100.0, nu=0.0,
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

    # BCs: Farah-style patch test
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
    K, _ = build_stiffness_matrix(npc, nsd, npd, ned, neq, p_mat, n_mat, KV, P, B,
        zeros(ncp,nsd), nen, nel, IEN, INC, LM, mats, NQUAD, 1.0)

    # Neumann loading
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

    # Interface pairs
    if slave_first == :upper
        pair_fwd = InterfacePair(2,1,1,6)
        pair_rev = InterfacePair(1,6,2,1)
    else
        pair_fwd = InterfacePair(1,6,2,1)
        pair_rev = InterfacePair(2,1,1,6)
    end
    pairs = [pair_fwd, pair_rev]   # always two-pass for this study
    Pc = build_interface_cps(pairs, p_mat, n_mat, KV, P, npd, nnp, formulation)

    return (K_bc=K_bc, F_bc=F_bc, B=B, P=P, p_mat=p_mat, n_mat=n_mat,
            KV=KV, ID=ID, nnp=nnp, ned=ned, nsd=nsd, npd=npd, neq=neq,
            npc=npc, IND=IND, pairs=pairs, Pc=Pc, slave_first=slave_first,
            E=E, nu=nu, σ_app=σ_app, δ=δ, L_upper=L_upper)
end

# ═══════════════════════════════════════════════════════════════════════════════
# build_and_clean_mortar — build C, Z at given eps; clean Dirichlet rows + inactive cols
# ═══════════════════════════════════════════════════════════════════════════════

function build_and_clean_mortar(s, NQUAD_mortar, epss, strategy, formulation;
                                drop_tol=1e-12)
    C, Z = build_mortar_coupling(s.Pc, s.pairs, s.p_mat, s.n_mat, s.KV, s.P, s.B,
        s.ID, s.nnp, s.ned, s.nsd, s.npd, s.neq, NQUAD_mortar, epss,
        strategy, formulation)

    # Constraint RHS correction for non-homogeneous Dirichlet DOFs
    nlm = size(C, 2)
    g = zeros(nlm)
    fixed_eqs = Set{Int}()
    for (eq, val) in s.IND
        push!(fixed_eqs, eq)
        g .-= C[eq, :] .* val
    end
    if !isempty(fixed_eqs)
        rows_C, cols_C, vals_C = findnz(C)
        keep = [i for i in eachindex(rows_C) if !(rows_C[i] in fixed_eqs)]
        C = sparse(rows_C[keep], cols_C[keep], vals_C[keep], size(C,1), size(C,2))
    end

    # Remove inactive multiplier columns
    active = trues(nlm)
    for j in 1:nlm
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

    return C, Z, g, active
end

# ═══════════════════════════════════════════════════════════════════════════════
# Main study
# ═══════════════════════════════════════════════════════════════════════════════

if get(ENV, "TM_SKIP_STUDY", "") != "1"

results_dir = get(ENV, "TM_RESULTS_DIR",
    joinpath(@__DIR__, "..", "..", "results", "flat_patch_test_3d",
             Dates.format(now(), "yyyy-mm-dd") * "_benchmark"))
mkpath(results_dir)

eps_methods = [
    ("TME",  TwinMortarFormulation(),  ElementBasedIntegration()),
    ("TMS",  TwinMortarFormulation(),  SegmentBasedIntegration()),
    ("DPME", DualPassFormulation(),    ElementBasedIntegration()),
    ("DPMS", DualPassFormulation(),    SegmentBasedIntegration()),
]
epss_range = 10.0 .^ (-2:0.5:20)
slave_choices = [:lower, :upper]
slave_labels = Dict(:lower => "sL", :upper => "sU")
p_range = [1, 2, 3, 4]

println("=== ε sweep with Schur complement conditioning ===")
println("  methods:  ", join([m[1] for m in eps_methods], ", "))
println("  p range:  ", p_range)
println("  ε range:  10^(", -2, ":0.5:", 20, ")  (", length(epss_range), " values)")
println("  slaves:   ", slave_choices)
println("  output:   ", joinpath(results_dir, "eps_sweep.csv"))
println()

open(joinpath(results_dir, "eps_sweep.csv"), "w") do io
    # Three solution strategies:
    #   KKT:   solve [K C; C' -Z][u;λ] = [f;g] directly
    #   S_lam: condense u → (C'K⁻¹C + Z)λ = C'K⁻¹f - g, then u = K⁻¹(f - Cλ)
    #   S_u:   condense λ → (K - C'Z⁻¹C)u = f,             then λ = -Z⁻¹Cu
    # Each gives (disp_err, lam_err). Subscript: _kkt, _sl, _su
    println(io, "method,slave,p,eps,",
            "disp_kkt,lam_kkt,",
            "disp_sl,lam_sl,",
            "disp_su,lam_su,",
            "kappa_S_lam,kappa_S_u,wall_s")

    for sf in slave_choices, p in p_range, (mname, form, strat) in eps_methods
        tag = mname * "_" * slave_labels[sf]
        @printf("  %s p=%d: setup ...", tag, p); flush(stdout)
        t_setup = time()

        local s, C, Z_ref, g, active_mask, nlm_orig
        local Cd, Zd_ref, Kd, KinvC, CtKinvC, KinvF, CtKinvF, CtZinvC, ZinvCt
        try
            s = setup_farah3d(p; formulation=form, strategy=strat, slave_first=sf)
            nlm_orig = length(s.Pc)

            # Build C and Z_ref at eps=1.0 (so Z_ref = Z_stored / 1.0 = Z_stored)
            C, Z_ref, g, active_mask = build_and_clean_mortar(s, p+2, 1.0, strat, form)

            # Precompute C'*K^{-1}*C (expensive, ONCE)
            Cd = Matrix(C)
            Zd_ref = Matrix(Z_ref)
            K_fact = lu(s.K_bc)
            KinvC = K_fact \ Cd                # neq × n_lam
            CtKinvC = Cd' * KinvC              # n_lam × n_lam dense (S_lam base)
            KinvF = K_fact \ Vector(s.F_bc)    # neq vector
            CtKinvF = Cd' * KinvF              # n_lam vector
            # For S_u = K - (1/eps)*C'*Z_ref^{-1}*C (Remark 3.1):
            # Z_ref may be singular at eps=1 → try, fall back to NaN
            Kd = Matrix(s.K_bc)                # neq × neq dense
            CtZinvC = nothing
            try
                ZinvCt = Zd_ref \ Cd'          # n_lam × neq
                CtZinvC = Cd * ZinvCt          # neq × neq dense
            catch
                @printf(" (Z_ref singular, skipping κ(S_u))")
            end
        catch e
            @printf(" SETUP ERROR: %s\n", sprint(showerror, e)[1:min(80,end)])
            for eps in epss_range
                @printf(io, "%s,%s,%d,%.6e,NaN,NaN,NaN,NaN\n",
                        mname, slave_labels[sf], p, eps)
            end
            continue
        end
        @printf(" %.1fs, sweeping ε ...", time() - t_setup); flush(stdout)

        nsd = s.nsd
        nlm_active = size(Z_ref, 1) ÷ nsd

        for eps in epss_range
            t_eps = time()
            try
                Zd = eps * Zd_ref
                # New positive-Z convention: S_λ = Z - C'K⁻¹C (Z is positive)
                S_lam = Zd - CtKinvC
                kappa_S_lam = cond(S_lam)

                kappa_S_u = NaN
                S_u_mat = nothing
                if CtZinvC !== nothing
                    # S_u = K - C Z⁻¹ C' = K - (1/eps) * CtZinvC
                    # Note: CtZinvC was precomputed with positive Z₀ now,
                    # so the sign is consistent.
                    S_u_mat = Kd - (1.0/eps) * CtZinvC
                    kappa_S_u = cond(S_u_mat)
                end

                # ── Strategy 1: S_λ (condense u) ─────────────────────────
                # (Z - C'K⁻¹C)λ = g - C'K⁻¹F
                lam_sl = S_lam \ (g - CtKinvF)
                U_sl   = KinvF - KinvC * lam_sl

                # ── Strategy 2: S_u (condense λ) ─────────────────────────
                U_su   = fill(NaN, length(KinvF))
                lam_su = fill(NaN, length(g))
                if S_u_mat !== nothing
                    # KKT: [K C; C' Z][u;λ]=[f;g]  (Z positive definite)
                    # Row 2: C'u + Zλ = g → λ = Z⁻¹(g - C'u)
                    # Row 1: Ku + Cλ = f → Ku + CZ⁻¹(g - C'u) = f
                    #   → (K - CZ⁻¹C')u = f - CZ⁻¹g
                    # S_u = K - CZ⁻¹C'  (matches paper Remark 3.1 exactly)
                    # With Zd_ref built at eps=1: CZ⁻¹C' = (1/eps) CtZinvC
                    # S_u(eps) = Kd - (1/eps) CtZinvC  (already computed above)
                    Zinv_g = (1.0/eps) * (Zd_ref \ g)
                    rhs_su = Vector(s.F_bc) - Cd * Zinv_g
                    U_su   = S_u_mat \ rhs_su
                    # Back-substitution: λ = Z⁻¹(g - C'u) = (1/eps) Z₀⁻¹ (g - C'u)
                    lam_su = (1.0/eps) * (Zd_ref \ (g - Cd' * U_su))
                end

                # ── Strategy 3: KKT direct ────────────────────────────────
                n_u = length(KinvF); n_l = length(g)
                A_kkt = [Kd Cd; Cd' Zd]
                rhs_kkt = [Vector(s.F_bc); g]
                x_kkt = A_kkt \ rhs_kkt
                U_kkt   = x_kkt[1:n_u]
                lam_kkt = x_kkt[n_u+1:end]

                # ── Error evaluation ──────────────────────────────────────
                function _disp_err(Uv)
                    e = 0.0
                    for pc in 1:s.npc, A in s.P[pc]
                        x, y, z = s.B[A, 1:3]
                        u_ex = [s.nu/s.E*s.σ_app*x, s.nu/s.E*s.σ_app*y, -s.σ_app*z/s.E]
                        for d in 1:s.ned
                            eq = s.ID[d, A]
                            u_h = eq == 0 ? 0.0 : Uv[eq]
                            e = max(e, abs(u_h - u_ex[d]))
                        end
                    end
                    return e
                end
                function _lam_err(lamv)
                    nlm_a = length(lamv) ÷ nsd
                    nlm_a == 0 && return NaN
                    lam_z = lamv[(nsd-1)*nlm_a+1:nsd*nlm_a]
                    ncp1 = length(s.P[1])
                    if form isa SinglePassFormulation
                        lam_ref_fn = cp -> -s.σ_app
                    else
                        lam_ref_fn = cp -> (cp <= ncp1 ? +s.σ_app : -s.σ_app)
                    end
                    active_z = findall(active_mask[(nsd-1)*nlm_orig+1:nsd*nlm_orig])
                    errs = Float64[]
                    for (k, ic) in enumerate(active_z)
                        cp = s.Pc[ic]
                        if cp <= ncp1
                            x, y = s.B[cp, 1], s.B[cp, 2]
                            in_ov = (x > s.δ - 1e-6 && x < s.δ + s.L_upper + 1e-6 &&
                                     y > s.δ - 1e-6 && y < s.δ + s.L_upper + 1e-6)
                            in_ov || continue
                        end
                        push!(errs, abs(lam_z[k] - lam_ref_fn(cp)))
                    end
                    return isempty(errs) ? NaN : maximum(errs)
                end

                de_kkt = _disp_err(U_kkt);  le_kkt = _lam_err(lam_kkt)
                de_sl  = _disp_err(U_sl);   le_sl  = _lam_err(lam_sl)
                de_su  = _disp_err(U_su);   le_su  = _lam_err(lam_su)

                wall = time() - t_eps
                @printf(io, "%s,%s,%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                        mname, slave_labels[sf], p, eps,
                        de_kkt, le_kkt, de_sl, le_sl, de_su, le_su,
                        kappa_S_lam, kappa_S_u, wall)
            catch e
                @printf(io, "%s,%s,%d,%.6e,%s,%.6e\n",
                        mname, slave_labels[sf], p, eps,
                        join(fill("NaN", 10), ","), time() - t_eps)
                if eps ≈ first(epss_range)
                    println("\n    ERROR at eps=$(eps): ", sprint(showerror, e)[1:min(200,end)])
                    flush(stdout)
                end
            end
        end
        @printf(" done (total %.1fs)\n", time() - t_setup)
    end
end

println("\nε sweep complete. Results saved to:\n  ", joinpath(results_dir, "eps_sweep.csv"))

end  # if TM_SKIP_STUDY
