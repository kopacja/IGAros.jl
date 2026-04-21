# run_strat_nquad_tme_3d.jl — TME p=1 sU strategy comparison vs n_gp.
#
# Sweeps NQUAD_mortar = 3..20 and ε ∈ {1e2, 1e4, 1e6, 1e8, 1e10} for TME only.
# For each (nq, ε), solves three ways (KKT, S_λ, S_u) and reports both
# disp and lam errors. Shows how the strategy comparison evolves with n_gp.
#
# Output: strat_nquad_tme_3d.csv

import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

# Skip the study section in run_flat_patchtest_3d.jl
ENV["TM_SKIP_STUDY"] = "1"
include(joinpath(@__DIR__, "run_flat_patchtest_3d.jl"))
delete!(ENV, "TM_SKIP_STUDY")

using Printf, Dates, LinearAlgebra, SparseArrays

# ── Direct solves for the three strategies ─────────────────────────────────

results_dir = get(ENV, "TM_RESULTS_DIR",
    joinpath(@__DIR__, "..", "..", "results", "flat_patch_test_3d",
             Dates.format(now(), "yyyy-mm-dd") * "_benchmark"))
mkpath(results_dir)

nq_range = [3, 4, 5, 6, 7, 8, 10, 12, 15, 20]
eps_range = [1e2, 1e4, 1e6, 1e8, 1e10]

println("=== TME p=1 sU: strategy × NQUAD × ε ===")
println("  nq:  ", nq_range)
println("  ε:   ", eps_range)
println("  output: ", joinpath(results_dir, "strat_nquad_tme_3d.csv"))
println()

# Helper: solve for given nq, ε using the three strategies and return errors
function solve_three_ways(nq::Int, eps_val::Float64)
    # Build the system using setup_farah3d-like inline code via solve_farah3d
    # We need access to K_bc, C, Z, F, g, Pc, etc. — call a custom helper.
    p_ord = 1; nsd=3; npd=3; ned=3; npc=2
    n_lower=5; n_upper=3; n_z=3
    E=100.0; nu=0.0; σ_app=0.5
    L_lower=10.0; H_lower=4.0; L_upper=5.0; H_upper=4.0
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

    # Farah-style BCs: P1 bottom fully fixed; pin P2 top corners
    tol = 1e-10
    bc = [Int[] for _ in 1:ned]
    for A in P[1]
        if B[A,3] < tol
            push!(bc[1], A); push!(bc[2], A); push!(bc[3], A)
        end
    end
    z_top = H_lower + H_upper
    corners = [(δ, δ, true, true), (δ+L_upper, δ, true, false), (δ, δ+L_upper, false, true)]
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
        zeros(ncp,nsd), nen, nel, IEN, INC, LM, mats, p_ord+1, 1.0)

    stress_fn = (x,y,z) -> begin σ=zeros(3,3); σ[3,3]=-σ_app; σ end
    F = zeros(neq)
    F = segment_load(n_mat[2,:], p_mat[2,:], KV[2], P[2], B,
                     nnp[2], nen[2], nsd, npd, ned, Int[], 6, ID, F, stress_fn, 1.0, p_ord+1)
    F = segment_load(n_mat[1,:], p_mat[1,:], KV[1], P[1], B,
                     nnp[1], nen[1], nsd, npd, ned, Int[], 6, ID, F, stress_fn, 1.0, p_ord+1)
    F = _subtract_overlap_load_elem!(F, p_mat[1,:], n_mat[1,:], KV[1], P[1], B,
        nnp[1], nsd, npd, ned, ID, σ_app, nq,
        [δ, δ], [δ+L_upper, δ+L_upper], H_lower)
    K_bc, F_bc = enforce_dirichlet(Tuple{Int,Float64}[], K, F)

    pairs = [InterfacePair(2,1,1,6), InterfacePair(1,6,2,1)]
    Pc = build_interface_cps(pairs, p_mat, n_mat, KV, P, npd, nnp, TwinMortarFormulation())
    C, Z = build_mortar_coupling(Pc, pairs, p_mat, n_mat, KV, P, B,
        ID, nnp, ned, nsd, npd, neq, nq, eps_val,
        ElementBasedIntegration(), TwinMortarFormulation())

    # Cleanup as in solve_farah3d
    nlm2 = size(C, 2)
    g = zeros(nlm2)
    drop_tol = 1e-12
    active = trues(nlm2)
    for j in 1:nlm2
        c_norm = norm(C[:, j]); z_norm = norm(Z[:, j]) + norm(Z[j, :])
        if c_norm < drop_tol && z_norm < drop_tol; active[j] = false; end
    end
    if !all(active)
        idx = findall(active)
        C = C[:, idx]; Z = Z[idx, idx]; g = g[idx]
    end
    droptol!(C, drop_tol); droptol!(Z, drop_tol)

    # Reference exact solution
    U_ex_full = zeros(neq)
    for pc in 1:npc, A in P[pc]
        eq = ID[3,A]; eq != 0 && (U_ex_full[eq] = -σ_app * B[A,3] / E)
    end

    # ── Solve three ways ──
    Cd = Matrix(C); Zd = Matrix(Z); Kd = Matrix(K_bc)
    K_fact = lu(K_bc)
    Fv = Vector(F_bc)
    KinvF = K_fact \ Fv

    # Strategy 1: KKT direct
    A_kkt = [Kd Cd; Cd' -Zd]
    x_kkt = A_kkt \ [Fv; g]
    n_u = neq
    U_kkt = x_kkt[1:n_u]; lam_kkt = x_kkt[n_u+1:end]

    # Strategy 2: S_λ (condense u)
    KinvC = K_fact \ Cd
    CtKinvC = Cd' * KinvC
    CtKinvF = Cd' * KinvF
    S_lam = CtKinvC + Zd
    lam_sl = S_lam \ (CtKinvF - g)
    U_sl = KinvF - KinvC * lam_sl

    # Strategy 3: S_u (condense λ)
    U_su = fill(NaN, n_u); lam_su = fill(NaN, length(g))
    try
        ZinvCt = Zd \ Cd'
        CtZinvC = Cd * ZinvCt
        S_u_mat = Kd - CtZinvC
        # RHS: f + C * Z⁻¹ * g
        Zinv_g = Zd \ g
        rhs_su = Fv + Cd * Zinv_g
        U_su = S_u_mat \ rhs_su
        lam_su = Zd \ (Cd' * U_su - g)
    catch; end

    # Compute errors via L² norm and λ reference
    function disp_l2(Uv)
        Ub = zeros(ncp, nsd)
        for A in 1:ncp, i in 1:nsd
            eq = ID[i, A]; eq != 0 && (Ub[A, i] = Uv[eq])
        end
        e2 = 0.0
        GPW = gauss_product(p_ord+1, npd)
        for pc in 1:npc
            ien_pc = IEN[pc]; inc_pc = INC[pc]
            for el in 1:nel[pc]
                anchor = ien_pc[el, 1]; n0 = inc_pc[anchor]
                for (gp, gw) in GPW
                    R, _, _, detJ, _ = shape_function(
                        p_mat[pc,:], n_mat[pc,:], KV[pc], B, P[pc], gp,
                        nen[pc], nsd, npd, el, n0, ien_pc, inc_pc)
                    detJ <= 0 && continue
                    u_h  = Ub[P[pc][ien_pc[el,:]], 1:nsd]' * R
                    X    = B[P[pc][ien_pc[el,:]], :]' * R
                    u_ex = [0.0, 0.0, -σ_app*X[3]/E]
                    diff = u_h[1:nsd] - u_ex
                    e2 += dot(diff, diff) * gw * detJ
                end
            end
        end
        return sqrt(e2)
    end

    function lam_err_fn(lamv)
        nlm_a = length(lamv) ÷ nsd
        nlm_a == 0 && return NaN
        lam_z = lamv[(nsd-1)*nlm_a+1:nsd*nlm_a]
        nlm_orig = length(Pc)
        active_z = findall(active[(nsd-1)*nlm_orig+1:nsd*nlm_orig])
        errs = Float64[]
        for (k, ic) in enumerate(active_z)
            cp = Pc[ic]
            if cp <= ncp1
                x, y = B[cp, 1], B[cp, 2]
                in_ov = (x > δ - 1e-6 && x < δ + L_upper + 1e-6 &&
                         y > δ - 1e-6 && y < δ + L_upper + 1e-6)
                in_ov || continue
            end
            ref = cp <= ncp1 ? +σ_app : -σ_app
            push!(errs, abs(lam_z[k] - ref))
        end
        isempty(errs) ? NaN : maximum(errs)
    end

    return (
        disp_kkt = disp_l2(U_kkt), lam_kkt = lam_err_fn(lam_kkt),
        disp_sl  = disp_l2(U_sl),  lam_sl  = lam_err_fn(lam_sl),
        disp_su  = disp_l2(U_su),  lam_su  = lam_err_fn(lam_su),
    )
end

open(joinpath(results_dir, "strat_nquad_tme_3d.csv"), "w") do io
    println(io, "nq,eps,disp_kkt,lam_kkt,disp_sl,lam_sl,disp_su,lam_su")
    for nq in nq_range
        for eps_val in eps_range
            r = solve_three_ways(nq, eps_val)
            @printf("nq=%2d ε=%.0e: dKKT=%.2e dSλ=%.2e dSu=%.2e | lKKT=%.2e lSλ=%.2e lSu=%.2e\n",
                    nq, eps_val, r.disp_kkt, r.disp_sl, r.disp_su,
                    r.lam_kkt, r.lam_sl, r.lam_su)
            @printf(io, "%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                    nq, eps_val, r.disp_kkt, r.lam_kkt,
                    r.disp_sl, r.lam_sl, r.disp_su, r.lam_su)
        end
    end
end

println("\nDone!")
