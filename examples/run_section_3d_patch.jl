# run_section_3d_patch.jl — Data for the 3D flat patch test paper section.
#
# Generates 5 datasets matching the planned figures of the section:
#   Fig. 1: λ_z along diagonal, slave=upper (P2 F1)
#   Fig. 2: λ_z along diagonal, slave=lower (P1 F6)
#   Fig. 3: max ‖λ_h - λ‖_∞ vs n_gp ∈ {3, 8, 10, 20} for SPME, DPME, TME (ε=1e7)
#   Fig. 4: max ‖λ_h - λ‖_∞ vs ε ∈ 10^(-2:0.5:20) for TMS, DPMS, TME, DPME
#   Fig. 5: κ(K_KKT) vs ε ∈ 10^(-2:0.5:20) for TMS, DPMS, TME, DPME
#
# Both Fig. 1 and Fig. 2 use the SAME diagonal coordinate range so that
# dual-pass methods (TM, DP) with two multiplier fields can be plotted.
# The diagonal coordinate s is centered at the geometric center (5, 5, 4)
# of both interfaces; CPs from BOTH P1 and P2 lying on the diagonal
# x = y are extracted, with s = (x − 5) · √2.
#
# Output: section_3d_patch/{fig1,fig2,fig3,fig4,fig5}.csv
#
# Geometry: Farah-style overhang (n_lower=5, n_upper=3, L_lower=10, L_upper=5)
# All studies use p = 1.

import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

ENV["TM_SKIP_STUDY"] = "1"
include(joinpath(@__DIR__, "run_flat_patchtest_3d.jl"))
delete!(ENV, "TM_SKIP_STUDY")

using Printf, Dates, LinearAlgebra, SparseArrays

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

const P_ORD = 1
const NSD = 3
const SIGMA_APP = 0.5
const N_LOWER = 5
const N_UPPER = 3
const N_Z = 3
const L_LOWER = 10.0
const L_UPPER = 5.0
const H_LOWER = 4.0
const H_UPPER = 4.0
const DELTA = (L_LOWER - L_UPPER) / 2  # = 2.5
const CENTER_X = 5.0   # geometric center of interface
const CENTER_Y = 5.0
const TOL_DIAG = 1e-6  # tolerance for "on diagonal x = y"

const NQUAD_DEFAULT = P_ORD + 2  # 3
const EPS_DEFAULT_TWO_PASS = 1.0  # for DPMS, TMS in Fig. 1 & 2 (machine precision)
const EPS_DEFAULT_ELEM = 1e7      # for DPME, TME in Fig. 1 & 2 (Z⁻¹C cancellation regime)

# ═══════════════════════════════════════════════════════════════════════════════
# Solve helper that returns full state needed for diagonal extraction
# ═══════════════════════════════════════════════════════════════════════════════

"""
    solve_and_extract(form, strat; epss, NQUAD_mortar, slave_first)

Run the 3D flat patch test for the given configuration and return:
- `lam_z` : full multiplier z-component vector
- `Pc_active` : active multiplier CP indices (after droptol cleanup)
- `B`, `P` : geometry and patch indexing
- `ncp1` : number of P1 CPs
- `nlm` : number of active multiplier CPs (per direction)
"""
function solve_and_extract(form, strat;
    epss::Float64 = EPS_DEFAULT_TWO_PASS,
    NQUAD_mortar::Int = NQUAD_DEFAULT,
    slave_first::Symbol = :upper)

    nsd = NSD; npd = 3; ned = 3; npc = 2

    # ── Build geometry (same as solve_farah3d) ────────────────────────────
    nc = (n, p) -> n + p
    nc_x1 = nc(N_LOWER, P_ORD); nc_y1 = nc(N_LOWER, P_ORD); nc_z1 = nc(N_Z, P_ORD)
    nc_x2 = nc(N_UPPER, P_ORD); nc_y2 = nc(N_UPPER, P_ORD); nc_z2 = nc(N_Z, P_ORD)

    B1 = zeros(nc_x1*nc_y1*nc_z1, 4)
    for k in 1:nc_z1, j in 1:nc_y1, i in 1:nc_x1
        A = (k-1)*nc_x1*nc_y1 + (j-1)*nc_x1 + i
        B1[A,:] = [(i-1)/(nc_x1-1)*L_LOWER, (j-1)/(nc_y1-1)*L_LOWER,
                   (k-1)/(nc_z1-1)*H_LOWER, 1.0]
    end
    B2 = zeros(nc_x2*nc_y2*nc_z2, 4)
    for k in 1:nc_z2, j in 1:nc_y2, i in 1:nc_x2
        A = (k-1)*nc_x2*nc_y2 + (j-1)*nc_x2 + i
        B2[A,:] = [DELTA + (i-1)/(nc_x2-1)*L_UPPER,
                   DELTA + (j-1)/(nc_y2-1)*L_UPPER,
                   H_LOWER + (k-1)/(nc_z2-1)*H_UPPER, 1.0]
    end
    ncp1 = size(B1, 1)
    B = vcat(B1, B2)
    P = [collect(1:ncp1), collect(ncp1+1:ncp1+size(B2,1))]
    p_mat = fill(P_ORD, npc, npd)
    n_mat = [nc_x1 nc_y1 nc_z1; nc_x2 nc_y2 nc_z2]
    KV = [[open_kv(N_LOWER, P_ORD), open_kv(N_LOWER, P_ORD), open_kv(N_Z, P_ORD)],
          [open_kv(N_UPPER, P_ORD), open_kv(N_UPPER, P_ORD), open_kv(N_Z, P_ORD)]]

    nel, nnp, nen = patch_metrics(npc, npd, p_mat, n_mat)
    IEN = build_ien(nsd, npd, npc, p_mat, n_mat, nel, nnp, nen)
    INC = [build_inc(n_mat[pc,:]) for pc in 1:npc]

    # ── Farah BCs ──────────────────────────────────────────────────────────
    tol = 1e-10
    bc = [Int[] for _ in 1:ned]
    for A in P[1]
        if B[A, 3] < tol
            push!(bc[1], A); push!(bc[2], A); push!(bc[3], A)
        end
    end
    z_top = H_LOWER + H_UPPER
    corners = [
        (DELTA, DELTA, true, true),
        (DELTA + L_UPPER, DELTA, true, false),
        (DELTA, DELTA + L_UPPER, false, true),
    ]
    for (cx, cy, fix_ux, fix_uy) in corners, A in P[2]
        x, y, z = B[A, 1:3]
        if abs(x - cx) < tol && abs(y - cy) < tol && abs(z - z_top) < tol
            fix_ux && push!(bc[1], A)
            fix_uy && push!(bc[2], A)
        end
    end
    for d in 1:ned; unique!(bc[d]); end
    neq, ID = build_id(bc, ned, size(B, 1))
    LM = build_lm(nen, ned, npc, nel, ID, IEN, P)

    mats = [LinearElastic(100.0, 0.0, :three_d) for _ in 1:npc]
    K, _ = build_stiffness_matrix(npc, nsd, npd, ned, neq, p_mat, n_mat, KV, P, B,
        zeros(size(B, 1), nsd), nen, nel, IEN, INC, LM, mats, P_ORD + 1, 1.0)

    # ── Loading: Neumann pressure on P2 top + P1 F6 with rim/overlap split ─
    stress_fn = (x, y, z) -> begin σ = zeros(3, 3); σ[3, 3] = -SIGMA_APP; σ end
    F = zeros(neq)
    F = segment_load(n_mat[2,:], p_mat[2,:], KV[2], P[2], B,
                     nnp[2], nen[2], nsd, npd, ned, Int[], 6, ID, F,
                     stress_fn, 1.0, P_ORD + 1)
    F = segment_load(n_mat[1,:], p_mat[1,:], KV[1], P[1], B,
                     nnp[1], nen[1], nsd, npd, ned, Int[], 6, ID, F,
                     stress_fn, 1.0, P_ORD + 1)
    if strat isa SegmentBasedIntegration
        F = _subtract_overlap_load!(F, p_mat[1,:], n_mat[1,:], KV[1], P[1], B,
            nnp[1], nsd, npd, ned, ID, SIGMA_APP, max(NQUAD_mortar, 7),
            [DELTA, DELTA], [DELTA + L_UPPER, DELTA + L_UPPER], H_LOWER)
    else
        F = _subtract_overlap_load_elem!(F, p_mat[1,:], n_mat[1,:], KV[1], P[1], B,
            nnp[1], nsd, npd, ned, ID, SIGMA_APP, NQUAD_mortar,
            [DELTA, DELTA], [DELTA + L_UPPER, DELTA + L_UPPER], H_LOWER)
    end
    K_bc, F_bc = enforce_dirichlet(Tuple{Int,Float64}[], K, F)

    # ── Mortar coupling ────────────────────────────────────────────────────
    if slave_first == :upper
        pair_fwd = InterfacePair(2, 1, 1, 6)
        pair_rev = InterfacePair(1, 6, 2, 1)
    else
        pair_fwd = InterfacePair(1, 6, 2, 1)
        pair_rev = InterfacePair(2, 1, 1, 6)
    end
    pairs = form isa SinglePassFormulation ? [pair_fwd] : [pair_fwd, pair_rev]
    Pc = build_interface_cps(pairs, p_mat, n_mat, KV, P, npd, nnp, form)
    C, Z = build_mortar_coupling(Pc, pairs, p_mat, n_mat, KV, P, B,
        ID, nnp, ned, nsd, npd, neq, NQUAD_mortar, epss, strat, form)

    # Cleanup: drop near-zero columns and entries
    drop_tol = 1e-12
    nlm2 = size(C, 2)
    g = zeros(nlm2)
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

    # Solve KKT directly
    U, lam = solve_mortar(K_bc, C, Z, F_bc; g=g)

    # Build active Pc list and the per-CP λ_z values
    nlm_orig = length(Pc)
    nlm_active = size(Z, 1) ÷ nsd
    active_z = findall(active[(nsd-1)*nlm_orig+1 : nsd*nlm_orig])
    Pc_active = [Pc[ic] for ic in active_z]
    lam_z = lam[(nsd-1)*nlm_active+1 : nsd*nlm_active]

    return (lam_z=lam_z, Pc_active=Pc_active, B=B, P=P, ncp1=ncp1,
            nlm=nlm_active, neq=neq, K_bc=K_bc, C=C, Z=Z)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Diagonal extraction
# ═══════════════════════════════════════════════════════════════════════════════

"""
    diagonal_lambda(result, patch::Symbol)

Extract λ_z values for CPs lying on the diagonal x = y.
`patch ∈ {:p1, :p2}` selects which surface's CPs to return.

Returns a vector of (s, λ_z) tuples sorted by diagonal coordinate s, where
s = (x − CENTER_X) · √2 (centered at the interface center, range = √2 · L_diag/2).
"""
function diagonal_lambda(result, patch::Symbol)
    pts = Tuple{Float64, Float64}[]
    for (k, cp) in enumerate(result.Pc_active)
        is_p1 = cp <= result.ncp1
        if patch == :p1 && !is_p1; continue; end
        if patch == :p2 && is_p1; continue; end
        x, y, z = result.B[cp, 1], result.B[cp, 2], result.B[cp, 3]
        # On the diagonal x = y?
        if abs(x - y) < TOL_DIAG
            s = (x - CENTER_X) * sqrt(2.0)
            push!(pts, (s, result.lam_z[k]))
        end
    end
    sort!(pts, by = first)
    return pts
end

# ═══════════════════════════════════════════════════════════════════════════════
# Output directory
# ═══════════════════════════════════════════════════════════════════════════════

results_dir = get(ENV, "TM_RESULTS_DIR",
    joinpath(@__DIR__, "..", "..", "results", "section_3d_patch",
             Dates.format(now(), "yyyy-mm-dd") * "_section"))
mkpath(results_dir)

println("=== Section 3D patch test data generation ===")
println("  output: $results_dir")
println()

# ═══════════════════════════════════════════════════════════════════════════════
# Method registry
# ═══════════════════════════════════════════════════════════════════════════════

method_specs = Dict(
    "SPMS" => (SinglePassFormulation(),  SegmentBasedIntegration(), false),  # ε ignored
    "DPMS" => (DualPassFormulation(),    SegmentBasedIntegration(), true),
    "TMS"  => (TwinMortarFormulation(),  SegmentBasedIntegration(), true),
    "SPME" => (SinglePassFormulation(),  ElementBasedIntegration(), false),
    "DPME" => (DualPassFormulation(),    ElementBasedIntegration(), true),
    "TME"  => (TwinMortarFormulation(),  ElementBasedIntegration(), true),
)

# ═══════════════════════════════════════════════════════════════════════════════
# Fig. 1 & 2: Diagonal λ profile
# ═══════════════════════════════════════════════════════════════════════════════

# Six methods to compare. ε for two-pass methods:
#   segment-based (DPMS, TMS): ε = 1
#   element-based (DPME, TME): ε = 1e7
fig12_methods = [
    ("SPMS", 0.0),
    ("DPMS", EPS_DEFAULT_TWO_PASS),
    ("TMS",  EPS_DEFAULT_TWO_PASS),
    ("SPME", 0.0),
    ("DPME", EPS_DEFAULT_ELEM),
    ("TME",  EPS_DEFAULT_ELEM),
]

println(">>> Fig. 1 & 2: Diagonal λ profiles")
for (fig_name, sf) in [("fig1", :upper), ("fig2", :lower)]
    open(joinpath(results_dir, "$fig_name.csv"), "w") do io
        # Columns: method, eps, patch (p1 or p2), s, lam_z
        # patch=p1 → λ on P1, patch=p2 → λ on P2.
        # For SP methods, only the slave surface has multipliers.
        println(io, "method,eps,patch,s,lam_z")
        for (mname, eps_val) in fig12_methods
            form, strat, _ = method_specs[mname]
            try
                r = solve_and_extract(form, strat;
                    epss = eps_val == 0.0 ? 1.0 : eps_val,
                    NQUAD_mortar = NQUAD_DEFAULT,
                    slave_first = sf)
                # Extract diagonal λ on both patches
                for patch in [:p1, :p2]
                    pts = diagonal_lambda(r, patch)
                    for (s, lz) in pts
                        @printf(io, "%s,%.6e,%s,%.6e,%.6e\n",
                                mname, eps_val, string(patch), s, lz)
                    end
                end
                npts_p1 = length(diagonal_lambda(r, :p1))
                npts_p2 = length(diagonal_lambda(r, :p2))
                @printf("  %s_%s: %d P1 + %d P2 diagonal CPs\n",
                        mname, fig_name, npts_p1, npts_p2)
            catch e
                @printf("  %s_%s: ERROR %s\n", mname, fig_name,
                        sprint(showerror, e)[1:min(60, end)])
            end
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Fig. 3: max ‖λ_h - λ‖_∞ vs n_gp ∈ {3, 8, 10, 20} for SPME, DPME, TME
# ═══════════════════════════════════════════════════════════════════════════════

println(">>> Fig. 3: λ error vs n_gp (element-based methods)")
fig3_methods = [
    ("SPME", 0.0),
    ("DPME", EPS_DEFAULT_ELEM),
    ("TME",  EPS_DEFAULT_ELEM),
]
nq_values = [3, 8, 10, 20]

# λ reference for slave=upper:
#   - P1 CPs in overlap: +σ_app (TM/DP) or skipped for SP
#   - P2 CPs (all overlap): -σ_app (slave for SP and TM/DP)
function lam_inf_error(r, sf, form)
    is_sp = form isa SinglePassFormulation
    errs = Float64[]
    for (k, cp) in enumerate(r.Pc_active)
        is_p1 = cp <= r.ncp1
        # For SP, only slave CPs exist
        x, y = r.B[cp, 1], r.B[cp, 2]
        if is_p1
            in_ov = (x > DELTA - 1e-6 && x < DELTA + L_UPPER + 1e-6 &&
                     y > DELTA - 1e-6 && y < DELTA + L_UPPER + 1e-6)
            in_ov || continue
        end
        if is_sp
            ref = -SIGMA_APP   # SP λ has slave-normal sign baked in
        else
            ref = is_p1 ? +SIGMA_APP : -SIGMA_APP
        end
        push!(errs, abs(r.lam_z[k] - ref))
    end
    isempty(errs) ? NaN : maximum(errs)
end

open(joinpath(results_dir, "fig3.csv"), "w") do io
    println(io, "method,eps,nq,slave,lam_err_inf")
    for sf in [:upper, :lower]
        for (mname, eps_val) in fig3_methods
            form, strat, _ = method_specs[mname]
            for nq in nq_values
                try
                    r = solve_and_extract(form, strat;
                        epss = eps_val == 0.0 ? 1.0 : eps_val,
                        NQUAD_mortar = nq, slave_first = sf)
                    err = lam_inf_error(r, sf, form)
                    @printf(io, "%s,%.6e,%d,%s,%.6e\n",
                            mname, eps_val, nq, string(sf), err)
                    @printf("  %s_%s nq=%2d: err=%.3e\n", mname, sf, nq, err)
                catch e
                    @printf(io, "%s,%.6e,%d,%s,NaN\n",
                            mname, eps_val, nq, string(sf))
                end
            end
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Fig. 4: max ‖λ_h - λ‖_∞ vs ε for TMS, DPMS, TME, DPME
# ═══════════════════════════════════════════════════════════════════════════════

println(">>> Fig. 4: λ error vs ε (two-pass methods)")
fig4_methods = ["TMS", "DPMS", "TME", "DPME"]
eps_range = 10.0 .^ (-2:0.5:20)

open(joinpath(results_dir, "fig4.csv"), "w") do io
    println(io, "method,eps,slave,lam_err_inf")
    for sf in [:upper, :lower]
        for mname in fig4_methods
            form, strat, _ = method_specs[mname]
            for eps_val in eps_range
                try
                    r = solve_and_extract(form, strat;
                        epss = eps_val, NQUAD_mortar = NQUAD_DEFAULT,
                        slave_first = sf)
                    err = lam_inf_error(r, sf, form)
                    @printf(io, "%s,%.6e,%s,%.6e\n",
                            mname, eps_val, string(sf), err)
                catch
                    @printf(io, "%s,%.6e,%s,NaN\n", mname, eps_val, string(sf))
                end
            end
            @printf("  %s_%s: ε sweep done\n", mname, sf)
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Fig. 5: κ(K_KKT) vs ε for TMS, DPMS, TME, DPME
# ═══════════════════════════════════════════════════════════════════════════════

println(">>> Fig. 5: κ(K_KKT) vs ε")
fig5_methods = ["TMS", "DPMS", "TME", "DPME"]

open(joinpath(results_dir, "fig5.csv"), "w") do io
    println(io, "method,eps,slave,kappa_kkt")
    for sf in [:upper]   # one slave orientation suffices for conditioning
        for mname in fig5_methods
            form, strat, _ = method_specs[mname]
            for eps_val in eps_range
                try
                    r = solve_and_extract(form, strat;
                        epss = eps_val, NQUAD_mortar = NQUAD_DEFAULT,
                        slave_first = sf)
                    Kd = Matrix(r.K_bc); Cd = Matrix(r.C); Zd = Matrix(r.Z)
                    A_kkt = [Kd Cd; Cd' Zd]
                    κ = cond(A_kkt)
                    @printf(io, "%s,%.6e,%s,%.6e\n",
                            mname, eps_val, string(sf), κ)
                catch
                    @printf(io, "%s,%.6e,%s,NaN\n", mname, eps_val, string(sf))
                end
            end
            @printf("  %s_%s: κ sweep done\n", mname, sf)
        end
    end
end

println()
println("=== All section data written to $results_dir ===")
