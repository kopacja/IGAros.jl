# debug_sphere_3patch.jl
#
# Systematic debugging of 3-patch deltoidal sphere with mortar coupling.
# Run levels one at a time; fix issues before proceeding.

import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using IGAros
using LinearAlgebra, SparseArrays, Printf

include(joinpath(@__DIR__, "pressurized_sphere.jl"))
include(joinpath(@__DIR__, "pressurized_sphere_deltoidal.jl"))
include(joinpath(@__DIR__, "pressurized_sphere_3patch.jl"))

# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL 0: Geometry verification
# ═══════════════════════════════════════════════════════════════════════════════

"""
    check_geometry(p_ord, exp_level; r_i, r_o)

Verify that the 3-patch deltoidal geometry is correct:
  1. All CPs lie on a sphere of appropriate radius (after projection)
  2. Interface facets physically overlap (matching physical coordinates)
  3. Normals at interface are consistently oriented
"""
function check_geometry(p_ord::Int, exp_level::Int;
                        r_i::Float64 = 1.0, r_o::Float64 = 1.2,
                        n_base::Int = 2)
    nsd = 3; npd = 3; npc = 3

    # ── Build geometry ────────────────────────────────────────────────────
    B0, P = sphere_geometry_3patch(p_ord; r_i=r_i, r_o=r_o)
    p_mat = fill(p_ord, npc, npd)
    n_ang = p_ord + 1; n_rad = p_ord + 1
    n_mat = fill(0, npc, npd)
    for pc in 1:npc; n_mat[pc, :] = [n_ang, n_ang, n_rad]; end
    KV = [[vcat(zeros(p_ord+1), ones(p_ord+1)) for _ in 1:3] for _ in 1:npc]

    # h-refinement (conforming: same ξ and η)
    n_elem = n_base * 2^exp_level
    n_rad_elem = 2^exp_level
    u_surf = Float64[i/n_elem for i in 1:n_elem-1]
    u_rad  = Float64[i/n_rad_elem for i in 1:n_rad_elem-1]

    kref_data = Vector{Float64}[]
    for t in 1:npc
        push!(kref_data, vcat([Float64(t), 1.0], u_surf))
        push!(kref_data, vcat([Float64(t), 2.0], u_surf))
        push!(kref_data, vcat([Float64(t), 3.0], u_rad))
    end
    n_mat_ref, _, KV_ref, B_ref, P_ref = krefinement(
        nsd, npd, npc, n_mat, p_mat, KV, B0, P, kref_data)

    if p_ord < 4
        project_cps_to_sphere!(B_ref, P_ref, KV_ref, n_mat_ref, p_mat, r_i, r_o)
    end

    nel, nnp, nen = patch_metrics(npc, npd, p_mat, n_mat_ref)

    println("\n" * "="^70)
    @printf("LEVEL 0: Geometry check — p=%d, exp=%d, n_elem=%d\n", p_ord, exp_level, n_elem)
    println("="^70)

    # ── Check 1: CPs on sphere ────────────────────────────────────────────
    println("\n--- Check 1: CP radii ---")
    for pc in 1:npc
        n1, n2, n3 = n_mat_ref[pc, :]
        max_err = 0.0
        for k in 1:n3
            # Expected radius from linear ζ mapping
            ga_zeta_k = (k == 1) ? 0.0 : (k == n3) ? 1.0 :
                sum(KV_ref[pc][3][k+1 : k+p_ord]) / p_ord
            r_expect = r_i + ga_zeta_k * (r_o - r_i)
            for j in 1:n2, i in 1:n1
                idx = (k-1)*n1*n2 + (j-1)*n1 + i
                A = P_ref[pc][idx]
                x, y, z = B_ref[A, 1:3]
                r_cp = sqrt(x^2 + y^2 + z^2)
                err = abs(r_cp - r_expect) / r_expect
                max_err = max(max_err, err)
            end
        end
        @printf("  Patch %d: max relative radius error = %.2e\n", pc, max_err)
    end

    # ── Check 2: Interface facet overlap ──────────────────────────────────
    println("\n--- Check 2: Interface facet physical overlap ---")
    interface_pairs_def = [(1, 4, 2, 3), (1, 3, 3, 4), (2, 4, 3, 3)]

    for (s_pc, s_face, m_pc, m_face) in interface_pairs_def
        ps_s, ns_s, KVs_s, Ps_s, _, _, _, norm_sign_s, _, b_dirs_s, _ =
            get_segment_patch(p_mat[s_pc,:], n_mat_ref[s_pc,:], KV_ref[s_pc],
                              P_ref[s_pc], npd, nnp[s_pc], s_face)
        ps_m, ns_m, KVs_m, Ps_m, _, _, _, norm_sign_m, _, b_dirs_m, _ =
            get_segment_patch(p_mat[m_pc,:], n_mat_ref[m_pc,:], KV_ref[m_pc],
                              P_ref[m_pc], npd, nnp[m_pc], m_face)

        @printf("\n  Pair: Patch %d F%d (free=%s) ↔ Patch %d F%d (free=%s)\n",
                s_pc, s_face, string(b_dirs_s), m_pc, m_face, string(b_dirs_m))
        @printf("    Slave  CPs: %d (%d×%d), norm_sign=%+d\n",
                length(Ps_s), ns_s[1], ns_s[2], norm_sign_s)
        @printf("    Master CPs: %d (%d×%d), norm_sign=%+d\n",
                length(Ps_m), ns_m[1], ns_m[2], norm_sign_m)

        # Compute bounding boxes
        s_coords = B_ref[Ps_s, 1:3]
        m_coords = B_ref[Ps_m, 1:3]
        for d in 1:3
            label = ["x", "y", "z"][d]
            @printf("    %s range: slave [%.4f, %.4f], master [%.4f, %.4f]\n",
                    label,
                    minimum(s_coords[:, d]), maximum(s_coords[:, d]),
                    minimum(m_coords[:, d]), maximum(m_coords[:, d]))
        end

        # Check physical overlap: for each slave CP, find closest master CP
        min_dist = Inf; max_dist = 0.0; mean_dist = 0.0
        for i in 1:length(Ps_s)
            xs = B_ref[Ps_s[i], 1:3]
            best = Inf
            for j in 1:length(Ps_m)
                xm = B_ref[Ps_m[j], 1:3]
                best = min(best, norm(xs - xm))
            end
            min_dist = min(min_dist, best)
            max_dist = max(max_dist, best)
            mean_dist += best
        end
        mean_dist /= length(Ps_s)
        @printf("    CP distance to nearest partner: min=%.2e, max=%.2e, mean=%.2e\n",
                min_dist, max_dist, mean_dist)

        # Evaluate a few physical points on each surface and compare
        println("    Spot-check physical points at (u,v) grid:")
        test_pts = [0.0, 0.25, 0.5, 0.75, 1.0]
        max_gap = 0.0
        for u in test_pts, v in test_pts
            x_s, _, _, _, _ = eval_surface_point(u, v, ps_s, ns_s, KVs_s, B_ref, Ps_s, nsd)
            x_m, _, _, _, _ = eval_surface_point(u, v, ps_m, ns_m, KVs_m, B_ref, Ps_m, nsd)
            gap = norm(x_s - x_m)
            max_gap = max(max_gap, gap)
        end
        @printf("    Max physical gap at 5×5 grid: %.2e\n", max_gap)
        if max_gap > 1e-6
            println("    ⚠ WARNING: Large gap — faces may not physically coincide!")
            # Print corner coordinates for diagnosis
            corners = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
            for (u, v) in corners
                x_s, _, _, _, _ = eval_surface_point(u, v, ps_s, ns_s, KVs_s, B_ref, Ps_s, nsd)
                x_m, _, _, _, _ = eval_surface_point(u, v, ps_m, ns_m, KVs_m, B_ref, Ps_m, nsd)
                @printf("      (u,v)=(%.1f,%.1f): slave=(%.4f,%.4f,%.4f) master=(%.4f,%.4f,%.4f) gap=%.2e\n",
                        u, v, x_s[1], x_s[2], x_s[3], x_m[1], x_m[2], x_m[3], norm(x_s-x_m))
            end
        else
            println("    ✓ Faces physically coincide")
        end
    end

    # ── Check 3: Normal consistency ───────────────────────────────────────
    println("\n--- Check 3: Normal orientation at interface midpoint ---")
    for (s_pc, s_face, m_pc, m_face) in interface_pairs_def
        ps_s, ns_s, KVs_s, Ps_s, _, _, _, norm_sign_s, _, _, _ =
            get_segment_patch(p_mat[s_pc,:], n_mat_ref[s_pc,:], KV_ref[s_pc],
                              P_ref[s_pc], npd, nnp[s_pc], s_face)
        ps_m, ns_m, KVs_m, Ps_m, _, _, _, norm_sign_m, _, _, _ =
            get_segment_patch(p_mat[m_pc,:], n_mat_ref[m_pc,:], KV_ref[m_pc],
                              P_ref[m_pc], npd, nnp[m_pc], m_face)

        # Evaluate at midpoint (0.5, 0.5)
        x_s, dxdu_s, dxdv_s, _, _ = eval_surface_point(0.5, 0.5, ps_s, ns_s, KVs_s, B_ref, Ps_s, nsd)
        x_m, dxdu_m, dxdv_m, _, _ = eval_surface_point(0.5, 0.5, ps_m, ns_m, KVs_m, B_ref, Ps_m, nsd)

        n_s = norm_sign_s * cross(dxdu_s, dxdv_s)
        n_s = n_s / norm(n_s)
        n_m = norm_sign_m * cross(dxdu_m, dxdv_m)
        n_m = n_m / norm(n_m)

        dot_nm = dot(n_s, n_m)
        @printf("  P%dF%d ↔ P%dF%d: n_s=(%.3f,%.3f,%.3f), n_m=(%.3f,%.3f,%.3f), dot=%.4f %s\n",
                s_pc, s_face, m_pc, m_face,
                n_s[1], n_s[2], n_s[3], n_m[1], n_m[2], n_m[3], dot_nm,
                abs(dot_nm + 1.0) < 0.1 ? "✓ opposing" :
                abs(dot_nm - 1.0) < 0.1 ? "⚠ SAME direction" : "⚠ OBLIQUE")
    end

    # ── Check 4: Symmetry BCs ─────────────────────────────────────────────
    println("\n--- Check 4: Symmetry BC count ---")
    ned = 3; ncp = size(B_ref, 1)
    dBC = deltoidal_symmetry_bcs(B_ref, P_ref, ned)
    for d in 1:3
        label = ["ux=0 (x=0 plane)", "uy=0 (y=0 plane)", "uz=0 (z=0 plane)"][d]
        @printf("  %s: %d CPs constrained\n", label, length(dBC[d]))
    end
    neq, ID = build_id(dBC, ned, ncp)
    @printf("  Total DOFs: %d (of %d)\n", neq, ncp * ned)

    return nothing
end

# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL 1: Conforming mortar convergence
# ═══════════════════════════════════════════════════════════════════════════════

"""
    run_level1(p_ord; exp_range, epss_list)

Run conforming mortar convergence study at fixed ε values.
This tests the mortar coupling without non-conformity complications.
"""
function run_level1(p_ord::Int;
                    exp_range = 0:3,
                    epss_list = [1e2, 1e4, 1e6],
                    n_base::Int = 2,
                    E::Float64 = 1000.0,
                    nu::Float64 = 0.3,
                    p_i::Float64 = 1.0,
                    r_i::Float64 = 1.0,
                    r_o::Float64 = 1.2)
    println("\n" * "="^70)
    @printf("LEVEL 1: Conforming mortar convergence — p=%d\n", p_ord)
    println("="^70)

    for epss in epss_list
        @printf("\n--- ε = %.0e ---\n", epss)
        @printf("  %4s  %12s  %6s  %12s  %6s  %12s  %6s\n",
                "exp", "l2_disp", "rate", "energy", "rate", "stress", "rate")
        prev_l2 = prev_en = prev_σ = 0.0
        for exp in exp_range
            try
                res = solve_sphere_3patch_mortar(p_ord, exp;
                    conforming = true,
                    E = E, nu = nu, p_i = p_i, epss = epss,
                    r_i = r_i, r_o = r_o, n_base = n_base,
                    NQUAD_mortar = p_ord + 2)
                rate_l2 = exp > first(exp_range) ? log2(prev_l2 / res.l2_rel) : NaN
                rate_en = exp > first(exp_range) ? log2(prev_en / res.en_rel) : NaN
                rate_σ  = exp > first(exp_range) ? log2(prev_σ  / res.σ_rel)  : NaN
                @printf("  %4d  %12.4e  %6.2f  %12.4e  %6.2f  %12.4e  %6.2f\n",
                        exp, res.l2_rel, rate_l2, res.en_rel, rate_en, res.σ_rel, rate_σ)
                prev_l2 = res.l2_rel; prev_en = res.en_rel; prev_σ = res.σ_rel
            catch e
                @printf("  %4d  ERROR: %s\n", exp, sprint(showerror, e))
                break
            end
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL 2: Non-conforming mortar convergence
# ═══════════════════════════════════════════════════════════════════════════════

function run_level2(p_ord::Int;
                    exp_range = 0:3,
                    epss::Float64 = 1e4,  # use best from Level 1
                    n_base::Int = 2,
                    mesh_ratio::Float64 = 2.0,
                    E::Float64 = 1000.0,
                    nu::Float64 = 0.3,
                    p_i::Float64 = 1.0,
                    r_i::Float64 = 1.0,
                    r_o::Float64 = 1.2)
    println("\n" * "="^70)
    @printf("LEVEL 2: Non-conforming mortar — p=%d, ε=%.0e, ratio=%.1f\n",
            p_ord, epss, mesh_ratio)
    println("="^70)

    @printf("  %4s  %12s  %6s  %12s  %6s  %12s  %6s\n",
            "exp", "l2_disp", "rate", "energy", "rate", "stress", "rate")
    prev_l2 = prev_en = prev_σ = 0.0
    for exp in exp_range
        try
            res = solve_sphere_3patch_mortar(p_ord, exp;
                conforming = false,
                mesh_ratio = mesh_ratio,
                E = E, nu = nu, p_i = p_i, epss = epss,
                r_i = r_i, r_o = r_o, n_base = n_base,
                NQUAD_mortar = p_ord + 2)
            rate_l2 = exp > first(exp_range) ? log2(prev_l2 / res.l2_rel) : NaN
            rate_en = exp > first(exp_range) ? log2(prev_en / res.en_rel) : NaN
            rate_σ  = exp > first(exp_range) ? log2(prev_σ  / res.σ_rel)  : NaN
            @printf("  %4d  %12.4e  %6.2f  %12.4e  %6.2f  %12.4e  %6.2f\n",
                    exp, res.l2_rel, rate_l2, res.en_rel, rate_en, res.σ_rel, rate_σ)
            prev_l2 = res.l2_rel; prev_en = res.en_rel; prev_σ = res.σ_rel
        catch e
            @printf("  %4d  ERROR: %s\n", exp, sprint(showerror, e))
            break
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL 3: ε sweep at a fixed refinement level
# ═══════════════════════════════════════════════════════════════════════════════

function run_level3(p_ord::Int;
                    exp_level::Int = 2,
                    epss_range = 10.0 .^ (-1:8),
                    conforming::Bool = false,
                    mesh_ratio::Float64 = 2.0,
                    n_base::Int = 2,
                    E::Float64 = 1000.0,
                    nu::Float64 = 0.3,
                    p_i::Float64 = 1.0,
                    r_i::Float64 = 1.0,
                    r_o::Float64 = 1.2)
    conf_str = conforming ? "conforming" : "non-conforming"
    println("\n" * "="^70)
    @printf("LEVEL 3: ε sweep — p=%d, exp=%d, %s\n", p_ord, exp_level, conf_str)
    println("="^70)

    @printf("  %12s  %12s  %12s  %12s\n", "ε", "l2_disp", "energy", "stress")
    for epss in epss_range
        try
            res = solve_sphere_3patch_mortar(p_ord, exp_level;
                conforming = conforming,
                mesh_ratio = mesh_ratio,
                E = E, nu = nu, p_i = p_i, epss = epss,
                r_i = r_i, r_o = r_o, n_base = n_base,
                NQUAD_mortar = p_ord + 2)
            @printf("  %12.0e  %12.4e  %12.4e  %12.4e\n",
                    epss, res.l2_rel, res.en_rel, res.σ_rel)
        catch e
            @printf("  %12.0e  ERROR: %s\n", epss, sprint(showerror, e))
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Run all levels for p=1
# ═══════════════════════════════════════════════════════════════════════════════

println("\n" * "#"^70)
println("# DEBUGGING 3-PATCH DELTOIDAL SPHERE — p=1")
println("#"^70)

# Level 0: geometry sanity
check_geometry(1, 1)

# Level 1: conforming mortar convergence
run_level1(1; exp_range=0:4, epss_list=[1e2, 1e4, 1e6])
