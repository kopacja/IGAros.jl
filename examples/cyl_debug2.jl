# cyl_debug2.jl — deep diagnostics for concentric cylinder problem
import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using IGAros, LinearAlgebra, SparseArrays, Printf

include("concentric_cylinders.jl")

# ── Rebuild solve_cylinder internals and check specific quantities ─────────────
function solve_cylinder_deep(
    p_ord::Int, exp_level::Int;
    r_i=1.0, r_c=1.5, r_o=2.0, E=100.0, nu=0.3, p_o=1.0,
    epss=0.0, NQUAD=p_ord+1, NQUAD_mortar=10
)
    nsd = 2; npd = 2; ned = 2; npc = 2
    thickness = 1.0

    B0, P = cylinder_geometry(p_ord; r_i=r_i, r_c=r_c, r_o=r_o)
    p_mat = fill(p_ord, npc, npd)
    n_mat = fill(p_ord + 1, npc, npd)
    KV    = generate_knot_vectors(npc, npd, p_mat, n_mat)

    s_ang    = (1/3) / 2^exp_level
    s_ang_nc = s_ang / 2
    s_rad    = (1/2) / 2^exp_level
    u_ang    = collect(s_ang    : s_ang    : 1 - s_ang/2)
    u_ang_nc = collect(s_ang_nc : s_ang_nc : 1 - s_ang_nc/2)
    u_rad    = collect(s_rad    : s_rad    : 1 - s_rad/2)
    epss_use = epss > 0.0 ? epss : s_ang^2 / E

    kref_data = Vector{Float64}[
        vcat([1.0, 1.0], u_ang_nc),
        vcat([2.0, 1.0], u_ang),
        vcat([1.0, 2.0], u_rad),
        vcat([2.0, 2.0], u_rad),
    ]

    B0_hack = copy(B0)
    B0_hack[P[1], 2] .+= 1000.0
    n_mat_ref, _, KV_ref, B_hack_ref, P_ref = krefinement(
        nsd, npd, npc, n_mat, p_mat, KV, B0_hack, P, kref_data
    )
    B_ref = copy(B_hack_ref)
    for i in axes(B_ref, 1)
        B_ref[i, 2] > 100.0 && (B_ref[i, 2] -= 1000.0)
    end
    ncp = size(B_ref, 1)

    nel, nnp, nen = patch_metrics(npc, npd, p_mat, n_mat_ref)
    IEN = build_ien(nsd, npd, npc, p_mat, n_mat_ref, nel, nnp, nen)
    INC = [build_inc(n_mat_ref[pc, :]) for pc in 1:npc]

    dBC = [1 4 2 1 2;
           2 2 2 1 2]
    bc_per_dof = dirichlet_bc_control_points(p_mat, n_mat_ref, KV_ref, P_ref,
                                              npd, nnp, ned, dBC)
    neq, ID = build_id(bc_per_dof, ned, ncp)
    LM      = build_lm(nen, ned, npc, nel, ID, IEN, P_ref)

    mats = [LinearElastic(E, nu, :plane_strain), LinearElastic(E, nu, :plane_strain)]
    Ub0  = zeros(ncp, nsd)
    K, _ = build_stiffness_matrix(npc, nsd, npd, ned, neq,
                                   p_mat, n_mat_ref, KV_ref, P_ref, B_ref, Ub0,
                                   nen, nel, IEN, INC, LM, mats, NQUAD, thickness)

    stress_fn = (x, y) -> lame_stress(x, y; p_o=p_o, r_i=r_i, r_o=r_o)
    F = zeros(neq)
    F = segment_load(n_mat_ref[2,:], p_mat[2,:], KV_ref[2], P_ref[2], B_ref,
                     nnp[2], nen[2], nsd, npd, ned,
                     Int[], 3, ID, F, stress_fn, thickness, NQUAD)

    K_bc, F_bc = enforce_dirichlet(Tuple{Int,Float64}[], K, F)

    pairs = [InterfacePair(1, 3, 2, 1),
             InterfacePair(2, 1, 1, 3)]
    Pc = build_interface_cps(pairs, p_mat, n_mat_ref, KV_ref, P_ref, npd, nnp)
    C, Z = build_mortar_coupling(Pc, pairs, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
                                  ID, nnp, ned, nsd, npd, neq, NQUAD_mortar, epss_use)

    U, Lam = solve_mortar(K_bc, C, Z, F_bc)

    @printf("=== Deep diag exp=%d ===\n", exp_level)

    # ── 1. Inject exact displacement and compute error ─────────────────────────
    U_exact = zeros(neq)
    for cp in 1:ncp, i in 1:nsd
        eq = ID[i, cp]
        eq == 0 && continue
        x, y = B_ref[cp, 1], B_ref[cp, 2]
        ux, uy = lame_displacement(x, y; p_o=p_o, r_i=r_i, r_o=r_o, E=E, nu=nu)
        U_exact[eq] = (i == 1) ? ux : uy
    end
    err_abs_ex, err_ref_ex = l2_stress_error_cyl(
        U_exact, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
        nen, nel, IEN, INC, mats, NQUAD, thickness, stress_fn)
    @printf("  Exact displacement: rel=%.4e (should be ≈0 or at IGA interp error)\n",
            err_abs_ex / err_ref_ex)

    # ── 2. Check displacement on interface from both sides ─────────────────────
    # Patch 1 facet 3: outer arc of inner patch (r=r_c)
    # Patch 2 facet 1: inner arc of outer patch (r=r_c)
    # Sample at a few θ values

    Ub = zeros(ncp, nsd)
    for cp in 1:ncp, i in 1:nsd
        eq = ID[i, cp]
        eq != 0 && (Ub[cp, i] = U[eq])
    end
    Ub_exact = zeros(ncp, nsd)
    for cp in 1:ncp
        x, y = B_ref[cp, 1], B_ref[cp, 2]
        ux, uy = lame_displacement(x, y; p_o=p_o, r_i=r_i, r_o=r_o, E=E, nu=nu)
        Ub_exact[cp, 1] = ux
        Ub_exact[cp, 2] = uy
    end

    # Interface CPs from Patch 1 facet 3 (η=n)
    npc1 = nnp[1]
    iface1 = [P_ref[1][a] for a in 1:npc1 if begin
        nc = nurbs_coords(a, npd, n_mat_ref[1,:])
        nc[2] == n_mat_ref[1,2]
    end]
    iface2 = [P_ref[2][a] for a in 1:nnp[2] if begin
        nc = nurbs_coords(a, npd, n_mat_ref[2,:])
        nc[2] == 1
    end]

    @printf("  Iface1 CPs (%d): avg|u-u_ex|=%.3e  (P1 facet3, should be close to exact)\n",
            length(iface1),
            length(iface1)>0 ? mean(norm.(eachrow(Ub[iface1,:] .- Ub_exact[iface1,:]))) : NaN)
    @printf("  Iface2 CPs (%d): avg|u-u_ex|=%.3e  (P2 facet1, should match Iface1)\n",
            length(iface2),
            length(iface2)>0 ? mean(norm.(eachrow(Ub[iface2,:] .- Ub_exact[iface2,:]))) : NaN)

    # Sample u_r at several angular positions on the interface
    @printf("  Interface u_r comparison (P1 vs P2 vs exact):\n")
    for θ_deg in [15, 30, 45, 60, 75]
        θ = θ_deg * π / 180
        xq, yq = r_c * cos(θ), r_c * sin(θ)
        # Interpolate from Patch 1 side (closest CP to this angle on iface1)
        if length(iface1) > 0
            angles1 = [atan(B_ref[cp,2], B_ref[cp,1]) for cp in iface1]
            best1 = argmin(abs.(angles1 .- θ))
            cp1 = iface1[best1]
            ur1 = Ub[cp1, 1] * cos(θ) + Ub[cp1, 2] * sin(θ)
        else
            ur1 = NaN
        end
        if length(iface2) > 0
            angles2 = [atan(B_ref[cp,2], B_ref[cp,1]) for cp in iface2]
            best2 = argmin(abs.(angles2 .- θ))
            cp2 = iface2[best2]
            ur2 = Ub[cp2, 1] * cos(θ) + Ub[cp2, 2] * sin(θ)
        else
            ur2 = NaN
        end
        ux_ex, uy_ex = lame_displacement(xq, yq; p_o=p_o, r_i=r_i, r_o=r_o, E=E, nu=nu)
        ur_ex = ux_ex * cos(θ) + uy_ex * sin(θ)
        @printf("    θ=%2d°: ur_P1=%.5f  ur_P2=%.5f  ur_ex=%.5f\n",
                θ_deg, ur1, ur2, ur_ex)
    end

    # ── 3. Check if C is consistent ────────────────────────────────────────────
    # C^T * U should ≈ Z * Lam
    constraint_res = norm(C' * U - Z * Lam)
    @printf("  Constraint residual ||C^T U - Z λ||=%.4e\n", constraint_res)
    @printf("  KKT residual ||K U + C λ - F||=%.4e\n", norm(K_bc*U + C*Lam - F_bc))

    # ── 4. Error per patch ─────────────────────────────────────────────────────
    for pc in 1:npc
        err2 = 0.0; ref2 = 0.0
        D = elastic_constants(mats[pc], nsd)
        GPW = gauss_product(NQUAD, npd)
        ien = IEN[pc]; inc = INC[pc]
        for el in 1:nel[pc]
            anchor = ien[el, 1]
            n0     = inc[anchor]
            for (gp, gw) in GPW
                R_s, dR_dx, _, detJ, _ = shape_function(
                    p_mat[pc,:], n_mat_ref[pc,:], KV_ref[pc], B_ref, P_ref[pc], gp,
                    nen[pc], nsd, npd, el, n0, ien, inc)
                detJ <= 0 && continue
                gwJ = gw * detJ * thickness
                B0 = strain_displacement_matrix(nsd, nen[pc], dR_dx')
                Ue = vec(Ub[P_ref[pc][ien[el,:]], 1:nsd]')
                σ_h_v = D * (B0 * Ue)
                Xe = B_ref[P_ref[pc][ien[el,:]], :]
                X  = Xe' * R_s
                σ_ex = stress_fn(X[1], X[2])
                σ_h_m = [σ_h_v[1] σ_h_v[3]; σ_h_v[3] σ_h_v[2]]
                diff_m = σ_h_m - σ_ex
                err2 += dot(diff_m, diff_m) * gwJ
                ref2 += dot(σ_ex, σ_ex) * gwJ
            end
        end
        @printf("  Patch %d: ||e||=%.4e  ||σ_ref||=%.4e  rel=%.4f\n",
                pc, sqrt(err2), sqrt(ref2), sqrt(err2)/sqrt(ref2))
    end
    println()
end

using Statistics
solve_cylinder_deep(2, 1)
solve_cylinder_deep(2, 3)
