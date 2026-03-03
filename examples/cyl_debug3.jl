# cyl_debug3.jl — check F, uncoupled solve, large epss
import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using IGAros, LinearAlgebra, SparseArrays, Printf

include("concentric_cylinders.jl")

# ── Large epss → mortar effectively off ───────────────────────────────────────
println("=== Large epss=1e6 (mortar off) ===")
for e in 0:3
    rel, abs_e = solve_cylinder(2, e; epss=1e6)
    @printf("exp=%d  rel=%.4e  abs=%.4e\n", e, rel, abs_e)
end

# ── Internal check: F, K, uncoupled solve ─────────────────────────────────────
function check_fk(p_ord::Int, exp_level::Int)
    nsd=2; npd=2; ned=2; npc=2
    r_i=1.0; r_c=1.5; r_o=2.0; E=100.0; nu=0.3; p_o=1.0
    thickness=1.0; NQUAD=p_ord+1

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

    kref_data = Vector{Float64}[
        vcat([1.0, 1.0], u_ang_nc),
        vcat([2.0, 1.0], u_ang),
        vcat([1.0, 2.0], u_rad),
        vcat([2.0, 2.0], u_rad),
    ]
    B0_hack = copy(B0)
    B0_hack[P[1], 2] .+= 1000.0
    n_mat_ref, _, KV_ref, B_hack_ref, P_ref = krefinement(
        nsd, npd, npc, n_mat, p_mat, KV, B0_hack, P, kref_data)
    B_ref = copy(B_hack_ref)
    for i in axes(B_ref, 1)
        B_ref[i, 2] > 100.0 && (B_ref[i, 2] -= 1000.0)
    end
    ncp = size(B_ref, 1)

    nel, nnp, nen = patch_metrics(npc, npd, p_mat, n_mat_ref)
    IEN = build_ien(nsd, npd, npc, p_mat, n_mat_ref, nel, nnp, nen)
    INC = [build_inc(n_mat_ref[pc, :]) for pc in 1:npc]
    dBC = [1 4 2 1 2; 2 2 2 1 2]
    bc_per_dof = dirichlet_bc_control_points(p_mat, n_mat_ref, KV_ref, P_ref, npd, nnp, ned, dBC)
    neq, ID = build_id(bc_per_dof, ned, ncp)
    LM = build_lm(nen, ned, npc, nel, ID, IEN, P_ref)

    mats = [LinearElastic(E, nu, :plane_strain), LinearElastic(E, nu, :plane_strain)]
    Ub0 = zeros(ncp, nsd)
    K, _ = build_stiffness_matrix(npc, nsd, npd, ned, neq,
                                   p_mat, n_mat_ref, KV_ref, P_ref, B_ref, Ub0,
                                   nen, nel, IEN, INC, LM, mats, NQUAD, thickness)

    stress_fn = (x, y) -> lame_stress(x, y; p_o=p_o, r_i=r_i, r_o=r_o)
    F = zeros(neq)
    F = segment_load(n_mat_ref[2,:], p_mat[2,:], KV_ref[2], P_ref[2], B_ref,
                     nnp[2], nen[2], nsd, npd, ned,
                     Int[], 3, ID, F, stress_fn, thickness, NQUAD)

    @printf("\n=== check_fk exp=%d ===\n", exp_level)
    @printf("  neq=%d  nnz(K)=%d\n", neq, nnz(K))
    @printf("  norm(F)=%.4e  sum(F)=%.4e\n", norm(F), sum(F))
    nzF = F[abs.(F) .> 1e-14]
    @printf("  |non-zero F|: n=%d  max=%.4e  min=%.4e\n",
            length(nzF), isempty(nzF) ? NaN : maximum(nzF), isempty(nzF) ? NaN : minimum(nzF))
    @printf("  K diag: min=%.4e  max=%.4e\n", minimum(diag(K)), maximum(diag(K)))

    # Solve K * U_nc = F (no mortar, patches share K → Patch 1 driven only by K off-diag)
    U_nc = K \ F
    @printf("  No-mortar solve: norm(U)=%.4e\n", norm(U_nc))

    # Sample displacement at midpoint of outer arc (θ≈45°, r=r_o) for Patch 2
    npc2 = nnp[2]
    outer2 = [P_ref[2][a] for a in 1:npc2
              if nurbs_coords(a, npd, n_mat_ref[2,:])[2] == n_mat_ref[2,2]]
    angles = [atan(B_ref[cp,2], B_ref[cp,1]) for cp in outer2]
    best = argmin(abs.(angles .- π/4))
    cp = outer2[best]
    eq1 = ID[1, cp];  eq2 = ID[2, cp]
    u1 = (eq1 != 0) ? U_nc[eq1] : 0.0
    u2 = (eq2 != 0) ? U_nc[eq2] : 0.0
    ur = u1*cos(π/4) + u2*sin(π/4)
    ux_ex, uy_ex = lame_displacement(B_ref[cp,1], B_ref[cp,2];
                                     p_o=p_o, r_i=r_i, r_o=r_o, E=E, nu=nu)
    ur_ex = ux_ex*cos(π/4) + uy_ex*sin(π/4)
    @printf("  Outer arc CP θ≈45°: ur_nc=%.5f  ur_ex=%.5f  ratio=%.3f\n",
            ur, ur_ex, ur / ur_ex)

    # Sample at interface r=r_c from Patch 2 inner arc
    iface2 = [P_ref[2][a] for a in 1:npc2
              if nurbs_coords(a, npd, n_mat_ref[2,:])[2] == 1]
    if !isempty(iface2)
        angles2 = [atan(B_ref[cp,2], B_ref[cp,1]) for cp in iface2]
        best2 = argmin(abs.(angles2 .- π/4))
        cp2 = iface2[best2]
        eq1_2 = ID[1, cp2]; eq2_2 = ID[2, cp2]
        u1_2 = (eq1_2 != 0) ? U_nc[eq1_2] : 0.0
        u2_2 = (eq2_2 != 0) ? U_nc[eq2_2] : 0.0
        ur2 = u1_2*cos(π/4) + u2_2*sin(π/4)
        ux_ex2, uy_ex2 = lame_displacement(B_ref[cp2,1], B_ref[cp2,2];
                                           p_o=p_o, r_i=r_i, r_o=r_o, E=E, nu=nu)
        ur_ex2 = ux_ex2*cos(π/4) + uy_ex2*sin(π/4)
        @printf("  Interface CP (P2 fac1) θ≈45°: ur_nc=%.5f  ur_ex=%.5f\n", ur2, ur_ex2)
    end
end

check_fk(2, 1)
check_fk(2, 2)
