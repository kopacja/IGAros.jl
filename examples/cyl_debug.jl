# cyl_debug.jl — quick diagnostic for concentric_cylinders.jl
import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using IGAros, LinearAlgebra, SparseArrays, Printf

include("concentric_cylinders.jl")

# ── Run diagnostics at exp=0 and exp=1 ────────────────────────────────────────
function solve_cylinder_debug(
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

    @printf("--- exp=%d ---\n", exp_level)
    @printf("  n_mat = [%d %d; %d %d]\n",
            n_mat_ref[1,1], n_mat_ref[1,2], n_mat_ref[2,1], n_mat_ref[2,2])
    @printf("  ncp=%d  neq=%d\n", ncp, neq)
    @printf("  BC constrained DOFs: dof1=%d, dof2=%d\n",
            length(bc_per_dof[1]), length(bc_per_dof[2]))
    @printf("  norm(F_bc)=%.4e  norm(K_bc diagonal)=%.4e\n",
            norm(F_bc), norm(diag(K_bc)))
    @printf("  C size: %dx%d  nnz(C)=%d\n", size(C,1), size(C,2), nnz(C))
    @printf("  norm(C)=%.4e  norm(Z)=%.4e\n", norm(C), norm(Z))
    @printf("  norm(U)=%.4e  norm(Lam)=%.4e\n", norm(U), norm(Lam))
    @printf("  any(isnan,U)=%s  any(isnan,Lam)=%s\n",
            any(isnan, U), any(isnan, Lam))

    # Check interface CP physical coords
    npc1 = nnp[1]
    npc2 = nnp[2]
    # Patch 1 facet 3: η=n CPs
    from_P1 = [P_ref[1][a] for a in 1:npc1 if begin
        nc = nurbs_coords(a, npd, n_mat_ref[1,:])
        nc[2] == n_mat_ref[1,2]
    end]
    from_P2 = [P_ref[2][a] for a in 1:npc2 if begin
        nc = nurbs_coords(a, npd, n_mat_ref[2,:])
        nc[2] == 1
    end]
    @printf("  |P1 facet3 CPs|=%d, |P2 facet1 CPs|=%d\n",
            length(from_P1), length(from_P2))
    if length(from_P1) > 0
        @printf("  P1 facet3 y-range: [%.4f, %.4f]\n",
                minimum(B_ref[from_P1, 2]), maximum(B_ref[from_P1, 2]))
        @printf("  P1 facet3 r-range: [%.4f, %.4f]\n",
                minimum(sqrt.(B_ref[from_P1,1].^2 + B_ref[from_P1,2].^2)),
                maximum(sqrt.(B_ref[from_P1,1].^2 + B_ref[from_P1,2].^2)))
    end
    if length(from_P2) > 0
        @printf("  P2 facet1 y-range: [%.4f, %.4f]\n",
                minimum(B_ref[from_P2, 2]), maximum(B_ref[from_P2, 2]))
        @printf("  P2 facet1 r-range: [%.4f, %.4f]\n",
                minimum(sqrt.(B_ref[from_P2,1].^2 + B_ref[from_P2,2].^2)),
                maximum(sqrt.(B_ref[from_P2,1].^2 + B_ref[from_P2,2].^2)))
    end
    println()
end

println("=== Concentric cylinder diagnostics, p=2 ===")
for e in 0:2
    solve_cylinder_debug(2, e)
end
