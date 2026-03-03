# cyl_debug6.jl — check if patches share CPs after krefinement
import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using IGAros, LinearAlgebra, Printf

include("concentric_cylinders.jl")

function check_cp_overlap(p_ord::Int, exp_level::Int)
    nsd=2; npd=2; ned=2; npc=2
    r_i=1.0; r_c=1.5; r_o=2.0; E=100.0; nu=0.3

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

    # Without offset hack — do patches merge?
    @printf("\n--- exp=%d (no y-offset hack) ---\n", exp_level)
    n_mat_ref_no, _, _, _, P_ref_no = krefinement(
        nsd, npd, npc, n_mat, p_mat, KV, B0, P, kref_data)
    shared_no = intersect(P_ref_no[1], P_ref_no[2])
    @printf("  P1∩P2 (no hack): %d shared CPs\n", length(shared_no))
    if !isempty(shared_no)
        @printf("  First 5 shared: %s\n", string(shared_no[1:min(5,end)]))
    end

    # With offset hack — do patches share CPs?
    B0_hack = copy(B0)
    B0_hack[P[1], 2] .+= 1000.0
    n_mat_ref, _, KV_ref, B_hack_ref, P_ref = krefinement(
        nsd, npd, npc, n_mat, p_mat, KV, B0_hack, P, kref_data)
    shared = intersect(P_ref[1], P_ref[2])
    @printf("  P1∩P2 (with hack): %d shared CPs\n", length(shared))
    if !isempty(shared)
        @printf("  First 5 shared: %s\n", string(shared[1:min(5,end)]))
        # Print physical coords of these shared CPs
        for cp in shared[1:min(3,end)]
            @printf("    CP %d: x=%.4f y=%.4f z=%.4f w=%.4f\n", cp,
                    B_hack_ref[cp,1], B_hack_ref[cp,2], B_hack_ref[cp,3], B_hack_ref[cp,4])
        end
    end

    # Check K sparsity — are there off-diagonal P1×P2 entries?
    B_ref = copy(B_hack_ref)
    for i in axes(B_ref, 1); B_ref[i, 2] > 100.0 && (B_ref[i, 2] -= 1000.0); end
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
                                   nen, nel, IEN, INC, LM, mats, p_ord+1, 1.0)

    # Which equation numbers belong to Patch 1 vs Patch 2?
    eq_P1 = Set{Int}()
    eq_P2 = Set{Int}()
    for cp in P_ref[1], i in 1:ned
        eq = ID[i, cp]; eq != 0 && push!(eq_P1, eq)
    end
    for cp in P_ref[2], i in 1:ned
        eq = ID[i, cp]; eq != 0 && push!(eq_P2, eq)
    end
    eq_shared = intersect(eq_P1, eq_P2)
    @printf("  DOF overlap: eq_P1=%d  eq_P2=%d  shared=%d\n",
            length(eq_P1), length(eq_P2), length(eq_shared))

    # Check for cross-patch K entries
    I_K, J_K, V_K = findnz(K)
    cross_K = sum(1 for (i, j) in zip(I_K, J_K)
                  if (i in eq_P1 && j in eq_P2) || (i in eq_P2 && j in eq_P1))
    @printf("  K cross-patch nonzeros: %d\n", cross_K)
    if cross_K == 0
        @printf("  -> K IS block-diagonal (patches fully decoupled)\n")
    else
        @printf("  -> K has cross-patch entries! Patches are coupled in K.\n")
        # Find max cross-patch K entry
        cross_vals = [abs(v) for (i,j,v) in zip(I_K,J_K,V_K)
                      if (i in eq_P1 && j in eq_P2) || (i in eq_P2 && j in eq_P1)]
        @printf("  Max |K_cross| = %.4e\n", maximum(cross_vals))
    end
end

check_cp_overlap(2, 0)
check_cp_overlap(2, 1)
