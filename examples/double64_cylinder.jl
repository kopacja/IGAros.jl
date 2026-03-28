"""
Extended-precision verification: solve concentric cylinders with Double64.
Demonstrates that the convergence horizon is an arithmetic artifact.

Strategy: assemble in Float64 (normal IGAros), convert to dense Double64
for the linear solve, compute error in Float64.
"""

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

# Install DoubleFloats if needed
try
    using DoubleFloats
catch
    Pkg.add("DoubleFloats")
    using DoubleFloats
end

using IGAros
using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "concentric_cylinders.jl"))

function solve_cylinder_double64(
    p_ord::Int, exp_level::Int;
    conforming::Bool  = false,
    r_i::Float64 = 1.0, r_c::Float64 = 1.5, r_o::Float64 = 2.0,
    E::Float64 = 100.0, nu::Float64 = 0.3, p_o::Float64 = 1.0,
    epss::Float64 = 0.0,
    NQUAD::Int = p_ord + 1, NQUAD_mortar::Int = p_ord + 2,
    strategy::IntegrationStrategy = ElementBasedIntegration(),
    formulation::FormulationStrategy = TwinMortarFormulation(),
    n_ang_p2_base::Int = 3, n_ang_p1_base::Int = 6,
)
    nsd = 2; npd = 2; ned = 2; npc = 2; thickness = 1.0

    # ── Assembly (all Float64, identical to solve_cylinder) ────────────────
    B0, P = cylinder_geometry(p_ord; r_i=r_i, r_c=r_c, r_o=r_o)
    p_mat = fill(p_ord, npc, npd)
    n_mat = fill(p_ord + 1, npc, npd)
    KV    = generate_knot_vectors(npc, npd, p_mat, n_mat)

    s_ang    = 1.0 / (n_ang_p2_base * 2^exp_level)
    s_ang_nc = 1.0 / (n_ang_p1_base * 2^exp_level)
    s_rad    = (1/2) / 2^exp_level
    u_ang    = collect(s_ang    : s_ang    : 1 - s_ang/2)
    u_ang_nc = collect(s_ang_nc : s_ang_nc : 1 - s_ang_nc/2)
    u_rad    = collect(s_rad    : s_rad    : 1 - s_rad/2)
    epss_use = epss > 0.0 ? epss : 1e6

    kref_data = Vector{Float64}[
        vcat([1.0, 1.0], conforming ? u_ang : u_ang_nc),
        vcat([2.0, 1.0], u_ang),
        vcat([1.0, 2.0], u_rad),
        vcat([2.0, 2.0], u_rad),
    ]
    B0_hack = copy(B0);  B0_hack[P[1], 2] .+= 1000.0
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

    pairs = [InterfacePair(1, 3, 2, 1), InterfacePair(2, 1, 1, 3)]
    Pc = build_interface_cps(pairs, p_mat, n_mat_ref, KV_ref, P_ref, npd, nnp, formulation)
    C, Z = build_mortar_coupling(Pc, pairs, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
                                  ID, nnp, ned, nsd, npd, neq, NQUAD_mortar, epss_use,
                                  strategy, formulation)

    # ── Double64 solve ─────────────────────────────────────────────────────
    # Layout matches solve_mortar: [K C; C' -Z]
    # K is (neq×neq), C is (neq×nlm2), Z is (nlm2×nlm2)
    K64 = Matrix(K_bc); C64 = Matrix(C); Z64 = Matrix(Z)
    nlm2 = size(Z64, 1)

    # Float64 solve (dense, same as solve_mortar but without sparse)
    A64 = [K64 C64; C64' -Z64]
    b64 = [F_bc; zeros(nlm2)]
    x64 = A64 \ b64
    U64 = x64[1:neq]

    # Double64 solve
    Ad = Double64.(A64)
    bd = Double64.(b64)
    xd = Ad \ bd
    U  = Float64.(xd[1:neq])

    # ── Error computation ──────────────────────────────────────────────────
    s_abs_d, s_ref = l2_stress_error_cyl(
        U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
        nen, nel, IEN, INC, mats, NQUAD, thickness, stress_fn)

    s_abs_64, _ = l2_stress_error_cyl(
        U64, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
        nen, nel, IEN, INC, mats, NQUAD, thickness, stress_fn)

    en_abs_d, en_ref = energy_error_cyl(
        U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
        nen, nel, IEN, INC, mats, NQUAD, thickness, stress_fn)

    en_abs_64, _ = energy_error_cyl(
        U64, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
        nen, nel, IEN, INC, mats, NQUAD, thickness, stress_fn)

    return (
        stress_d64  = s_abs_d / s_ref,
        stress_f64  = s_abs_64 / s_ref,
        energy_d64  = en_abs_d / en_ref,
        energy_f64  = en_abs_64 / en_ref,
        neq         = neq,
        nlm         = nlm2,
    )
end

function run_double64_experiment()
    println("═"^80)
    println("  Double64 vs Float64: concentric cylinders convergence")
    println("  ε = 1e6, E = 100, non-conforming 2:1")
    println("  Double64 ε_mach ≈ $(eps(Double64))")
    println("═"^80)

    for p in [2, 3, 4]
        exp_max = p == 2 ? 3 : (p == 3 ? 3 : 3)
        println("\n── p = $p ──")
        @printf("  %3s  %6s  %14s  %14s  %8s  %14s  %14s  %8s\n",
                "exp", "neq", "energy F64", "energy D64", "rate D64",
                "stress F64", "stress D64", "rate D64")
        @printf("  %s\n", "─"^96)

        prev_en = NaN; prev_st = NaN
        for exp in 0:exp_max
            r = solve_cylinder_double64(p, exp; epss=1e6)
            rate_en = isnan(prev_en) ? NaN : log2(prev_en / r.energy_d64)
            rate_st = isnan(prev_st) ? NaN : log2(prev_st / r.stress_d64)
            @printf("  %3d  %6d  %14.6e  %14.6e  %8.2f  %14.6e  %14.6e  %8.2f\n",
                    exp, r.neq, r.energy_f64, r.energy_d64, rate_en,
                    r.stress_f64, r.stress_d64, rate_st)
            flush(stdout)
            prev_en = r.energy_d64; prev_st = r.stress_d64
        end
    end

    # ── Also test with larger ε (now safe with Double64) ───────────────────
    println("\n\n── ε-sweep at p=4, exp=2: Float64 vs Double64 ──")
    @printf("  %10s  %14s  %14s  %14s  %14s\n",
            "ε", "energy F64", "energy D64", "stress F64", "stress D64")
    @printf("  %s\n", "─"^68)
    for k in 2:10
        eps = 10.0^k
        r = solve_cylinder_double64(4, 2; epss=eps)
        @printf("  %10.0e  %14.6e  %14.6e  %14.6e  %14.6e\n",
                eps, r.energy_f64, r.energy_d64, r.stress_f64, r.stress_d64)
    end
end

run_double64_experiment()
