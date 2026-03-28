# examples/cz_cancellation.jl
#
# Direct numerical test of the C–Z integration-error cancellation mechanism
# described in §4.5 of the Twin Mortar paper.
#
# Setup: flat 2D non-conforming patch test, p=1.
#   Patch 1: [0, 0.5]×[0,1]  with n_s interface elements (slave side)
#   Patch 2: [0.5,1]×[0,1]   with n_m interface elements (master side)
#   Exact solution: u_x = x/E, u_y = 0  (uniaxial tension, ν=0)
#
# For p=1 on a flat interface the ONLY source of integration error is the
# kinks in the piecewise-linear shape functions at the non-matching knots.
# Segment-based integration resolves all kinks exactly → reference solution.
# Element-based integration misses the kinks → quadrature error in M.
#
# The test compares four configurations:
#   TM-Seg  : Twin Mortar   + segment-based  (exact integration → reference)
#   TM-Elm  : Twin Mortar   + element-based   (quadrature error in both C and Z)
#   SP-Seg  : Single-pass   + segment-based  (exact integration → reference)
#   SP-Elm  : Single-pass   + element-based   (quadrature error in C only)

import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using IGAros
using LinearAlgebra, SparseArrays, Printf

# ─────────────────────── Geometry builder ──────────────────────────────────

"""
    flat_patch_test(n_s, n_m; E, epss, NQUAD_mortar, strategy, formulation)

Build and solve a flat 2D non-conforming p=1 patch test.

- `n_s` : number of interface elements on slave  (Patch 1 η-direction)
- `n_m` : number of interface elements on master (Patch 2 η-direction)

Returns `(U, C, Z, K_bc, F_bc, ID, B, ncp, neq, Pc)`.
"""
function flat_patch_test(
    n_s::Int, n_m::Int;
    E::Float64        = 1000.0,
    nu::Float64       = 0.0,
    epss::Float64     = 1.0,
    NQUAD_mortar::Int = 3,
    strategy::IntegrationStrategy    = ElementBasedIntegration(),
    formulation::FormulationStrategy = TwinMortarFormulation(),
)
    nsd = 2; npd = 2; ned = 2; npc = 2

    # ── Knot vectors ───────────────────────────────────────────────────────
    # Patch 1: p=1, n_ξ=2 (one element in x), n_η=n_s+1
    # Patch 2: p=1, n_ξ=2 (one element in x), n_η=n_m+1
    p = [1 1; 1 1]
    n = [2 n_s+1; 2 n_m+1]

    KV = generate_knot_vectors(npc, npd, p, n)

    # ── Control points ─────────────────────────────────────────────────────
    # Patch 1: [0,0.5] × [0,1], CPs on a regular n_ξ × n_η grid
    ncp1 = 2 * (n_s + 1)
    ncp2 = 2 * (n_m + 1)
    ncp  = ncp1 + ncp2

    B = zeros(ncp, 4)
    idx = 0
    for j in 0:n_s, i in 0:1   # fast ξ, slow η
        idx += 1
        B[idx, :] = [i * 0.5, j / n_s, 0.0, 1.0]
    end
    for j in 0:n_m, i in 0:1
        idx += 1
        B[idx, :] = [0.5 + i * 0.5, j / n_m, 0.0, 1.0]
    end

    P = [collect(1:ncp1), collect(ncp1+1:ncp)]

    # ── Connectivity ───────────────────────────────────────────────────────
    nel, nnp, nen = patch_metrics(npc, npd, p, n)
    IEN = build_ien(nsd, npd, npc, p, n, nel, nnp, nen)
    INC = [build_inc(n[pc, :]) for pc in 1:npc]

    mat = LinearElastic(E, nu, :plane_strain)

    # ── Boundary conditions ────────────────────────────────────────────────
    # u_x = 0 on left face of Patch 1 (facet 4, ξ=1): CPs with x=0
    # u_y = 0 on bottom of both patches (facet 1, η=1): CPs with y=0
    bc_ux = Int[i for i in 1:ncp1 if B[i, 1] ≈ 0.0]
    bc_uy = Int[i for i in 1:ncp  if B[i, 2] ≈ 0.0]
    bc_per_dof = [bc_ux, bc_uy]
    neq, ID = build_id(bc_per_dof, ned, ncp)
    LM = build_lm(nen, ned, npc, nel, ID, IEN, P)

    # ── Stiffness ──────────────────────────────────────────────────────────
    Ub0 = zeros(ncp, nsd)
    K, _ = build_stiffness_matrix(npc, nsd, npd, ned, neq,
                                   p, n, KV, P, B, Ub0,
                                   nen, nel, IEN, INC, LM,
                                   [mat, mat], 2, 1.0)

    # ── Load: unit traction in x on right face of Patch 2 ─────────────────
    F = zeros(neq)
    F = segment_load(n[2,:], p[2,:], KV[2], P[2], B,
                     nnp[2], nen[2], nsd, npd, ned,
                     Int[], 2, ID, F, 1.0, 1.0, 2)

    K_bc, F_bc = enforce_dirichlet(Tuple{Int,Float64}[], K, F)

    # ── Mortar coupling ───────────────────────────────────────────────────
    # Interface: right face of Patch 1 (facet 2) ↔ left face of Patch 2 (facet 4)
    pairs_tm = [InterfacePair(1, 2, 2, 4), InterfacePair(2, 4, 1, 2)]
    pairs_sp = [InterfacePair(1, 2, 2, 4)]
    pairs = formulation isa SinglePassFormulation ? pairs_sp : pairs_tm

    Pc = build_interface_cps(pairs, p, n, KV, P, npd, nnp, formulation)
    C, Z = build_mortar_coupling(Pc, pairs, p, n, KV, P, B, ID, nnp,
                                  ned, nsd, npd, neq, NQUAD_mortar, epss,
                                  strategy, formulation)

    U, Lambda = solve_mortar(K_bc, C, Z, F_bc)

    return (U=U, C=C, Z=Z, K_bc=K_bc, F_bc=F_bc, ID=ID, B=B, ncp=ncp,
            neq=neq, Pc=Pc, E=E)
end

# ─────────────────────── Higher-order flat patch test ─────────────────────

"""
    flat_patch_test_hp(p_ord, n_s, n_m; E, epss, NQUAD_mortar, strategy, formulation)

Build and solve a flat 2D non-conforming patch test at polynomial order `p_ord`.
Uses k-refinement from a single-element p-order geometry.

- `p_ord` : polynomial order (1, 2, 3, 4)
- `n_s`   : number of interface elements on slave  (Patch 1 η-direction)
- `n_m`   : number of interface elements on master (Patch 2 η-direction)
"""
function flat_patch_test_hp(
    p_ord::Int, n_s::Int, n_m::Int;
    E::Float64        = 1000.0,
    nu::Float64       = 0.0,
    epss::Float64     = 1.0,
    NQUAD_mortar::Int = p_ord + 2,
    strategy::IntegrationStrategy    = ElementBasedIntegration(),
    formulation::FormulationStrategy = TwinMortarFormulation(),
)
    p_ord == 1 && return flat_patch_test(n_s, n_m; E=E, nu=nu, epss=epss,
                                          NQUAD_mortar=NQUAD_mortar,
                                          strategy=strategy, formulation=formulation)
    nsd = 2; npd = 2; ned = 2; npc = 2

    # ── Initial single-element geometry at order p_ord ────────────────────
    # Each patch: 1 element in ξ (x-dir), 1 element in η (y-dir)
    # CPs on a regular (p+1) × (p+1) grid, unit weights
    p_mat = fill(p_ord, npc, npd)
    n_init = fill(p_ord + 1, npc, npd)
    KV_init = generate_knot_vectors(npc, npd, p_mat, n_init)

    ncp_p1 = (p_ord + 1)^2  # CPs per patch (single element)
    B0 = zeros(2 * ncp_p1, 4)
    idx = 0
    # Patch 1: [0, 0.5] × [0, 1]
    for j in 0:p_ord, i in 0:p_ord
        idx += 1
        B0[idx, :] = [i * 0.5 / p_ord, j / p_ord, 0.0, 1.0]
    end
    # Patch 2: [0.5, 1] × [0, 1]
    for j in 0:p_ord, i in 0:p_ord
        idx += 1
        B0[idx, :] = [0.5 + i * 0.5 / p_ord, j / p_ord, 0.0, 1.0]
    end

    P_init = [collect(1:ncp_p1), collect(ncp_p1+1:2*ncp_p1)]

    # ── h-refinement via krefinement ──────────────────────────────────────
    # Insert knots to get n_s elements in Patch 1 η and n_m in Patch 2 η
    s_s  = 1.0 / n_s
    s_m  = 1.0 / n_m
    u_s  = collect(s_s : s_s : 1 - s_s/2)
    u_m  = collect(s_m : s_m : 1 - s_m/2)

    kref_data = Vector{Float64}[
        vcat([1.0, 2.0], u_s),  # Patch 1 η (interface dir)
        vcat([2.0, 2.0], u_m),  # Patch 2 η (interface dir)
    ]

    # y-offset hack: patches share y-coordinates at interface (x=0.5)
    B0_hack = copy(B0)
    B0_hack[P_init[1], 2] .+= 1000.0

    n_mat, _, KV, B_hack, P = krefinement(
        nsd, npd, npc, n_init, p_mat, KV_init, B0_hack, P_init, kref_data)

    B = copy(B_hack)
    for i in axes(B, 1)
        B[i, 2] > 100.0 && (B[i, 2] -= 1000.0)
    end

    ncp = size(B, 1)

    # ── Connectivity ──────────────────────────────────────────────────────
    nel, nnp, nen = patch_metrics(npc, npd, p_mat, n_mat)
    IEN = build_ien(nsd, npd, npc, p_mat, n_mat, nel, nnp, nen)
    INC = [build_inc(n_mat[pc, :]) for pc in 1:npc]

    mat = LinearElastic(E, nu, :plane_strain)

    # ── Boundary conditions ───────────────────────────────────────────────
    bc_ux = Int[i for i in 1:ncp if B[i, 1] ≈ 0.0]
    bc_uy = Int[i for i in 1:ncp if B[i, 2] ≈ 0.0]
    bc_per_dof = [bc_ux, bc_uy]
    neq, ID = build_id(bc_per_dof, ned, ncp)
    LM = build_lm(nen, ned, npc, nel, ID, IEN, P)

    # ── Stiffness ─────────────────────────────────────────────────────────
    Ub0 = zeros(ncp, nsd)
    K, _ = build_stiffness_matrix(npc, nsd, npd, ned, neq,
                                   p_mat, n_mat, KV, P, B, Ub0,
                                   nen, nel, IEN, INC, LM,
                                   [mat, mat], p_ord + 1, 1.0)

    # ── Load: unit traction in x on right face of Patch 2 ────────────────
    F = zeros(neq)
    F = segment_load(n_mat[2,:], p_mat[2,:], KV[2], P[2], B,
                     nnp[2], nen[2], nsd, npd, ned,
                     Int[], 2, ID, F, 1.0, 1.0, p_ord + 1)

    K_bc, F_bc = enforce_dirichlet(Tuple{Int,Float64}[], K, F)

    # ── Mortar coupling ──────────────────────────────────────────────────
    pairs_tm = [InterfacePair(1, 2, 2, 4), InterfacePair(2, 4, 1, 2)]
    pairs_sp = [InterfacePair(1, 2, 2, 4)]
    pairs = formulation isa SinglePassFormulation ? pairs_sp : pairs_tm

    Pc = build_interface_cps(pairs, p_mat, n_mat, KV, P, npd, nnp, formulation)
    C, Z = build_mortar_coupling(Pc, pairs, p_mat, n_mat, KV, P, B, ID, nnp,
                                  ned, nsd, npd, neq, NQUAD_mortar, epss,
                                  strategy, formulation)

    U, Lambda = solve_mortar(K_bc, C, Z, F_bc)

    return (U=U, C=C, Z=Z, K_bc=K_bc, F_bc=F_bc, ID=ID, B=B, ncp=ncp,
            neq=neq, Pc=Pc, E=E)
end

# ─────────────────────── Displacement error vs exact ─────────────────────

function disp_error(U, ID, B, ncp, E)
    err2 = 0.0; ref2 = 0.0
    for A in 1:ncp
        ux_ex = B[A, 1] / E
        ux    = ID[1, A] != 0 ? U[ID[1, A]] : 0.0
        err2 += (ux - ux_ex)^2
        ref2 += ux_ex^2
    end
    return sqrt(err2 / ref2)
end

# ─────────────────────── First-order perturbation analysis ────────────────

"""
    run_perturbation_analysis(n_s, n_m; nquad_range, epss)

Perturbation analysis of the full KKT system (avoids forming Z⁻¹,
which is ill-conditioned due to the near-null space of Z).

The code's KKT system is
    A x = b,    A = [K  C; Cᵀ  −Z],   x = [U; λ],   b = [F; 0].

Perturbing C → C+δC, Z → Z+δZ gives δA = [0 δC; δCᵀ −δZ], and
    δx ≈ −A⁻¹ δA x.

We decompose δA into three independent contributions:
    δA_C  = [0 δC; δCᵀ 0]     (C perturbation only)
    δA_Z  = [0  0;  0  −δZ]   (Z perturbation only)

and measure:
    δx_C  = −A⁻¹ δA_C x       (solution change from C error alone)
    δx_Z  = −A⁻¹ δA_Z x       (solution change from Z error alone)
    δx    = −A⁻¹ δA x          (total first-order change)

If the C and Z errors cancel:  ‖δx‖ < ‖δx_C‖ + ‖δx_Z‖.
The cosine between δx_C and δx_Z reveals the direction.
"""
function run_perturbation_analysis(
    n_s::Int = 2,
    n_m::Int = 3;
    nquad_range = 1:6,
    epss::Float64 = 1.0,
)
    E = 1000.0

    # ── Reference: segment-based (exact) ─────────────────────────────────
    r = flat_patch_test(n_s, n_m; E=E, epss=epss,
                        strategy=SegmentBasedIntegration(),
                        formulation=TwinMortarFormulation())
    Cd = Matrix(r.C);  Zd = Matrix(r.Z)
    Kd = Matrix(r.K_bc)
    neq = r.neq;  nm = size(Zd, 1)
    A_ref = [Kd Cd; Cd' -Zd]
    b = [r.F_bc; zeros(nm)]
    x_ref = A_ref \ b   # [U_ref; λ_ref]

    println("\n=== KKT perturbation analysis: p=1, n_s=$n_s, n_m=$n_m, ε=$epss ===")
    @printf("%-6s  %10s  %10s  %10s  %10s  %10s  %10s  %10s\n",
            "NQUAD", "‖δC‖/‖C‖", "‖δZ‖/‖Z‖",
            "‖δU_C‖", "‖δU_Z‖", "cos(δU)", "‖δU_tot‖", "‖δU_act‖")
    @printf("%s\n", "-"^88)

    Ai = A_ref \ I(size(A_ref, 1))   # A⁻¹ (dense, small system)

    for nq in nquad_range
        e = flat_patch_test(n_s, n_m; E=E, epss=epss, NQUAD_mortar=nq,
                            strategy=ElementBasedIntegration(),
                            formulation=TwinMortarFormulation())
        Ce = Matrix(e.C);  Ze = Matrix(e.Z)
        dC = Ce - Cd;  dZ = Ze - Zd

        # ── KKT perturbation matrices ────────────────────────────────────
        dA_C = [zeros(neq,neq) dC; dC' zeros(nm,nm)]         # C error only
        dA_Z = [zeros(neq,neq) zeros(neq,nm);
                zeros(nm,neq) -dZ]                             # Z error only
        dA   = dA_C + dA_Z                                    # total

        # ── First-order responses ────────────────────────────────────────
        dx_C = -Ai * (dA_C * x_ref)   # from C perturbation
        dx_Z = -Ai * (dA_Z * x_ref)   # from Z perturbation
        dx   = -Ai * (dA   * x_ref)   # total first-order

        # Extract displacement part only (first neq entries)
        dU_C = dx_C[1:neq]
        dU_Z = dx_Z[1:neq]
        dU   = dx[1:neq]

        # Actual solution difference
        A_elm = [Kd Ce; Ce' -Ze]
        x_elm = A_elm \ b
        dU_actual = x_elm[1:neq] - x_ref[1:neq]

        nC = norm(dU_C);  nZ = norm(dU_Z)
        cos_angle = (nC > 1e-20 && nZ > 1e-20) ?
            dot(dU_C, dU_Z) / (nC * nZ) : 0.0

        @printf("%-6d  %10.2e  %10.2e  %10.2e  %10.2e  %+10.4f  %10.2e  %10.2e\n",
                nq, norm(dC)/norm(Cd), norm(dZ)/norm(Zd),
                nC, nZ, cos_angle, norm(dU), norm(dU_actual))
    end

    # ── Detailed breakdown for NQUAD=1 and NQUAD=2 ──────────────────────
    for nq_detail in [1, min(2, maximum(nquad_range))]
        nq_detail > maximum(nquad_range) && continue
        e = flat_patch_test(n_s, n_m; E=E, epss=epss, NQUAD_mortar=nq_detail,
                            strategy=ElementBasedIntegration(),
                            formulation=TwinMortarFormulation())
        Ce = Matrix(e.C);  Ze = Matrix(e.Z)
        dC = Ce - Cd;  dZ = Ze - Zd

        dA_C = [zeros(neq,neq) dC; dC' zeros(nm,nm)]
        dA_Z = [zeros(neq,neq) zeros(neq,nm); zeros(nm,neq) -dZ]
        dA   = dA_C + dA_Z

        dx_C = -Ai * (dA_C * x_ref)
        dx_Z = -Ai * (dA_Z * x_ref)
        dx   = -Ai * (dA   * x_ref)

        dU_C = dx_C[1:neq]; dU_Z = dx_Z[1:neq]; dU = dx[1:neq]
        A_elm = [Kd Ce; Ce' -Ze]
        dU_actual = (A_elm \ b)[1:neq] - x_ref[1:neq]

        nC = norm(dU_C); nZ = norm(dU_Z)
        cos_angle = (nC > 1e-20 && nZ > 1e-20) ?
            dot(dU_C, dU_Z) / (nC * nZ) : 0.0

        println("\n--- Detailed breakdown at NQUAD=$nq_detail ---")
        @printf("‖δC‖/‖C‖        = %10.4e\n", norm(dC)/norm(Cd))
        @printf("‖δZ‖/‖Z‖        = %10.4e\n", norm(dZ)/norm(Zd))
        @printf("‖δU_C‖ (C only)  = %10.4e\n", nC)
        @printf("‖δU_Z‖ (Z only)  = %10.4e\n", nZ)
        @printf("‖δU_C + δU_Z‖    = %10.4e   (sum of individual)\n", norm(dU_C + dU_Z))
        @printf("‖δU_C‖ + ‖δU_Z‖  = %10.4e   (triangle bound)\n", nC + nZ)
        @printf("‖δU_total‖       = %10.4e   (first-order, joint)\n", norm(dU))
        @printf("‖δU_actual‖      = %10.4e   (true KKT difference)\n", norm(dU_actual))
        @printf("cos(δU_C, δU_Z)  = %+.6f", cos_angle)
        if cos_angle > 0.1
            println("   → REINFORCE (errors add up)")
        elseif cos_angle < -0.1
            println("   → CANCEL (errors partially cancel)")
        else
            println("   → ORTHOGONAL")
        end
        # Cancellation ratio: how much smaller is ‖δU‖ than the triangle bound?
        if nC + nZ > 0
            @printf("Cancellation ratio: ‖δU‖/(‖δU_C‖+‖δU_Z‖) = %.4f\n",
                    norm(dU) / (nC + nZ))
        end
    end

    # ── Single-pass comparison ───────────────────────────────────────────
    # For SP: C contains cross-mesh M, Z=0.  δC carries the full integration
    # error and there is no Z to absorb it.
    println("\n--- Single-pass comparison ---")
    r_sp = try
        flat_patch_test(n_s, n_m; E=E, epss=epss,
                        strategy=SegmentBasedIntegration(),
                        formulation=SinglePassFormulation())
    catch; nothing; end
    r_sp === nothing && (println("SP-Seg singular — skipped"); return)

    Cd_sp = Matrix(r_sp.C); Zd_sp = Matrix(r_sp.Z)  # Z=0 for SP
    Kd_sp = Matrix(r_sp.K_bc); nm_sp = size(Zd_sp, 1)
    A_sp = [Kd_sp Cd_sp; Cd_sp' -Zd_sp]
    b_sp = [r_sp.F_bc; zeros(nm_sp)]
    x_sp = A_sp \ b_sp
    Ai_sp = A_sp \ I(size(A_sp, 1))

    for nq_detail in [1, 2]
        nq_detail > maximum(nquad_range) && continue
        e_sp = try
            flat_patch_test(n_s, n_m; E=E, epss=epss, NQUAD_mortar=nq_detail,
                            strategy=ElementBasedIntegration(),
                            formulation=SinglePassFormulation())
        catch; nothing; end
        e_sp === nothing && (@printf("SP-Elm NQUAD=%d singular\n", nq_detail); continue)

        Ce_sp = Matrix(e_sp.C); Ze_sp = Matrix(e_sp.Z)
        dC_sp = Ce_sp - Cd_sp

        dA_C_sp = [zeros(r_sp.neq, r_sp.neq) dC_sp;
                    dC_sp' zeros(nm_sp, nm_sp)]
        dx_C_sp = -Ai_sp * (dA_C_sp * x_sp)
        dU_C_sp = dx_C_sp[1:r_sp.neq]

        A_sp_elm = [Kd_sp Ce_sp; Ce_sp' -Ze_sp]
        dU_sp_actual = (A_sp_elm \ b_sp)[1:r_sp.neq] - x_sp[1:r_sp.neq]

        println("\nSP at NQUAD=$nq_detail:")
        @printf("  ‖δC‖/‖C‖    = %10.4e\n", norm(dC_sp)/norm(Cd_sp))
        @printf("  ‖δU_C‖      = %10.4e   (C perturbation → displacement)\n", norm(dU_C_sp))
        @printf("  ‖δU_actual‖ = %10.4e   (true KKT difference)\n", norm(dU_sp_actual))
    end
end

# ─────────────────────── Main cancellation study ─────────────────────────

"""
    run_cancellation_study(n_s, n_m; nquad_range, epss)

For a flat p=1 patch test with `n_s` vs `n_m` interface elements,
compare segment-based (exact) vs element-based (approximate) integration
for both Twin Mortar and Single-Pass formulations.

Reports:
- Matrix-level errors: δC, δZ (Frobenius norm, element- vs segment-based)
- Solution-level errors: displacement error vs exact (u_x = x/E)
"""
function run_cancellation_study(
    n_s::Int = 2,
    n_m::Int = 3;
    nquad_range = 1:6,
    epss::Float64 = 1.0,
)
    E = 1000.0

    configs = [
        ("TM-Seg", TwinMortarFormulation(),  SegmentBasedIntegration()),
        ("TM-Elm", TwinMortarFormulation(),  ElementBasedIntegration()),
        ("SP-Seg", SinglePassFormulation(),  SegmentBasedIntegration()),
        ("SP-Elm", SinglePassFormulation(),  ElementBasedIntegration()),
    ]

    # ── References: segment-based (exact for flat p=1) ───────────────────
    r_tm = flat_patch_test(n_s, n_m; E=E, epss=epss,
                           strategy=SegmentBasedIntegration(),
                           formulation=TwinMortarFormulation())
    r_sp = try
        flat_patch_test(n_s, n_m; E=E, epss=epss,
                        strategy=SegmentBasedIntegration(),
                        formulation=SinglePassFormulation())
    catch e
        e isa SingularException || rethrow()
        @warn "SP-Seg singular for n_s=$n_s, n_m=$n_m — skipping SP"
        nothing
    end

    nC_tm = norm(r_tm.C)
    nZ_tm = norm(r_tm.Z)

    println("\n=== C–Z cancellation study: p=1 flat interface, n_s=$n_s, n_m=$n_m, ε=$epss ===")
    err_tm_seg = disp_error(r_tm.U, r_tm.ID, r_tm.B, r_tm.ncp, E)
    @printf("TM-Seg displacement error vs exact: %.4e\n", err_tm_seg)
    if r_sp !== nothing
        err_sp_seg = disp_error(r_sp.U, r_sp.ID, r_sp.B, r_sp.ncp, E)
        @printf("SP-Seg displacement error vs exact: %.4e\n", err_sp_seg)
    end

    # ── Table 1: Matrix-level errors (TM element-based vs segment-based) ─
    println("\n--- Matrix-level errors: TM element-based vs TM segment-based ---")
    @printf("%-6s  %12s  %12s\n", "NQUAD", "δC_TM", "δZ_TM")
    @printf("%s\n", "-"^34)

    for nq in nquad_range
        e_tm = flat_patch_test(n_s, n_m; E=E, epss=epss, NQUAD_mortar=nq,
                               strategy=ElementBasedIntegration(),
                               formulation=TwinMortarFormulation())
        dC = norm(e_tm.C - r_tm.C) / nC_tm
        dZ = norm(e_tm.Z - r_tm.Z) / nZ_tm
        @printf("%-6d  %12.4e  %12.4e\n", nq, dC, dZ)
    end

    # ── Table 2: Solution error vs exact for all 4 configs ───────────────
    println("\n--- Displacement error vs exact solution (u_x = x/E) ---")
    header = "NQUAD"
    for (label, _, _) in configs
        header *= @sprintf("  %12s", label)
    end
    println(header)
    @printf("%s\n", "-"^(6 + 14 * length(configs)))

    for nq in nquad_range
        @printf("%-6d", nq)
        for (label, form, strat) in configs
            err = try
                r = flat_patch_test(n_s, n_m; E=E, epss=epss, NQUAD_mortar=nq,
                                    strategy=strat, formulation=form)
                disp_error(r.U, r.ID, r.B, r.ncp, E)
            catch e
                e isa SingularException || rethrow()
                NaN
            end
            @printf("  %12.4e", err)
        end
        println()
    end

    # ── Table 3: Quadrature-induced error (element- vs segment-based ref) ─
    println("\n--- Quadrature-induced error: ‖U_elm - U_seg‖ / ‖U_seg‖ ---")
    @printf("%-6s  %12s", "NQUAD", "TM")
    r_sp !== nothing && @printf("  %12s", "SP")
    println()
    @printf("%s\n", "-"^(r_sp !== nothing ? 34 : 20))

    for nq in nquad_range
        e_tm = flat_patch_test(n_s, n_m; E=E, epss=epss, NQUAD_mortar=nq,
                               strategy=ElementBasedIntegration(),
                               formulation=TwinMortarFormulation())
        dU_tm = norm(e_tm.U - r_tm.U) / norm(r_tm.U)
        @printf("%-6d  %12.4e", nq, dU_tm)

        if r_sp !== nothing
            dU_sp = try
                e_sp = flat_patch_test(n_s, n_m; E=E, epss=epss, NQUAD_mortar=nq,
                                       strategy=ElementBasedIntegration(),
                                       formulation=SinglePassFormulation())
                norm(e_sp.U - r_sp.U) / norm(r_sp.U)
            catch e
                e isa SingularException || rethrow()
                NaN
            end
            @printf("  %12.4e", dU_sp)
        end
        println()
    end
end

"""
    run_cancellation_mesh_sweep(; mesh_configs, nquad_range, epss)

Run cancellation study for several non-conforming mesh ratios to show
that the effect is robust and scales with the number of kinks.
"""
function run_cancellation_mesh_sweep(;
    mesh_configs = [(2, 3), (3, 5), (4, 7), (6, 10)],
    nquad_range  = 1:4,
    epss::Float64 = 1.0,
)
    for (n_s, n_m) in mesh_configs
        run_cancellation_study(n_s, n_m; nquad_range=nquad_range, epss=epss)
        println()
    end
end

# ─────────────────────── Partition-of-unity violation test ────────────────────

"""
    run_pou_test(n_s, n_m; NQUAD_range, epss)

Check whether D and M reproduce constant and linear fields under
element-based vs segment-based integration.

For a single interface pair (slave→master):
  D_{AB} = ∫_Γ R^s_A R^s_B dΓ       (slave-slave, exact under element-based)
  M_{AB} = ∫_Γ R^s_A R^m_B dΓ       (slave-master, has kinks under element-based)

Partition-of-unity test (constant field d=1):
  D·1_s should equal M·1_m  (because ΣR^s = ΣR^m = 1 pointwise)

Linear reproduction test (field u=y on the interface):
  D·y_s should equal M·y_m  (both represent the same linear field)

For SP: the constraint D·d_s = M·d_m must hold exactly.
Any violation ‖D·d_s - M·d_m‖ is the constraint residual that
directly corrupts the displacement.
"""
function run_pou_test(
    n_s::Int = 4,
    n_m::Int = 7;
    NQUAD_range = 1:6,
)
    nsd = 2; npd = 2; ned = 2; npc = 2

    # ── Build mesh infrastructure (same as flat_patch_test) ──────────────
    p = [1 1; 1 1]
    n = [2 n_s+1; 2 n_m+1]
    KV = generate_knot_vectors(npc, npd, p, n)

    ncp1 = 2 * (n_s + 1)
    ncp2 = 2 * (n_m + 1)
    ncp  = ncp1 + ncp2

    B = zeros(ncp, 4)
    idx = 0
    for j in 0:n_s, i in 0:1
        idx += 1
        B[idx, :] = [i * 0.5, j / n_s, 0.0, 1.0]
    end
    for j in 0:n_m, i in 0:1
        idx += 1
        B[idx, :] = [0.5 + i * 0.5, j / n_m, 0.0, 1.0]
    end

    P = [collect(1:ncp1), collect(ncp1+1:ncp)]
    nel, nnp, nen = patch_metrics(npc, npd, p, n)

    # ── Interface pair: Patch 1 facet 2 (x=0.5) → Patch 2 facet 4 (x=0.5)
    pair = InterfacePair(1, 2, 2, 4)

    println("\n=== Partition-of-unity / linear reproduction test ===")
    println("p=1, n_s=$n_s slave elements, n_m=$n_m master elements")
    println("Interface at x=0.5, y ∈ [0,1]\n")

    @printf("%-6s  %12s  %12s  %12s  %12s  %12s\n",
            "NQUAD", "‖(D·1-M·1)‖", "‖(D·y-M·y)‖",
            "‖δD‖/‖D‖", "‖δM‖/‖M‖", "‖D⁻¹M·y-y‖")
    @printf("%s\n", "-"^78)

    # ── Reference: high-NQUAD element-based (exact for flat p=1) ────────
    D_ref, M_ref, s_cps, m_cps = build_mortar_mass_matrices(
        pair, p, n, KV, P, B, nnp, nsd, npd, 20, ElementBasedIntegration())

    # Build field vectors for interface CPs
    ones_s = ones(length(s_cps))
    ones_m = ones(length(m_cps))
    y_s = [B[cp, 2] for cp in s_cps]   # y-coordinates of slave interface CPs
    y_m = [B[cp, 2] for cp in m_cps]   # y-coordinates of master interface CPs

    D_ref_dense = Matrix(D_ref)
    M_ref_dense = Matrix(M_ref)

    @printf("  ref   %12.2e  %12.2e  %12s  %12s  %12.2e\n",
            norm(D_ref_dense * ones_s - M_ref_dense * ones_m),
            norm(D_ref_dense * y_s - M_ref_dense * y_m),
            "—", "—",
            norm(D_ref_dense \ (M_ref_dense * y_m) - y_s))

    for nq in NQUAD_range
        D_elm, M_elm, _, _ = build_mortar_mass_matrices(
            pair, p, n, KV, P, B, nnp, nsd, npd, nq, ElementBasedIntegration())

        Dd = Matrix(D_elm)
        Md = Matrix(M_elm)

        pou_const = norm(Dd * ones_s - Md * ones_m)
        pou_lin   = norm(Dd * y_s - Md * y_m)
        dD = norm(Dd - D_ref_dense) / norm(D_ref_dense)
        dM = norm(Md - M_ref_dense) / norm(M_ref_dense)

        # D⁻¹M maps master field to slave: for exact integration, D⁻¹M·y_m = y_s
        proj_err = try
            norm(Dd \ (Md * y_m) - y_s)
        catch
            NaN
        end

        @printf("%-6d  %12.2e  %12.2e  %12.2e  %12.2e  %12.2e\n",
                nq, pou_const, pou_lin, dD, dM, proj_err)
    end

    # ── Show D⁻¹M·y_m pointwise for NQUAD=2 ────────────────────────────
    println("\n--- Pointwise D⁻¹M·y_m vs y_s at NQUAD=2 (element-based) ---")
    D1, M1, _, _ = build_mortar_mass_matrices(
        pair, p, n, KV, P, B, nnp, nsd, npd, 2, ElementBasedIntegration())
    proj_y = Matrix(D1) \ (Matrix(M1) * y_m)
    @printf("%-4s  %10s  %10s  %12s\n", "CP", "y_s", "D⁻¹M·y_m", "error")
    @printf("%s\n", "-"^40)
    for (i, cp) in enumerate(s_cps)
        @printf("%-4d  %10.6f  %10.6f  %12.4e\n", cp, y_s[i], proj_y[i], proj_y[i] - y_s[i])
    end

    # ── Same for segment-based (should be exact) ────────────────────────
    println("\n--- Pointwise D⁻¹M·y_m vs y_s at segment-based (reference) ---")
    proj_y_ref = D_ref_dense \ (M_ref_dense * y_m)
    @printf("%-4s  %10s  %10s  %12s\n", "CP", "y_s", "D⁻¹M·y_m", "error")
    @printf("%s\n", "-"^40)
    for (i, cp) in enumerate(s_cps)
        @printf("%-4d  %10.6f  %10.6f  %12.4e\n", cp, y_s[i], proj_y_ref[i], proj_y_ref[i] - y_s[i])
    end

    # ── Check: does x-field (the actual patch test direction) also reproduce? ─
    # The patch test is u_x = x, u_y = 0.  On the interface x=0.5, so u_x=0.5/E.
    # All interface CPs have x=0.5, so d_s = d_m = [0.5, 0.5, ...] (constant!)
    # A constant field is always reproduced → SP-Elm passes the p=1 flat patch test.
    # The failure must come from curved interfaces or non-constant fields.
    println("\n--- Key insight ---")
    x_s = [B[cp, 1] for cp in s_cps]
    x_m = [B[cp, 1] for cp in m_cps]
    println("Slave interface x-coords:  ", round.(x_s, digits=4))
    println("Master interface x-coords: ", round.(x_m, digits=4))
    println("For u_x=x/E patch test: all interface CPs have x=0.5")
    println("  → interface displacement is CONSTANT (0.5/E)")
    println("  → D·d_s = M·d_m holds exactly (POU) even under elem-based")
    println("  → SP-Elm does NOT fail the flat constant-field patch test!")

    # ── The actual source of error: check y-direction field ─────────────
    # For a y-direction patch test (u_y = y/E), d_s = y_s, d_m = y_m
    # This IS a non-trivial field along the interface.
    for nq in [1, 2, 3]
        D_elm, M_elm, _, _ = build_mortar_mass_matrices(
            pair, p, n, KV, P, B, nnp, nsd, npd, nq, ElementBasedIntegration())
        Dd = Matrix(D_elm); Md = Matrix(M_elm)
        residual = Dd * y_s - Md * y_m
        @printf("\nNQUAD=%d: ‖D·y_s - M·y_m‖ = %.4e   (linear field along interface)\n",
                nq, norm(residual))
        @printf("         ‖D·1_s - M·1_m‖ = %.4e   (constant field)\n",
                norm(Dd * ones_s - Md * ones_m))
        # Show individual row residuals
        if nq == 1
            println("  Row residuals (D·y - M·y):")
            for i in 1:length(s_cps)
                @printf("    row %d (y_s=%.3f): %.4e\n", i, y_s[i], residual[i])
            end
        end
    end
end

# ─────────────────────── Force-moment analysis ───────────────────────────────

"""
    compute_force_moments(D, M, s_cps, m_cps, B; dim=2)

Compute the force-moment equilibrium errors δ_0, δ_1, δ_2 for a single
interface pass, using a uniform test multiplier λ = 1.

From Eq. (\\ref{eq:force_moments}):
    δ_k = 1ᵀ (D·φ_s − M·φ_m)
where φ is a monomial of degree k evaluated at the CP coordinates.

- `D`  : slave-slave mass matrix (n_s × n_s)
- `M`  : slave-master cross-mass matrix (n_s × n_m)
- `s_cps`, `m_cps` : global CP indices for slave/master interface nodes
- `B`  : control point array (ncp × 4), columns [x, y, z, w]
- `dim`: spatial dimension for the interface coordinate (default 2 = y)

Returns `(δ_0, δ_1, δ_2)`.
"""
function compute_force_moments(D, M, s_cps, m_cps, B; dim::Int = 2)
    Dd = Matrix(D);  Md = Matrix(M)
    ns = length(s_cps)

    ones_s = ones(ns)
    ones_m = ones(length(m_cps))
    y_s  = [B[cp, dim] for cp in s_cps]
    y_m  = [B[cp, dim] for cp in m_cps]
    y2_s = y_s .^ 2
    y2_m = y_m .^ 2

    λ = ones(ns)   # uniform test multiplier
    δ_0 = dot(λ, Dd * ones_s - Md * ones_m)
    δ_1 = dot(λ, Dd * y_s    - Md * y_m)
    δ_2 = dot(λ, Dd * y2_s   - Md * y2_m)

    return (δ_0 = δ_0, δ_1 = δ_1, δ_2 = δ_2)
end

"""
    run_moment_table(n_s, n_m; nquad_range, epss)

Print force-moment equilibrium errors δ_0, δ_1, δ_2 for four methods
(SPMS, SPME, DPM, TM) on the flat p=1 patch test.

For single-pass methods: one set of moments from the single pass.
For two-pass methods (DPM, TM): moments per pass, showing antisymmetry.
"""
function run_moment_table(
    n_s::Int = 2,
    n_m::Int = 3;
    nquad_range = 1:6,
)
    nsd = 2; npd = 2; npc = 2
    p = [1 1; 1 1]
    n = [2 n_s+1; 2 n_m+1]
    KV = generate_knot_vectors(npc, npd, p, n)

    ncp1 = 2 * (n_s + 1)
    ncp2 = 2 * (n_m + 1)
    ncp  = ncp1 + ncp2

    B = zeros(ncp, 4)
    idx = 0
    for j in 0:n_s, i in 0:1
        idx += 1
        B[idx, :] = [i * 0.5, j / n_s, 0.0, 1.0]
    end
    for j in 0:n_m, i in 0:1
        idx += 1
        B[idx, :] = [0.5 + i * 0.5, j / n_m, 0.0, 1.0]
    end

    P = [collect(1:ncp1), collect(ncp1+1:ncp)]
    nel, nnp, nen = patch_metrics(npc, npd, p, n)

    pair1 = InterfacePair(1, 2, 2, 4)   # pass 1: slave=P1, master=P2
    pair2 = InterfacePair(2, 4, 1, 2)   # pass 2: slave=P2, master=P1

    println("\n=== Force-moment analysis: flat p=1 interface, n_s=$n_s, n_m=$n_m ===")
    println("Moments computed with uniform test multiplier λ = 1")
    println("Interface at x = 0.5, coordinate dimension = y\n")

    strat = ElementBasedIntegration()
    nquad_ref = 20   # high NQUAD as reference for "segment-quality" results

    for nq in nquad_range
        println("─── NQUAD = $nq ───")
        @printf("%-8s  %12s  %12s  %14s  %14s  %14s\n",
                "Method", "δ₀", "δ₁", "δ₂ (pass 1)", "δ₂ (pass 2)", "δ₂ (sum)")
        @printf("%s\n", "─"^80)

        # ── Single-pass reference (high NQUAD ≈ segment-based) ────────────
        D1_ref, M12_ref, s1, m1 = build_mortar_mass_matrices(
            pair1, p, n, KV, P, B, nnp, nsd, npd, nquad_ref, strat)
        mom_ref = compute_force_moments(D1_ref, M12_ref, s1, m1, B)
        @printf("%-8s  %12.2e  %12.2e  %14.4e  %14s  %14s\n",
                "SP-ref", mom_ref.δ_0, mom_ref.δ_1, mom_ref.δ_2, "—", "—")

        # ── Single-pass element-based (SPME) ──────────────────────────────
        D1_e, M12_e, _, _ = build_mortar_mass_matrices(
            pair1, p, n, KV, P, B, nnp, nsd, npd, nq, strat)
        mom_sp = compute_force_moments(D1_e, M12_e, s1, m1, B)
        @printf("%-8s  %12.2e  %12.2e  %14.4e  %14s  %14s\n",
                "SPME", mom_sp.δ_0, mom_sp.δ_1, mom_sp.δ_2, "—", "—")

        # ── Two-pass reference (high NQUAD ≈ DPM) ────────────────────────
        D2_ref, M21_ref, s2, m2 = build_mortar_mass_matrices(
            pair2, p, n, KV, P, B, nnp, nsd, npd, nquad_ref, strat)
        mom2_ref = compute_force_moments(D2_ref, M21_ref, s2, m2, B)
        @printf("%-8s  %12.2e  %12.2e  %14.4e  %14.4e  %14.4e\n",
                "DP-ref", mom_ref.δ_0, mom_ref.δ_1,
                mom_ref.δ_2, mom2_ref.δ_2, mom_ref.δ_2 + mom2_ref.δ_2)

        # ── Twin Mortar (element-based, both passes) ─────────────────────
        D2_e, M21_e, _, _ = build_mortar_mass_matrices(
            pair2, p, n, KV, P, B, nnp, nsd, npd, nq, strat)
        mom2_tm = compute_force_moments(D2_e, M21_e, s2, m2, B)
        @printf("%-8s  %12.2e  %12.2e  %14.4e  %14.4e  %14.4e\n",
                "TM", mom_sp.δ_0, mom_sp.δ_1,
                mom_sp.δ_2, mom2_tm.δ_2, mom_sp.δ_2 + mom2_tm.δ_2)

        println()
    end
end

# ─────────────────────── Patch test ε-sweep ──────────────────────────────────

"""
    run_patch_eps_sweep(n_s, n_m; eps_range, p_ord, NQUAD_mortar)

Displacement error vs stabilization parameter ε for Twin Mortar
on the flat 2D patch test.  Shows conditional patch test pass.
"""
function run_patch_eps_sweep(
    n_s::Int = 2,
    n_m::Int = 3;
    eps_range = 10.0 .^ (-2:8),
    NQUAD_mortar::Int = 3,
)
    E = 1000.0

    println("\n=== Patch test ε-sweep: p=1, n_s=$n_s, n_m=$n_m, NQUAD=$NQUAD_mortar ===")
    @printf("%-12s  %14s\n", "ε", "‖δu‖/‖u‖")
    @printf("%s\n", "─"^28)

    for eps in eps_range
        r = flat_patch_test(n_s, n_m; E=E, epss=eps,
                            NQUAD_mortar=NQUAD_mortar,
                            strategy=ElementBasedIntegration(),
                            formulation=TwinMortarFormulation())
        err = disp_error(r.U, r.ID, r.B, r.ncp, E)
        @printf("%-12.1e  %14.4e\n", eps, err)
    end
end

# ─────────────────────── Lambda norms for patch test ─────────────────────────

"""
    run_lambda_norms(n_s, n_m; p_range, epss_tm, epss_dpm, NQUAD_mortar)

For each polynomial order p and each method (SPMS, SPME, DPM, TM),
solve the flat patch test and report ‖λ^(s)‖_∞ (and ‖λ^(m)‖_∞ for
two-pass methods).  This populates tab:patch_lambda.
"""
function run_lambda_norms(
    n_s::Int = 4,
    n_m::Int = 7;
    p_range   = 1:4,
    epss_tm::Float64  = 1.0,
    epss_dpm::Float64 = 1.0,
)
    E = 1000.0

    configs = [
        ("SPMS", SinglePassFormulation(), SegmentBasedIntegration(), 0.0),
        ("SPME", SinglePassFormulation(), ElementBasedIntegration(), 0.0),
        ("DPM",  DualPassFormulation(),   SegmentBasedIntegration(), epss_dpm),
        ("TM",   TwinMortarFormulation(), ElementBasedIntegration(), epss_tm),
    ]

    for p_ord in p_range
        NQUAD_m = p_ord + 2
        println("\n=== Lambda norms: flat patch test, p=$p_ord, n_s=$n_s, n_m=$n_m ===")
        @printf("%-6s  %14s  %14s  %14s\n",
                "Method", "‖λ^(s)‖_∞", "‖λ^(m)‖_∞", "‖δu‖/‖u‖")
        @printf("%s\n", "─"^52)

        for (label, form, strat, eps) in configs
            r = try
                flat_patch_test_hp(p_ord, n_s, n_m; E=E, epss=eps,
                                NQUAD_mortar=NQUAD_m,
                                strategy=strat, formulation=form)
            catch e
                e isa SingularException || rethrow()
                @printf("%-6s  %14s  %14s  %14s\n", label, "SINGULAR", "—", "—")
                continue
            end

            _, Lambda = solve_mortar(r.K_bc, r.C, r.Z, r.F_bc)
            n_mult = length(Lambda)
            err = disp_error(r.U, r.ID, r.B, r.ncp, E)

            if form isa SinglePassFormulation
                @printf("%-6s  %14.4e  %14s  %14.4e\n",
                        label, norm(Lambda, Inf), "—", err)
            else
                half = n_mult ÷ 2
                lam_s = Lambda[1:half]
                lam_m = Lambda[half+1:end]
                @printf("%-6s  %14.4e  %14.4e  %14.4e\n",
                        label, norm(lam_s, Inf), norm(lam_m, Inf), err)
            end
        end
    end
end

# ─────────────────────── Entry point ─────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    run_cancellation_study(2, 3)
    println("\n\n")
    run_cancellation_study(4, 7)
end
