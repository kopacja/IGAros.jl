# examples/concentric_cylinders.jl
#
# Concentric cylinders — ε-sensitivity study (Twin Mortar method).
# Two-patch quarter-annulus, non-conforming curved (circular arc) interface.
# Exact Lamé solution for traction loading and L2 stress error.
#
# Parameters recovered from paper (§6.3):
#   r_i = 1, r_o = 2, r_c = 1.5  (inner/outer/interface radii)
#   E = 100, ν = 0.3, plane strain
#   p_o = 1  (external pressure on outer boundary)
#
# The scaling-law verification in §5.4 gives ε ~ h²/E ~ 10⁻⁴→10⁻⁶
# for h ≈ 0.1→0.01 at E=100, confirming that ε=10⁻⁸ is conservative.
#
# Non-conforming mesh: Patch 1 (inner, r_i→r_c) has twice as many
# angular elements as Patch 2 (outer, r_c→r_o).
#
# Interface:
#   Patch 1 facet 3 (η=n, outer arc of inner patch) ↔
#   Patch 2 facet 1 (η=1, inner arc of outer patch)

import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using IGAros
using LinearAlgebra, SparseArrays, Printf

# ─────────────────────── Lamé exact solution ─────────────────────────────────

"""
    lame_stress(x, y; p_o, r_i, r_o) -> Matrix{Float64}(2×2)

Exact Lamé Cauchy stress for a thick cylinder under external pressure p_o
(zero internal pressure), plane strain. Returns 2×2 Cartesian tensor.
"""
function lame_stress(x::Real, y::Real;
                     p_o::Float64, r_i::Float64, r_o::Float64)::Matrix{Float64}
    r  = sqrt(x^2 + y^2)
    A  = -p_o * r_o^2 / (r_o^2 - r_i^2)
    σ_rr = A * (1 - r_i^2/r^2)   # radial stress (compression → negative)
    σ_θθ = A * (1 + r_i^2/r^2)   # hoop stress

    c, s = x/r, y/r              # cosθ, sinθ
    T    = [c s; -s c]           # rotation matrix (Cartesian → polar columns)
    return T' * [σ_rr 0.0; 0.0 σ_θθ] * T   # σ_cart = T^T σ_polar T
end

"""
    lame_displacement(x, y; p_o, r_i, r_o, E, nu) -> (ux, uy)

Exact Lamé displacement field (plane strain) in Cartesian coordinates.
"""
function lame_displacement(x::Real, y::Real;
                           p_o::Float64, r_i::Float64, r_o::Float64,
                           E::Float64,   nu::Float64)
    r   = sqrt(x^2 + y^2)
    den = E * (r_o^2 - r_i^2)
    C1  = -p_o * r_o^2 * (1 + nu) * (1 - 2nu) / den   # coefficient of r
    C2  = -p_o * r_i^2 * r_o^2   * (1 + nu)           / den   # coefficient of 1/r
    u_r = C1 * r + C2 / r
    return u_r * x/r, u_r * y/r
end

# ─────────────────────── Bezier degree elevation ─────────────────────────────

"""
    bezier_elevate(Bh) -> Bh_elevated

Elevate a Bezier curve by one degree. `Bh` is a (p+1)×ncols matrix of
homogeneous control points. Returns (p+2)×ncols elevated points.
"""
function bezier_elevate(Bh::Matrix{Float64})::Matrix{Float64}
    p  = size(Bh, 1) - 1
    Qh = zeros(p + 2, size(Bh, 2))
    Qh[1,   :] = Bh[1,   :]
    Qh[end, :] = Bh[end, :]
    for i in 1:p
        α = i / (p + 1)
        Qh[i+1, :] = α * Bh[i, :] + (1 - α) * Bh[i+1, :]
    end
    return Qh
end

# ─────────────────────── Patch geometry ──────────────────────────────────────

"""
    cylinder_geometry(p_ord; r_i, r_c, r_o) -> (B, P)

Build the two-patch NURBS geometry for a quarter annular cross-section.

  Patch 1: annular sector r ∈ [r_i, r_c], θ ∈ [90°, 0°]   (inner)
  Patch 2: annular sector r ∈ [r_c, r_o], θ ∈ [90°, 0°]   (outer)

ξ direction (angular): from θ = 90° (ξ=0) to θ = 0° (ξ=1)
η direction (radial):  from r_inner  (η=0) to r_outer (η=1)

Facet conventions (for each patch):
  Facet 1 (η=1, bottom): inner arc — free (Patch 1: r_i), or interface (Patch 2: r_c)
  Facet 2 (ξ=n, right):  θ = 0°  → uy = 0 symmetry BC
  Facet 3 (η=n, top):    outer arc — interface (Patch 1: r_c), or load (Patch 2: r_o)
  Facet 4 (ξ=1, left):   θ = 90° → ux = 0 symmetry BC

Returns B (ncp × 4, [x, y, 0, w]) and P (patch-to-global CP index list).
"""
function cylinder_geometry(p_ord::Int;
                           r_i::Float64 = 1.0,
                           r_c::Float64 = 1.5,
                           r_o::Float64 = 2.0)::Tuple{Matrix{Float64}, Vector{Vector{Int}}}

    # ── Quarter-circle arc at radius r, θ from 90° down to 0° ────────────────
    # Quadratic NURBS exact circle: 3 CPs with weights [1, cos(π/4), 1]
    function arc_cps_q2(r)
        return [0.0  r    1.0;      # at θ = 90°
                r    r    cos(π/4); # corner (weight < 1 for exact circle)
                r    0.0  1.0]      # at θ = 0°
    end

    # Degree-elevate from quadratic (p=2) to p_ord
    function arc_cps(r)
        Bh = arc_cps_q2(r)
        Bh[:, 1:2] .*= Bh[:, 3:3]  # to homogeneous [Wx, Wy, W]
        for _ in 3:p_ord
            Bh = bezier_elevate(Bh)
        end
        Bh[:, 1:2] ./= Bh[:, 3:3]  # back to Euclidean [x, y, w]
        return Bh                    # (p_ord+1) × 3
    end

    n_ang = p_ord + 1   # CPs in ξ (angular) direction
    n_rad = p_ord + 1   # CPs in η (radial) direction

    # ── Linear blend in η (radial direction) between two arcs ────────────────
    # Scaling inner arc CPs by r_outer/r_inner preserves the NURBS circle
    # representation (same weights, scaled coordinates) → exact circles at
    # every η level.
    function blend_patch(r_inner, r_outer)
        arc_in  = arc_cps(r_inner)
        arc_out = arc_cps(r_outer)
        B = zeros(n_ang * n_rad, 3)
        for j in 0:n_rad-1
            t = j / (n_rad - 1)
            for i in 1:n_ang
                B[j*n_ang + i, :] = (1 - t) .* arc_in[i, :] .+ t .* arc_out[i, :]
            end
        end
        return B
    end

    B1_xyw = blend_patch(r_i, r_c)   # Patch 1: inner sector
    B2_xyw = blend_patch(r_c, r_o)   # Patch 2: outer sector

    # ── Assemble global B in IGAros format [x, y, z=0, w] ───────────────────
    ncp1 = n_ang * n_rad
    ncp2 = n_ang * n_rad
    B_out = zeros(ncp1 + ncp2, 4)
    B_out[1:ncp1,       [1,2,4]] = B1_xyw
    B_out[ncp1+1:end,   [1,2,4]] = B2_xyw

    P = [collect(1:ncp1), collect(ncp1+1:ncp1+ncp2)]
    return B_out, P
end

# ─────────────────────── L2 stress error ─────────────────────────────────────

"""
    l2_stress_error_cyl(U, ID, npc, nsd, npd, p, n, KV, P, B,
                        nen, nel, IEN, INC, materials, NQUAD, thickness,
                        stress_fn) -> (err_abs, err_ref)

Compute relative L2 stress error with a user-supplied exact stress function.
"""
function l2_stress_error_cyl(
    U::Vector{Float64},
    ID::Matrix{Int},
    npc::Int, nsd::Int, npd::Int,
    p::Matrix{Int}, n::Matrix{Int},
    KV::Vector{<:AbstractVector{<:AbstractVector{Float64}}},
    P::Vector{Vector{Int}},
    B::Matrix{Float64},
    nen::Vector{Int}, nel::Vector{Int},
    IEN::Vector{Matrix{Int}},
    INC::Vector{<:AbstractVector{<:AbstractVector{Int}}},
    materials::Vector{LinearElastic},
    NQUAD::Int,
    thickness::Float64,
    stress_fn::Function   # (x, y) -> 2×2 exact stress matrix
)::Tuple{Float64, Float64}

    ncp = size(B, 1)
    Ub  = zeros(ncp, nsd)
    for A in 1:ncp, i in 1:nsd
        eq = ID[i, A]
        eq != 0 && (Ub[A, i] = U[eq])
    end

    err2 = 0.0
    ref2 = 0.0
    GPW  = gauss_product(NQUAD, npd)

    for pc in 1:npc
        ien = IEN[pc]; inc = INC[pc]
        D   = elastic_constants(materials[pc], nsd)

        for el in 1:nel[pc]
            anchor = ien[el, 1]
            n0     = inc[anchor]

            for (gp, gw) in GPW
                R_s, dR_dx, _, detJ, _ = shape_function(
                    p[pc,:], n[pc,:], KV[pc], B, P[pc], gp,
                    nen[pc], nsd, npd, el, n0, ien, inc
                )
                detJ <= 0 && continue

                gwJ = gw * detJ * thickness

                B0    = strain_displacement_matrix(nsd, nen[pc], dR_dx')
                Ue    = vec(Ub[P[pc][ien[el,:]], 1:nsd]')
                σ_h_v = D * (B0 * Ue)

                Xe = B[P[pc][ien[el,:]], :]
                X  = Xe' * R_s
                σ_ex = stress_fn(X[1], X[2])

                σ_h_m = [σ_h_v[1] σ_h_v[3]; σ_h_v[3] σ_h_v[2]]
                diff_m = σ_h_m - σ_ex

                err2 += dot(diff_m, diff_m) * gwJ
                ref2 += dot(σ_ex,   σ_ex)   * gwJ
            end
        end
    end

    return sqrt(err2), sqrt(ref2)
end

# ─────────────────────── Single-level solve ───────────────────────────────────

"""
    solve_cylinder(p_ord, exp_level; conforming, epss, ...) -> (err_rel, err_abs)

Run one refinement level of the concentric-cylinder benchmark.
Non-conforming: Patch 1 (inner) has twice as many angular elements as Patch 2.
"""
function solve_cylinder(
    p_ord::Int,
    exp_level::Int;
    conforming::Bool  = false,
    r_i::Float64      = 1.0,
    r_c::Float64      = 1.5,
    r_o::Float64      = 2.0,
    E::Float64        = 100.0,
    nu::Float64       = 0.3,
    p_o::Float64      = 1.0,
    epss::Float64     = 0.0,      # 0 = auto: 1e4 (large enough for correct coupling)
    NQUAD::Int        = p_ord + 1,
    NQUAD_mortar::Int = 10
)::Tuple{Float64, Float64}

    nsd = 2; npd = 2; ned = 2; npc = 2
    thickness = 1.0

    # ── Initial coarse geometry ───────────────────────────────────────────────
    B0, P = cylinder_geometry(p_ord; r_i=r_i, r_c=r_c, r_o=r_o)
    p_mat = fill(p_ord, npc, npd)
    n_mat = fill(p_ord + 1, npc, npd)
    KV    = generate_knot_vectors(npc, npd, p_mat, n_mat)

    # ── h-refinement ─────────────────────────────────────────────────────────
    # ξ (angular): step s_ang = 1/(3·2^exp), Patch 1 finer (s_ang_nc = s_ang/2)
    # η (radial):  step s_rad = 1/(2·2^exp), same for both patches
    s_ang    = (1/3) / 2^exp_level
    s_ang_nc = s_ang / 2            # finer angular for inner patch (non-conforming)
    s_rad    = (1/2) / 2^exp_level

    u_ang    = collect(s_ang    : s_ang    : 1 - s_ang/2)
    u_ang_nc = collect(s_ang_nc : s_ang_nc : 1 - s_ang_nc/2)
    u_rad    = collect(s_rad    : s_rad    : 1 - s_rad/2)

    # Stabilization parameter: must be large enough for Z to dominate the
    # constraint (empirically needs epss >> E/h; fixed 1e4 works for p=2 on
    # all tested refinement levels). Small values (e.g. h²/E) over-constrain
    # the interface and produce constant ~82% error.
    epss_use = epss > 0.0 ? epss : 1e4

    kref_data = Vector{Float64}[
        vcat([1.0, 1.0], conforming ? u_ang : u_ang_nc),  # Patch 1 angular
        vcat([2.0, 1.0], u_ang),                           # Patch 2 angular
        vcat([1.0, 2.0], u_rad),                           # Patch 1 radial
        vcat([2.0, 2.0], u_rad),                           # Patch 2 radial
    ]

    # ── y-offset hack (prevent CP merging at curved interface) ───────────────
    # Interface CPs of Patch 1 (outer arc at r_c) coincide physically with
    # Patch 2's inner arc at r_c. Without the offset, krefinement merges them.
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

    # ── Connectivity ──────────────────────────────────────────────────────────
    nel, nnp, nen = patch_metrics(npc, npd, p_mat, n_mat_ref)
    IEN = build_ien(nsd, npd, npc, p_mat, n_mat_ref, nel, nnp, nen)
    INC = [build_inc(n_mat_ref[pc, :]) for pc in 1:npc]

    # ── Dirichlet BCs (symmetry) ──────────────────────────────────────────────
    # Facet 4 (ξ=1, θ=90°, x=0):  ux = 0 on both patches
    # Facet 2 (ξ=n, θ=0°,  y=0):  uy = 0 on both patches
    dBC = [1 4 2 1 2;
           2 2 2 1 2]
    bc_per_dof = dirichlet_bc_control_points(p_mat, n_mat_ref, KV_ref, P_ref,
                                              npd, nnp, ned, dBC)
    neq, ID = build_id(bc_per_dof, ned, ncp)
    LM      = build_lm(nen, ned, npc, nel, ID, IEN, P_ref)

    # ── Stiffness matrix ──────────────────────────────────────────────────────
    mats = [LinearElastic(E, nu, :plane_strain), LinearElastic(E, nu, :plane_strain)]
    Ub0  = zeros(ncp, nsd)
    K, _ = build_stiffness_matrix(npc, nsd, npd, ned, neq,
                                   p_mat, n_mat_ref, KV_ref, P_ref, B_ref, Ub0,
                                   nen, nel, IEN, INC, LM, mats, NQUAD, thickness)

    # ── Load: Neumann traction on Patch 2 facet 3 (outer arc r = r_o) ────────
    stress_fn = (x, y) -> lame_stress(x, y; p_o=p_o, r_i=r_i, r_o=r_o)
    F = zeros(neq)
    F = segment_load(n_mat_ref[2,:], p_mat[2,:], KV_ref[2], P_ref[2], B_ref,
                     nnp[2], nen[2], nsd, npd, ned,
                     Int[], 3, ID, F, stress_fn, thickness, NQUAD)

    K_bc, F_bc = enforce_dirichlet(Tuple{Int,Float64}[], K, F)

    # ── Twin Mortar coupling at curved interface ──────────────────────────────
    # Patch 1 facet 3 (outer arc at r_c) ↔ Patch 2 facet 1 (inner arc at r_c)
    pairs = [InterfacePair(1, 3, 2, 1),   # slave=pc1(fac3), master=pc2(fac1)
             InterfacePair(2, 1, 1, 3)]   # slave=pc2(fac1), master=pc1(fac3)
    Pc = build_interface_cps(pairs, p_mat, n_mat_ref, KV_ref, P_ref, npd, nnp)

    C, Z = build_mortar_coupling(Pc, pairs, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
                                  ID, nnp, ned, nsd, npd, neq, NQUAD_mortar, epss_use)

    # ── Solve ─────────────────────────────────────────────────────────────────
    U, _ = solve_mortar(K_bc, C, Z, F_bc)

    # ── L2 stress error ───────────────────────────────────────────────────────
    err_abs, err_ref = l2_stress_error_cyl(
        U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
        nen, nel, IEN, INC, mats, NQUAD, thickness, stress_fn
    )

    return err_abs / err_ref, err_abs
end

# ─────────────────────── ε-sensitivity sweep ─────────────────────────────────

"""
    eps_sensitivity(p_ord, exp_level; eps_values, kwargs...)

Sweep ε values at a fixed mesh and print error vs ε table.
Also marks the scaling-law prediction ε = h²/E.
"""
function eps_sensitivity(
    p_ord::Int,
    exp_level::Int;
    eps_values = 10 .^ (-12:1:2),
    kwargs...
)
    E    = get(kwargs, :E,   100.0)
    s_ang = (1/3) / 2^exp_level       # coarser angular step (Patch 2)
    eps_predict = s_ang^2 / E         # scaling-law prediction

    @printf("\n=== ε-sensitivity: p=%d, exp=%d  (scaling-law ε* = %.2e) ===\n",
            p_ord, exp_level, eps_predict)
    @printf("%-14s  %12s  %12s\n", "ε", "‖e‖/‖σ‖ (rel)", "‖e‖ (abs)")
    @printf("%s\n", "-"^42)

    for eps in eps_values
        rel, abs_err = solve_cylinder(p_ord, exp_level; epss=Float64(eps), kwargs...)
        marker = abs(eps - eps_predict) / eps_predict < 0.5 ? " ← predicted ε*" : ""
        @printf("%-14.2e  %12.4e  %12.4e%s\n", eps, rel, abs_err, marker)
    end
end

"""
    run_convergence_cyl(; degrees, exp_range, kwargs...)

Print convergence table for the concentric cylinder benchmark (adaptive epss).
"""
function run_convergence_cyl(;
    degrees::Vector{Int} = [2, 3, 4],
    exp_range::UnitRange = 0:5,
    kwargs...
)
    exps = collect(exp_range)
    ne   = length(exps)

    println("\n=== Relative L2 stress error (non-conforming, large ε for correct coupling) ===")
    @printf("%-5s", "p\\exp")
    for e in exps;  @printf("  %10s", "exp=$e");  end;  println()
    @printf("%s\n", "-"^(5 + 12*ne))

    for p_ord in degrees
        @printf("p=%-2d", p_ord)
        for e in exps
            rel, _ = solve_cylinder(p_ord, e; kwargs...)
            @printf("  %10.3e", rel)
        end
        println()
    end

    println("\n--- Convergence rates ---")
    @printf("%-5s", "p\\exp")
    for i in 1:ne-1;  @printf("  %10s", "$(exps[i])→$(exps[i+1])");  end;  println()
    @printf("%s\n", "-"^(5 + 12*(ne-1)))

    for p_ord in degrees
        errs = [solve_cylinder(p_ord, e; kwargs...)[2] for e in exps]
        @printf("p=%-2d", p_ord)
        for i in 1:ne-1
            @printf("  %10.2f", log2(errs[i]/errs[i+1]))
        end
        println()
    end
end

# ─────────────────────── Entry point ─────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    # ε-sensitivity at exp=3 (h ≈ 0.1, scaling law gives ε* ≈ h²/E ≈ 4e-5)
    println("=== ε-sensitivity study (fixed mesh exp=3, p=2,3,4) ===")
    for p_ord in [2, 3, 4]
        eps_sensitivity(p_ord, 3;
            eps_values = 10 .^ (-10:1:2))
    end

    println("\n\n=== Convergence study (adaptive ε = s_ang²/E) ===")
    run_convergence_cyl(degrees=[2, 3, 4], exp_range=0:4)
end
