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
    # Special case p=1: use a straight chord between the two endpoints (w=1).
    # The p=2 representation starts from a 3-CP exact circle; taking only the
    # first 2 rows would give θ∈[90°,45°] instead of [90°,0°].
    function arc_cps(r)
        if p_ord == 1
            return [0.0  r    1.0;   # θ = 90°
                    r    0.0  1.0]   # θ = 0°
        end
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

"""
    energy_error_cyl(U, ID, ...) -> (err_E, ref_E)

Energy-norm error against the Lamé solution:
  err_E² = ∫_Ω (σ_h − σ_ex) : D⁻¹ : (σ_h − σ_ex) dΩ
  ref_E² = ∫_Ω σ_ex : D⁻¹ : σ_ex dΩ
where D is the plane-strain elasticity tensor (3×3 Voigt).
"""
function energy_error_cyl(
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
        ien  = IEN[pc]; inc = INC[pc]
        D    = elastic_constants(materials[pc], nsd)
        Dinv = inv(D)   # 3×3 compliance matrix (plane-strain)

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
                σ_h_v = D * (B0 * Ue)   # Voigt [σxx, σyy, τxy]

                Xe = B[P[pc][ien[el,:]], :]
                X  = Xe' * R_s
                σ_ex   = stress_fn(X[1], X[2])
                σ_ex_v = [σ_ex[1,1], σ_ex[2,2], σ_ex[1,2]]

                Δσ_v = σ_h_v - σ_ex_v

                err2 += dot(Δσ_v, Dinv * Δσ_v) * gwJ
                ref2 += dot(σ_ex_v, Dinv * σ_ex_v) * gwJ
            end
        end
    end

    return sqrt(err2), sqrt(ref2)
end

"""
    l2_disp_error_cyl(U, ID, npc, nsd, npd, p, n, KV, P, B,
                      nen, nel, IEN, INC, NQUAD, thickness, disp_fn)
        -> (err_abs, err_ref)

Compute L2 displacement error against a user-supplied exact solution.
  err_abs² = ∫_Ω ||u_h − u_ex||² dΩ,   err_ref² = ∫_Ω ||u_ex||² dΩ
"""
function l2_disp_error_cyl(
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
    NQUAD::Int,
    thickness::Float64,
    disp_fn::Function   # (x, y) -> (ux, uy)
)::Tuple{Float64, Float64}

    ncp = size(B, 1)
    Ub  = zeros(ncp, nsd)
    for A in 1:ncp, i in 1:nsd
        eq = ID[i, A]; eq != 0 && (Ub[A, i] = U[eq])
    end

    err2 = 0.0; ref2 = 0.0
    GPW  = gauss_product(NQUAD, npd)

    for pc in 1:npc
        ien = IEN[pc]; inc = INC[pc]
        for el in 1:nel[pc]
            anchor = ien[el, 1]
            n0     = inc[anchor]
            for (gp, gw) in GPW
                R_s, _, _, detJ, _ = shape_function(
                    p[pc,:], n[pc,:], KV[pc], B, P[pc], gp,
                    nen[pc], nsd, npd, el, n0, ien, inc
                )
                detJ <= 0 && continue
                gwJ = gw * detJ * thickness

                Ue_mat = Ub[P[pc][ien[el,:]], 1:nsd]   # (nen_pc × nsd)
                u_h    = Ue_mat' * R_s                  # (nsd,)

                Xe     = B[P[pc][ien[el,:]], :]
                X      = Xe' * R_s
                ux_ex, uy_ex = disp_fn(X[1], X[2])
                u_ex   = [ux_ex, uy_ex]

                diff = u_h[1:nsd] - u_ex
                err2 += dot(diff, diff) * gwJ
                ref2 += dot(u_ex, u_ex) * gwJ
            end
        end
    end

    return sqrt(err2), sqrt(ref2)
end

# ─────────────────────── Single-level solve ───────────────────────────────────

"""
    solve_cylinder(p_ord, exp_level; ...) -> NTuple{6, Float64}

Run one refinement level of the concentric-cylinder benchmark.
Non-conforming: Patch 1 (inner) has twice as many angular elements as Patch 2.
Returns `(l2_stress_rel, l2_stress_abs, l2_disp_rel, l2_disp_abs, en_rel, en_abs)`.
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
    epss::Float64              = 0.0,      # 0 = auto: 1e4 (large enough for correct coupling)
    NQUAD::Int                 = p_ord + 1,
    NQUAD_mortar::Int          = p_ord + 2,
    strategy::IntegrationStrategy    = ElementBasedIntegration(),
    formulation::FormulationStrategy = TwinMortarFormulation(),
    n_ang_p2_base::Int = 3,   # outer-patch angular base (h ∝ 1/(base·2^exp))
    n_ang_p1_base::Int = 6,   # inner-patch angular base (2:1 by default)
)::NTuple{6, Float64}

    nsd = 2; npd = 2; ned = 2; npc = 2
    thickness = 1.0

    # ── Initial coarse geometry ───────────────────────────────────────────────
    B0, P = cylinder_geometry(p_ord; r_i=r_i, r_c=r_c, r_o=r_o)
    p_mat = fill(p_ord, npc, npd)
    n_mat = fill(p_ord + 1, npc, npd)
    KV    = generate_knot_vectors(npc, npd, p_mat, n_mat)

    # ── h-refinement ─────────────────────────────────────────────────────────
    # ξ (angular): base element counts set by n_ang_p2_base (outer) and n_ang_p1_base (inner)
    # η (radial):  step s_rad = 1/(2·2^exp), same for both patches
    s_ang    = 1.0 / (n_ang_p2_base * 2^exp_level)
    s_ang_nc = 1.0 / (n_ang_p1_base * 2^exp_level)
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

    # ── Mortar coupling at curved interface ───────────────────────────────────
    # Patch 1 facet 3 (outer arc at r_c) ↔ Patch 2 facet 1 (inner arc at r_c)
    pairs_tm = [InterfacePair(1, 3, 2, 1),   # slave=pc1(fac3), master=pc2(fac1)
                InterfacePair(2, 1, 1, 3)]   # slave=pc2(fac1), master=pc1(fac3)
    pairs_sp = [InterfacePair(1, 3, 2, 1)]
    pairs = formulation isa SinglePassFormulation ? pairs_sp : pairs_tm

    Pc = build_interface_cps(pairs, p_mat, n_mat_ref, KV_ref, P_ref, npd, nnp, formulation)
    C, Z = build_mortar_coupling(Pc, pairs, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
                                  ID, nnp, ned, nsd, npd, neq, NQUAD_mortar, epss_use,
                                  strategy, formulation)

    # ── Solve ─────────────────────────────────────────────────────────────────
    U, _ = solve_mortar(K_bc, C, Z, F_bc)

    # ── L2 stress error ───────────────────────────────────────────────────────
    s_abs, s_ref = l2_stress_error_cyl(
        U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
        nen, nel, IEN, INC, mats, NQUAD, thickness, stress_fn
    )

    # ── L2 displacement error ─────────────────────────────────────────────────
    disp_fn = (x, y) -> lame_displacement(x, y; p_o=p_o, r_i=r_i, r_o=r_o, E=E, nu=nu)
    d_abs, d_ref = l2_disp_error_cyl(
        U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
        nen, nel, IEN, INC, NQUAD, thickness, disp_fn
    )

    # ── Energy-norm error ──────────────────────────────────────────────────────
    en_abs, en_ref = energy_error_cyl(
        U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
        nen, nel, IEN, INC, mats, NQUAD, thickness, stress_fn
    )

    return s_abs / s_ref, s_abs, d_abs / d_ref, d_abs, en_abs / en_ref, en_abs
end

# ─────────────────────── Reference-solution convergence ───────────────────────

"""
    _eval_patch_stress(ξ_g, η_g, pc, p_mat, n_mat, KV, P, B, Ub, D, nsd)

Evaluate the 2×2 Cauchy stress tensor at global parametric coordinates
(ξ_g, η_g) in patch `pc` using the given FEM displacement array `Ub`.
Ub is ncp × nsd (already extracted from the equation-number solution).
"""
function _eval_patch_stress(
    ξ_g::Float64, η_g::Float64, pc::Int,
    p_mat::Matrix{Int}, n_mat::Matrix{Int},
    KV::Vector{<:AbstractVector{<:AbstractVector{Float64}}},
    P::Vector{Vector{Int}},
    B::Matrix{Float64},
    Ub::Matrix{Float64},
    D::Matrix{Float64},
    nsd::Int
)
    p1, p2 = p_mat[pc, 1], p_mat[pc, 2]
    n1     = n_mat[pc, 1]          # number of CPs in ξ (angular) direction
    kv1, kv2 = KV[pc][1], KV[pc][2]

    span1 = find_span(n_mat[pc,1]-1, p1, ξ_g, kv1)
    span2 = find_span(n_mat[pc,2]-1, p2, η_g, kv2)
    dN1_m = bspline_basis_and_deriv(span1, ξ_g, p1, 1, kv1)  # (2, p1+1)
    dN2_m = bspline_basis_and_deriv(span2, η_g, p2, 1, kv2)  # (2, p2+1)
    N1 = dN1_m[1, :];  dN1 = dN1_m[2, :]
    N2 = dN2_m[1, :];  dN2 = dN2_m[2, :]

    ncp_loc = (p1+1) * (p2+1)
    cp_ids  = zeros(Int, ncp_loc)
    Bw   = zeros(ncp_loc);  dBw1 = zeros(ncp_loc);  dBw2 = zeros(ncp_loc)
    Ws = 0.0;  dWs1 = 0.0;  dWs2 = 0.0

    a = 0
    for j_off in 0:p2
        li_eta = span2 - p2 + j_off          # 1-based η local index
        for i_off in 0:p1
            li_xi = span1 - p1 + i_off       # 1-based ξ local index
            a += 1
            local_patch_idx = (li_eta - 1) * n1 + li_xi   # 1-based patch CP
            gcp        = P[pc][local_patch_idx]
            cp_ids[a]  = gcp
            w          = B[gcp, end]
            Bw[a]      = N1[i_off+1] * N2[j_off+1] * w
            dBw1[a]    = dN1[i_off+1] * N2[j_off+1] * w
            dBw2[a]    = N1[i_off+1] * dN2[j_off+1] * w
            Ws += Bw[a];  dWs1 += dBw1[a];  dWs2 += dBw2[a]
        end
    end

    R    = Bw  / Ws
    dR1  = (dBw1 - R * dWs1) / Ws
    dR2  = (dBw2 - R * dWs2) / Ws

    # Physical Jacobian J[spatial_dim, param_dir]
    J = zeros(2, 2)
    for a in 1:ncp_loc
        gcp = cp_ids[a]
        J[1,1] += dR1[a] * B[gcp,1];  J[2,1] += dR1[a] * B[gcp,2]
        J[1,2] += dR2[a] * B[gcp,1];  J[2,2] += dR2[a] * B[gcp,2]
    end
    Ji = inv(J)

    dR_dx = zeros(2, ncp_loc)
    for a in 1:ncp_loc
        dR_dx[:, a] = Ji' * [dR1[a]; dR2[a]]
    end

    B0  = strain_displacement_matrix(nsd, ncp_loc, dR_dx)
    Ue  = zeros(ncp_loc * nsd)
    for a in 1:ncp_loc
        for i in 1:nsd
            Ue[(a-1)*nsd + i] = Ub[cp_ids[a], i]
        end
    end
    σ_v = D * (B0 * Ue)
    return [σ_v[1] σ_v[3]; σ_v[3] σ_v[2]]
end

"""
    solve_cylinder_data(p_ord, exp_level; kwargs...) -> NamedTuple

Run the concentric-cylinder solve and return all mesh/solution data needed
for the reference-solution L2 error computation.  The same keyword arguments
as `solve_cylinder` are accepted.
"""
function solve_cylinder_data(p_ord::Int, exp_level::Int;
    conforming::Bool  = false,
    r_i::Float64      = 1.0,
    r_c::Float64      = 1.5,
    r_o::Float64      = 2.0,
    E::Float64        = 100.0,
    nu::Float64       = 0.3,
    p_o::Float64      = 1.0,
    epss::Float64     = 0.0,
    NQUAD::Int        = p_ord + 1,
    NQUAD_mortar::Int = 10,
    strategy::IntegrationStrategy = ElementBasedIntegration()
)
    nsd = 2; npd = 2; ned = 2; npc = 2; thickness = 1.0

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
    epss_use = epss > 0.0 ? epss : 1e4

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
    for i in axes(B_ref, 1); B_ref[i, 2] > 100.0 && (B_ref[i, 2] -= 1000.0); end

    ncp = size(B_ref, 1)
    nel, nnp, nen = patch_metrics(npc, npd, p_mat, n_mat_ref)
    IEN = build_ien(nsd, npd, npc, p_mat, n_mat_ref, nel, nnp, nen)
    INC = [build_inc(n_mat_ref[pc, :]) for pc in 1:npc]

    dBC = [1 4 2 1 2; 2 2 2 1 2]
    bc_per_dof = dirichlet_bc_control_points(p_mat, n_mat_ref, KV_ref, P_ref,
                                              npd, nnp, ned, dBC)
    neq, ID = build_id(bc_per_dof, ned, ncp)
    LM = build_lm(nen, ned, npc, nel, ID, IEN, P_ref)

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
    Pc = build_interface_cps(pairs, p_mat, n_mat_ref, KV_ref, P_ref, npd, nnp)
    C, Z = build_mortar_coupling(Pc, pairs, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
                                  ID, nnp, ned, nsd, npd, neq, NQUAD_mortar, epss_use,
                                  strategy)
    U, _ = solve_mortar(K_bc, C, Z, F_bc)

    Ub = zeros(ncp, nsd)
    for A in 1:ncp, i in 1:nsd
        eq = ID[i, A];  eq != 0 && (Ub[A, i] = U[eq])
    end

    return (Ub=Ub, p_mat=p_mat, n_mat=n_mat_ref, KV=KV_ref, P=P_ref, B=B_ref,
            IEN=IEN, INC=INC, nel=nel, nen=nen, mats=mats,
            npc=npc, nsd=nsd, npd=npd, NQUAD=NQUAD, thickness=thickness)
end

"""
    l2_stress_error_vs_ref(coarse, ref) -> (err_abs, ref_norm)

Compute the L2 stress error of `coarse` against the reference solution `ref`
by evaluating both at the coarse-mesh Gauss points in parametric space.
Both meshes represent the same approximate geometry, so the geometry
approximation error cancels and only the mortar/discretisation error remains.
"""
function l2_stress_error_vs_ref(coarse, ref)
    npc = coarse.npc;  nsd = coarse.nsd;  npd = coarse.npd
    NQUAD = coarse.NQUAD;  thickness = coarse.thickness
    GPW   = gauss_product(NQUAD, npd)

    err2 = 0.0;  ref2 = 0.0

    for pc in 1:npc
        ien = coarse.IEN[pc];  inc = coarse.INC[pc]
        D   = elastic_constants(coarse.mats[pc], nsd)
        kv1 = coarse.KV[pc][1];  kv2 = coarse.KV[pc][2]

        for el in 1:coarse.nel[pc]
            anchor = ien[el, 1]
            n0     = inc[anchor]

            for (gp, gw) in GPW
                R_s, dR_dx, _, detJ, _ = shape_function(
                    coarse.p_mat[pc,:], coarse.n_mat[pc,:], coarse.KV[pc],
                    coarse.B, coarse.P[pc], gp,
                    coarse.nen[pc], nsd, npd, el, n0, ien, inc)
                detJ <= 0 && continue
                gwJ = gw * detJ * thickness

                # Coarse-mesh stress
                B0    = strain_displacement_matrix(nsd, coarse.nen[pc], dR_dx')
                Ue    = vec(coarse.Ub[coarse.P[pc][ien[el,:]], 1:nsd]')
                σ_h_v = D * (B0 * Ue)
                σ_h   = [σ_h_v[1] σ_h_v[3]; σ_h_v[3] σ_h_v[2]]

                # Global parametric coordinates of this Gauss point.
                # INC stores anchor nc: assembly uses kv[nc] to kv[nc+1] as element span
                # (same convention as shape_function in Geometry.jl)
                ξ_g = 0.5*(kv1[n0[1]] + kv1[n0[1]+1]) +
                      0.5*(kv1[n0[1]+1] - kv1[n0[1]]) * gp[1]
                η_g = 0.5*(kv2[n0[2]] + kv2[n0[2]+1]) +
                      0.5*(kv2[n0[2]+1] - kv2[n0[2]]) * gp[2]

                # Reference stress at same parametric point
                σ_ref = _eval_patch_stress(ξ_g, η_g, pc,
                            ref.p_mat, ref.n_mat, ref.KV, ref.P, ref.B,
                            ref.Ub, D, nsd)

                diff = σ_h - σ_ref
                err2 += dot(diff, diff)    * gwJ
                ref2 += dot(σ_ref, σ_ref) * gwJ
            end
        end
    end
    return sqrt(err2), sqrt(ref2)
end

"""
    run_convergence_cyl_ref(p_ord; exp_range, exp_ref, kwargs...)

Convergence study using a fine-mesh reference solution at `exp_ref` instead
of the analytical Lamé field.  Useful when the geometry cannot be exactly
represented at the given polynomial degree (e.g. p=1).
The L2 stress error is computed against the reference on the same approximate
geometry, so only the mortar coupling and discretisation errors are measured.
"""
function run_convergence_cyl_ref(
    p_ord::Int;
    exp_range::UnitRange = 0:4,
    exp_ref::Int         = 6,
    kwargs...
)
    println("\n=== Convergence vs reference (p=$p_ord, exp_ref=$exp_ref) ===")
    println("(L2 error vs fine-mesh reference — geometry error cancels)")
    @printf("%-6s  %12s  %12s\n", "exp", "‖e‖/‖σ_ref‖", "rate")
    @printf("%s\n", "-"^36)

    @printf("  [computing reference at exp=%d ...]\n", exp_ref)
    ref = solve_cylinder_data(p_ord, exp_ref; kwargs...)

    exps = collect(exp_range)
    prev = NaN
    for exp in exps
        c = solve_cylinder_data(p_ord, exp; kwargs...)
        e_abs, e_ref = l2_stress_error_vs_ref(c, ref)
        rate = isnan(prev) ? NaN : log2(prev / e_abs)
        @printf("exp=%d  %12.4e  %12.3f\n", exp, e_abs / e_ref, rate)
        prev = e_abs
    end
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

# ─────────────────────── Direct p=1 mesh generation ──────────────────────────

"""
    open_uniform_kv(n_elem, p) -> Vector{Float64}

Open (clamped) uniform B-spline knot vector for `n_elem` elements of degree `p`.
Length = n_elem + 2p + 1;  n_cp = n_elem + p control points.
"""
function open_uniform_kv(n_elem::Int, p::Int)::Vector{Float64}
    n_cp = n_elem + p
    kv   = zeros(n_cp + p + 1)
    kv[1:p+1]     .= 0.0
    kv[end-p:end] .= 1.0
    for i in 1:n_elem-1
        kv[p+1+i] = i / n_elem
    end
    return kv
end

"""
    cylinder_geometry_direct_p1(n_ang_p1, n_ang_p2, n_rad; r_i, r_c, r_o)
        -> (B, P, p_mat, n_mat, KV)

Build the two-patch **bilinear** (p=1) geometry for the quarter annular cross-section
by placing control points **directly on the circle arcs** (not via knot insertion).
Each mesh level is generated independently so the geometry converges to the
true annulus at O(h²), faster than the O(h¹) solution error.

  Patch 1: r ∈ [r_i, r_c], θ ∈ [90°→0°]   n_ang_p1 × n_rad bilinear elements
  Patch 2: r ∈ [r_c, r_o], θ ∈ [90°→0°]   n_ang_p2 × n_rad bilinear elements

ξ (angular): ξ=0 ↔ θ=90° (facet 4, ux=0),  ξ=1 ↔ θ=0° (facet 2, uy=0)
η (radial):  η=0 ↔ r_inner (facet 1),        η=1 ↔ r_outer (facet 3)

CP ordering: (η_idx - 1) * (n_ang+1) + ξ_idx   (ξ-fastest, 1-based)
All weights = 1 (non-rational bilinear).
"""
function cylinder_geometry_direct_p1(
    n_ang_p1::Int, n_ang_p2::Int, n_rad::Int;
    r_i::Float64 = 1.0,
    r_c::Float64 = 1.5,
    r_o::Float64 = 2.0
)::Tuple{Matrix{Float64}, Vector{Vector{Int}}, Matrix{Int}, Matrix{Int},
         Vector{Vector{Vector{Float64}}}}

    # Build CP array for one annular sector patch
    function build_patch(n_ang, r_inner, r_outer)
        n_ang_cps = n_ang + 1
        n_rad_cps = n_rad + 1
        B = zeros(n_ang_cps * n_rad_cps, 3)  # [x, y, w]
        for η_idx in 1:n_rad_cps
            r = r_inner + (η_idx - 1) * (r_outer - r_inner) / n_rad
            for ξ_idx in 1:n_ang_cps
                # θ from 90° at ξ_idx=1 to 0° at ξ_idx=n_ang+1
                θ  = (π/2) * (1 - (ξ_idx - 1) / n_ang)
                cp = (η_idx - 1) * n_ang_cps + ξ_idx   # 1-based patch CP index
                B[cp, 1] = r * cos(θ)
                B[cp, 2] = r * sin(θ)
                B[cp, 3] = 1.0   # unit weight (bilinear)
            end
        end
        return B
    end

    B1_xyw = build_patch(n_ang_p1, r_i, r_c)
    B2_xyw = build_patch(n_ang_p2, r_c, r_o)

    # Global B in IGAros format [x, y, z=0, w]
    ncp1 = size(B1_xyw, 1);  ncp2 = size(B2_xyw, 1)
    B_out = zeros(ncp1 + ncp2, 4)
    B_out[1:ncp1,     [1,2,4]] = B1_xyw
    B_out[ncp1+1:end, [1,2,4]] = B2_xyw

    P = [collect(1:ncp1), collect(ncp1+1:ncp1+ncp2)]

    # p=1 in both parametric directions for both patches
    p_mat = fill(1, 2, 2)

    # Number of CPs per patch per direction (patches may differ in angular direction)
    n_mat = [n_ang_p1+1  n_rad+1;
             n_ang_p2+1  n_rad+1]

    # Open uniform knot vectors
    kv_ang_p1 = open_uniform_kv(n_ang_p1, 1)
    kv_ang_p2 = open_uniform_kv(n_ang_p2, 1)
    kv_rad    = open_uniform_kv(n_rad,     1)

    KV = Vector{Vector{Vector{Float64}}}([
        [kv_ang_p1, kv_rad],   # Patch 1
        [kv_ang_p2, kv_rad]    # Patch 2
    ])

    return B_out, P, p_mat, n_mat, KV
end

"""
    solve_cylinder_p1(exp_level; ...) -> NTuple{6, Float64}

Concentric-cylinder benchmark with **p=1 bilinear elements**.
CPs are placed directly on the circle arcs at each refinement level
(not obtained by knot insertion), so geometry error is O(h²) and the
O(h¹) mortar/discretisation error dominates — enabling proper convergence study.

Returns `(l2_stress_rel, l2_stress_abs, l2_disp_rel, l2_disp_abs, en_rel, en_abs)`.

Mesh sizes match the p>1 solver:
  n_ang_p2 = 3·2^exp_level  (Patch 2, angular elements)
  n_ang_p1 = 6·2^exp_level  (Patch 1, non-conforming: 2× finer)
  n_rad    = 2·2^exp_level  (both patches, radial elements)
"""
function solve_cylinder_p1(
    exp_level::Int;
    conforming::Bool      = false,
    r_i::Float64          = 1.0,
    r_c::Float64          = 1.5,
    r_o::Float64          = 2.0,
    E::Float64            = 100.0,
    nu::Float64           = 0.3,
    p_o::Float64          = 1.0,
    epss::Float64         = 1e9,   # large ε avoids resonance bands (ε_bad≈10^(exp+1))
    NQUAD::Int            = 2,          # p+1 = 2 for bilinear
    NQUAD_mortar::Int     = 10,
    strategy::IntegrationStrategy  = ElementBasedIntegration(),
    formulation::FormulationStrategy = TwinMortarFormulation(),
    n_ang_p2_base::Int    = 3,     # base angular elements for outer patch
    n_ang_p1_base::Int    = 6,     # base angular elements for inner patch (non-conforming)
)::NTuple{6, Float64}

    nsd = 2; npd = 2; ned = 2; npc = 2; thickness = 1.0

    n_ang_p2 = n_ang_p2_base * 2^exp_level
    n_ang_p1 = conforming ? n_ang_p2 : n_ang_p1_base * 2^exp_level
    n_rad    = 2 * 2^exp_level
    epss_use = epss   # caller provides the value (default 1e9 is pre-validated)

    # ── Direct geometry (CPs on circle, no krefinement) ───────────────────────
    B, P, p_mat, n_mat, KV = cylinder_geometry_direct_p1(
        n_ang_p1, n_ang_p2, n_rad; r_i=r_i, r_c=r_c, r_o=r_o)

    ncp = size(B, 1)

    # ── Connectivity ──────────────────────────────────────────────────────────
    nel, nnp, nen = patch_metrics(npc, npd, p_mat, n_mat)
    IEN = build_ien(nsd, npd, npc, p_mat, n_mat, nel, nnp, nen)
    INC = [build_inc(n_mat[pc, :]) for pc in 1:npc]

    # ── Dirichlet BCs (symmetry planes) ──────────────────────────────────────
    # Facet 4 (ξ=1, θ=90°): ux = 0   |   Facet 2 (ξ=n, θ=0°): uy = 0
    dBC = [1 4 2 1 2;
           2 2 2 1 2]
    bc_per_dof = dirichlet_bc_control_points(p_mat, n_mat, KV, P,
                                              npd, nnp, ned, dBC)
    neq, ID = build_id(bc_per_dof, ned, ncp)
    LM      = build_lm(nen, ned, npc, nel, ID, IEN, P)

    # ── Stiffness ─────────────────────────────────────────────────────────────
    mats = [LinearElastic(E, nu, :plane_strain), LinearElastic(E, nu, :plane_strain)]
    Ub0  = zeros(ncp, nsd)
    K, _ = build_stiffness_matrix(npc, nsd, npd, ned, neq,
                                   p_mat, n_mat, KV, P, B, Ub0,
                                   nen, nel, IEN, INC, LM, mats, NQUAD, thickness)

    # ── Traction load on Patch 2 facet 3 (outer arc r = r_o) ─────────────────
    stress_fn = (x, y) -> lame_stress(x, y; p_o=p_o, r_i=r_i, r_o=r_o)
    F = zeros(neq)
    F = segment_load(n_mat[2,:], p_mat[2,:], KV[2], P[2], B,
                     nnp[2], nen[2], nsd, npd, ned,
                     Int[], 3, ID, F, stress_fn, thickness, NQUAD)

    K_bc, F_bc = enforce_dirichlet(Tuple{Int,Float64}[], K, F)

    # ── Mortar coupling at curved interface ───────────────────────────────────
    pairs_tm = [InterfacePair(1, 3, 2, 1),
                InterfacePair(2, 1, 1, 3)]
    pairs_sp = [InterfacePair(1, 3, 2, 1)]   # single-pass: slave=patch1 only

    pairs = formulation isa SinglePassFormulation ? pairs_sp : pairs_tm
    Pc    = build_interface_cps(pairs, p_mat, n_mat, KV, P, npd, nnp, formulation)
    C, Z  = build_mortar_coupling(Pc, pairs, p_mat, n_mat, KV, P, B,
                                   ID, nnp, ned, nsd, npd, neq, NQUAD_mortar, epss_use,
                                   strategy, formulation)

    U, _ = solve_mortar(K_bc, C, Z, F_bc)

    # ── L2 stress error vs Lamé ──────────────────────────────────────────────
    l2_abs, l2_ref = l2_stress_error_cyl(
        U, ID, npc, nsd, npd, p_mat, n_mat, KV, P, B,
        nen, nel, IEN, INC, mats, NQUAD, thickness, stress_fn)

    # ── L2 displacement error ─────────────────────────────────────────────────
    disp_fn = (x, y) -> lame_displacement(x, y; p_o=p_o, r_i=r_i, r_o=r_o, E=E, nu=nu)
    d_abs, d_ref = l2_disp_error_cyl(
        U, ID, npc, nsd, npd, p_mat, n_mat, KV, P, B,
        nen, nel, IEN, INC, NQUAD, thickness, disp_fn)

    # ── Energy-norm error ──────────────────────────────────────────────────────
    en_abs, en_ref = energy_error_cyl(
        U, ID, npc, nsd, npd, p_mat, n_mat, KV, P, B,
        nen, nel, IEN, INC, mats, NQUAD, thickness, stress_fn)

    return l2_abs / l2_ref, l2_abs, d_abs / d_ref, d_abs, en_abs / en_ref, en_abs
end

"""
    solve_cylinder_p1_single(exp_level; ...) -> NTuple{6, Float64}

Concentric-cylinder benchmark on a **single p=1 patch** (no interface).
CPs are placed directly on the arcs; n_ang = 3·2^exp angular elements,
n_rad = 4·2^exp radial elements covering r_i → r_o in one patch.
Serves as the reference "no-interface" convergence baseline.
Returns `(l2_stress_rel, l2_stress_abs, l2_disp_rel, l2_disp_abs, en_rel, en_abs)`.
"""
function solve_cylinder_p1_single(
    exp_level::Int;
    r_i::Float64 = 1.0,
    r_o::Float64 = 2.0,
    E::Float64   = 100.0,
    nu::Float64  = 0.3,
    p_o::Float64 = 1.0,
    NQUAD::Int   = 2,
)::NTuple{6, Float64}

    nsd = 2; npd = 2; ned = 2; npc = 1; thickness = 1.0

    n_ang = 3 * 2^exp_level   # angular elements (matches outer patch density)
    n_rad = 4 * 2^exp_level   # radial elements over full r_i → r_o

    # ── CP placement directly on arc (same approach as direct_p1) ─────────────
    n_ang_cps = n_ang + 1
    n_rad_cps = n_rad + 1
    ncp = n_ang_cps * n_rad_cps
    B   = zeros(ncp, 4)   # [x, y, z=0, w]
    for η_idx in 1:n_rad_cps
        r = r_i + (η_idx - 1) * (r_o - r_i) / n_rad
        for ξ_idx in 1:n_ang_cps
            θ  = (π/2) * (1 - (ξ_idx - 1) / n_ang)
            cp = (η_idx - 1) * n_ang_cps + ξ_idx
            B[cp, 1] = r * cos(θ)
            B[cp, 2] = r * sin(θ)
            B[cp, 4] = 1.0
        end
    end
    P = [collect(1:ncp)]

    p_mat = fill(1, 1, 2)
    n_mat = [n_ang_cps  n_rad_cps]   # 1 × 2 matrix
    KV    = [[open_uniform_kv(n_ang, 1), open_uniform_kv(n_rad, 1)]]

    # ── Connectivity ──────────────────────────────────────────────────────────
    nel, nnp, nen = patch_metrics(npc, npd, p_mat, n_mat)
    IEN = build_ien(nsd, npd, npc, p_mat, n_mat, nel, nnp, nen)
    INC = [build_inc(n_mat[1, :])]

    # ── BCs: ux=0 at facet 4 (θ=90°), uy=0 at facet 2 (θ=0°) ───────────────
    dBC = [1 4 1 1;
           2 2 1 1]
    bc_per_dof = dirichlet_bc_control_points(p_mat, n_mat, KV, P, npd, nnp, ned, dBC)
    neq, ID = build_id(bc_per_dof, ned, ncp)
    LM      = build_lm(nen, ned, npc, nel, ID, IEN, P)

    # ── Stiffness ─────────────────────────────────────────────────────────────
    mats = [LinearElastic(E, nu, :plane_strain)]
    Ub0  = zeros(ncp, nsd)
    K, _ = build_stiffness_matrix(npc, nsd, npd, ned, neq,
                                   p_mat, n_mat, KV, P, B, Ub0,
                                   nen, nel, IEN, INC, LM, mats, NQUAD, thickness)

    # ── Traction on facet 3 (outer arc r = r_o) ───────────────────────────────
    stress_fn = (x, y) -> lame_stress(x, y; p_o=p_o, r_i=r_i, r_o=r_o)
    F = zeros(neq)
    F = segment_load(n_mat[1,:], p_mat[1,:], KV[1], P[1], B,
                     nnp[1], nen[1], nsd, npd, ned,
                     Int[], 3, ID, F, stress_fn, thickness, NQUAD)

    K_bc, F_bc = enforce_dirichlet(Tuple{Int,Float64}[], K, F)
    U = linear_solve(K_bc, F_bc)

    # ── Errors vs Lamé solution ───────────────────────────────────────────────
    l2_abs, l2_ref = l2_stress_error_cyl(
        U, ID, npc, nsd, npd, p_mat, n_mat, KV, P, B,
        nen, nel, IEN, INC, mats, NQUAD, thickness, stress_fn)
    disp_fn = (x, y) -> lame_displacement(x, y; p_o=p_o, r_i=r_i, r_o=r_o, E=E, nu=nu)
    d_abs, d_ref = l2_disp_error_cyl(
        U, ID, npc, nsd, npd, p_mat, n_mat, KV, P, B,
        nen, nel, IEN, INC, NQUAD, thickness, disp_fn)
    en_abs, en_ref = energy_error_cyl(
        U, ID, npc, nsd, npd, p_mat, n_mat, KV, P, B,
        nen, nel, IEN, INC, mats, NQUAD, thickness, stress_fn)

    return l2_abs / l2_ref, l2_abs, d_abs / d_ref, d_abs, en_abs / en_ref, en_abs
end

"""
    run_convergence_cyl_p1(; exp_range, conforming, kwargs...)

Convergence study for p=1 bilinear elements on the concentric-cylinder benchmark.
Control points are placed directly on the circle arcs at each level, so the
geometry error O(h²) is below the O(h¹) mortar/discretisation error.
"""
function run_convergence_cyl_p1(;
    exp_range::UnitRange = 0:5,
    kwargs...
)
    println("\n=== p=1 convergence (CPs on circle arcs, direct mesh generation) ===")
    @printf("%-6s  %7s  %7s  %12s  %12s\n",
            "exp", "n_ang_P2", "n_rad", "‖e‖/‖σ‖", "rate")
    @printf("%s\n", "-"^52)

    exps = collect(exp_range)
    prev = NaN
    for exp in exps
        n_ang_p2 = 3 * 2^exp
        n_rad    = 2 * 2^exp
        l2_rel, _, _, _ = solve_cylinder_p1(exp; kwargs...)
        rate = isnan(prev) ? NaN : log2(prev / l2_rel)
        @printf("exp=%-2d  %7d  %7d  %12.4e  %12.3f\n",
                exp, n_ang_p2, n_rad, l2_rel, rate)
        prev = l2_rel
    end
end

# ─────────────────────── NQUAD integration accuracy sweep ────────────────────

"""
    run_nquad_sweep_disp(p_ord, exp_level; nquad_range, configs, epss, kwargs...)
        -> Dict{String, Vector{Float64}}

Sweep NQUAD_mortar and return the relative L2-displacement error for each
configuration (formulation × integration strategy).  The mesh is fixed at
the given `p_ord` and `exp_level` (non-conforming 2:1 ratio, as in the paper).

Returns a Dict label → Vector of errors (one per NQUAD in `nquad_range`).
"""
function run_nquad_sweep_disp(
    p_ord::Int     = 2,
    exp_level::Int = 3;
    nquad_range    = 1:p_ord+5,
    epss::Float64  = 1e6,
    configs = [
        ("TM-Seg", TwinMortarFormulation(), SegmentBasedIntegration()),
        ("TM-Elm", TwinMortarFormulation(), ElementBasedIntegration()),
        ("DP-Seg", DualPassFormulation(),   SegmentBasedIntegration()),
        ("DP-Elm", DualPassFormulation(),   ElementBasedIntegration()),
    ],
    kwargs...
)::Dict{String, Vector{Float64}}

    println("\n=== NQUAD mortar sweep: p=$p_ord, exp=$exp_level, ε=$epss ===")
    @printf("%-8s", "NQUAD")
    for (label,_,_) in configs;  @printf("  %12s", label);  end;  println()
    @printf("%s\n", "-"^(8 + 14*length(configs)))

    data = Dict(label => Float64[] for (label,_,_) in configs)

    for nq in nquad_range
        @printf("NQUAD=%-2d", nq)
        for (label, form, strat) in configs
            _, _, d_rel, _ = solve_cylinder(p_ord, exp_level;
                epss=epss, NQUAD_mortar=nq, strategy=strat, formulation=form, kwargs...)
            push!(data[label], d_rel)
            @printf("  %12.4e", d_rel)
        end
        println()
    end
    return data
end

# ─────────────────────── System conditioning study ───────────────────────────

"""
    compute_kappa(K, C, Z) -> Float64

Condition number κ(A) of the full augmented saddle-point matrix
    A = [K_bc   C  ]
        [C^T   -Z  ]
via dense SVD.  Practical for neq + n_mult ≲ 3000.

System convention: [K, C; Cᵀ, -Z] with C of size (neq × n_mult).

Key observations:
  • SP (Z=0): κ(A_SP) ≈ 10²² — effectively singular in double precision.
    The near-null space of [K, C; Cᵀ, 0] drives σ_min → 0 on curved
    non-conforming interfaces (no Z to regularise the multiplier block).
  • TM (Z≠0): κ(A_TM) ≈ 10¹⁰ — 12 orders of magnitude better.
    The Z block (ε-weighted mortar mass, negative semidefinite) fills the
    near-null space and provides the missing regularisation.
"""
function compute_kappa(
    K_bc::SparseMatrixCSC{Float64,Int},
    C::AbstractMatrix{Float64},
    Z::AbstractMatrix{Float64}
)::Float64
    Kd = Matrix(K_bc); Cd = Matrix(C); Zd = Matrix(Z)
    A = [Kd Cd; Cd' -Zd]
    sv = svdvals(A)
    return sv[1] / sv[end]
end

"""
    cylinder_p1_kappa(exp_level; formulation, strategy, epss, conforming)
        -> (κ, n_mult)

Build the p=1 cylinder system (without solving) and return:
  κ       = condition number κ(A) of the full augmented matrix [K, C; Cᵀ, -Z]
  n_mult  = number of Lagrange multiplier DOFs
"""
function cylinder_p1_kappa(
    exp_level::Int;
    formulation::FormulationStrategy = TwinMortarFormulation(),
    strategy::IntegrationStrategy    = ElementBasedIntegration(),
    epss::Float64      = 1e9,
    conforming::Bool   = false,
    r_i::Float64 = 1.0, r_c::Float64 = 1.5, r_o::Float64 = 2.0,
    E::Float64   = 100.0, nu::Float64 = 0.3, p_o::Float64 = 1.0,
    NQUAD::Int = 2, NQUAD_mortar::Int = 10,
    n_ang_p2_base::Int = 3,
    n_ang_p1_base::Int = 6,
)::Tuple{Float64,Int}

    nsd = 2; npd = 2; ned = 2; npc = 2; thickness = 1.0

    n_ang_p2 = n_ang_p2_base * 2^exp_level
    n_ang_p1 = conforming ? n_ang_p2 : n_ang_p1_base * 2^exp_level
    n_rad    = 2 * 2^exp_level

    B, P, p_mat, n_mat, KV = cylinder_geometry_direct_p1(
        n_ang_p1, n_ang_p2, n_rad; r_i=r_i, r_c=r_c, r_o=r_o)
    ncp = size(B, 1)

    nel, nnp, nen = patch_metrics(npc, npd, p_mat, n_mat)
    IEN = build_ien(nsd, npd, npc, p_mat, n_mat, nel, nnp, nen)
    INC = [build_inc(n_mat[pc, :]) for pc in 1:npc]

    dBC = [1 4 2 1 2; 2 2 2 1 2]
    bc_per_dof = dirichlet_bc_control_points(p_mat, n_mat, KV, P, npd, nnp, ned, dBC)
    neq, ID = build_id(bc_per_dof, ned, ncp)
    LM = build_lm(nen, ned, npc, nel, ID, IEN, P)

    mats = [LinearElastic(E, nu, :plane_strain), LinearElastic(E, nu, :plane_strain)]
    K, _ = build_stiffness_matrix(npc, nsd, npd, ned, neq,
                                   p_mat, n_mat, KV, P, B, zeros(ncp, nsd),
                                   nen, nel, IEN, INC, LM, mats, NQUAD, thickness)

    stress_fn = (x, y) -> lame_stress(x, y; p_o=p_o, r_i=r_i, r_o=r_o)
    F = zeros(neq)
    F = segment_load(n_mat[2,:], p_mat[2,:], KV[2], P[2], B,
                     nnp[2], nen[2], nsd, npd, ned,
                     Int[], 3, ID, F, stress_fn, thickness, NQUAD)
    K_bc, _ = enforce_dirichlet(Tuple{Int,Float64}[], K, F)

    pairs_tm = [InterfacePair(1, 3, 2, 1), InterfacePair(2, 1, 1, 3)]
    pairs_sp = [InterfacePair(1, 3, 2, 1)]
    pairs = formulation isa SinglePassFormulation ? pairs_sp : pairs_tm

    Pc = build_interface_cps(pairs, p_mat, n_mat, KV, P, npd, nnp, formulation)
    C, Z = build_mortar_coupling(Pc, pairs, p_mat, n_mat, KV, P, B,
                                  ID, nnp, ned, nsd, npd, neq, NQUAD_mortar, epss,
                                  strategy, formulation)

    κ = compute_kappa(K_bc, C, Z)
    return κ, size(C, 2)
end

"""
    cylinder_kappa(p_ord, exp_level; formulation, strategy, epss, ...) -> (κ, n_mult)

Build the cylinder system for arbitrary p (p≥2 uses k-refinement) and return
the condition number κ(A) of the full augmented matrix [K, C; Cᵀ, -Z].
For p=1, delegates to `cylinder_p1_kappa`.
"""
function cylinder_kappa(
    p_ord::Int, exp_level::Int;
    formulation::FormulationStrategy = TwinMortarFormulation(),
    strategy::IntegrationStrategy    = ElementBasedIntegration(),
    epss::Float64      = 1e6,
    conforming::Bool   = false,
    r_i::Float64 = 1.0, r_c::Float64 = 1.5, r_o::Float64 = 2.0,
    E::Float64   = 100.0, nu::Float64 = 0.3, p_o::Float64 = 1.0,
    NQUAD::Int          = p_ord + 1,
    NQUAD_mortar::Int   = p_ord + 2,
    n_ang_p2_base::Int  = 3,
    n_ang_p1_base::Int  = 6,
)::Tuple{Float64,Int}

    if p_ord == 1
        return cylinder_p1_kappa(exp_level;
            formulation=formulation, strategy=strategy, epss=epss,
            conforming=conforming, r_i=r_i, r_c=r_c, r_o=r_o,
            E=E, nu=nu, p_o=p_o, NQUAD=NQUAD, NQUAD_mortar=NQUAD_mortar,
            n_ang_p2_base=n_ang_p2_base, n_ang_p1_base=n_ang_p1_base)
    end

    nsd = 2; npd = 2; ned = 2; npc = 2; thickness = 1.0

    # ── Geometry (same as solve_cylinder) ──────────────────────────────────
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

    epss_use = epss > 0.0 ? epss : 1e4

    kref_data = Vector{Float64}[
        vcat([1.0, 1.0], conforming ? u_ang : u_ang_nc),
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

    # ── Connectivity + BCs ────────────────────────────────────────────────
    nel, nnp, nen = patch_metrics(npc, npd, p_mat, n_mat_ref)
    IEN = build_ien(nsd, npd, npc, p_mat, n_mat_ref, nel, nnp, nen)

    dBC = [1 4 2 1 2; 2 2 2 1 2]
    bc_per_dof = dirichlet_bc_control_points(p_mat, n_mat_ref, KV_ref, P_ref,
                                              npd, nnp, ned, dBC)
    neq, ID = build_id(bc_per_dof, ned, ncp)
    LM      = build_lm(nen, ned, npc, nel, ID, IEN, P_ref)

    # ── Stiffness ─────────────────────────────────────────────────────────
    mats = [LinearElastic(E, nu, :plane_strain), LinearElastic(E, nu, :plane_strain)]
    INC  = [build_inc(n_mat_ref[pc, :]) for pc in 1:npc]
    K, _ = build_stiffness_matrix(npc, nsd, npd, ned, neq,
                                   p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
                                   zeros(ncp, nsd),
                                   nen, nel, IEN, INC, LM, mats, NQUAD, thickness)
    K_bc, _ = enforce_dirichlet(Tuple{Int,Float64}[], K, zeros(neq))

    # ── Mortar coupling ───────────────────────────────────────────────────
    pairs_tm = [InterfacePair(1, 3, 2, 1), InterfacePair(2, 1, 1, 3)]
    pairs_sp = [InterfacePair(1, 3, 2, 1)]
    pairs = formulation isa SinglePassFormulation ? pairs_sp : pairs_tm

    Pc = build_interface_cps(pairs, p_mat, n_mat_ref, KV_ref, P_ref, npd, nnp, formulation)
    C, Z = build_mortar_coupling(Pc, pairs, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
                                  ID, nnp, ned, nsd, npd, neq, NQUAD_mortar, epss_use,
                                  strategy, formulation)

    κ = compute_kappa(K_bc, C, Z)
    return κ, size(C, 2)
end

"""
    cylinder_matrices(p_ord, exp_level; epss, ...) -> (K_bc, C, Z)

Build the cylinder system and return the raw matrices for spectral analysis.
For p=1 uses direct mesh; for p≥2 uses k-refinement.
"""
function cylinder_matrices(
    p_ord::Int, exp_level::Int;
    formulation::FormulationStrategy = TwinMortarFormulation(),
    strategy::IntegrationStrategy    = ElementBasedIntegration(),
    epss::Float64      = 1e6,
    conforming::Bool   = false,
    r_i::Float64 = 1.0, r_c::Float64 = 1.5, r_o::Float64 = 2.0,
    E::Float64   = 100.0, nu::Float64 = 0.3,
    NQUAD::Int          = p_ord + 1,
    NQUAD_mortar::Int   = p_ord + 2,
    n_ang_p2_base::Int  = 3,
    n_ang_p1_base::Int  = 6,
)
    nsd = 2; npd = 2; ned = 2; npc = 2; thickness = 1.0

    if p_ord == 1
        n_ang_p2 = n_ang_p2_base * 2^exp_level
        n_ang_p1 = conforming ? n_ang_p2 : n_ang_p1_base * 2^exp_level
        n_rad    = 2 * 2^exp_level
        B, P, p_mat, n_mat, KV = cylinder_geometry_direct_p1(
            n_ang_p1, n_ang_p2, n_rad; r_i=r_i, r_c=r_c, r_o=r_o)
        n_mat_ref = n_mat; KV_ref = KV; B_ref = B; P_ref = P
    else
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
        kref_data = Vector{Float64}[
            vcat([1.0, 1.0], conforming ? u_ang : u_ang_nc),
            vcat([2.0, 1.0], u_ang),
            vcat([1.0, 2.0], u_rad),
            vcat([2.0, 2.0], u_rad),
        ]
        B0_hack = copy(B0); B0_hack[P[1], 2] .+= 1000.0
        n_mat_ref, _, KV_ref, B_hack_ref, P_ref = krefinement(
            nsd, npd, npc, n_mat, p_mat, KV, B0_hack, P, kref_data)
        B_ref = copy(B_hack_ref)
        for i in axes(B_ref, 1)
            B_ref[i, 2] > 100.0 && (B_ref[i, 2] -= 1000.0)
        end
    end

    ncp = size(B_ref, 1)
    nel, nnp, nen = patch_metrics(npc, npd, p_mat, n_mat_ref)
    IEN = build_ien(nsd, npd, npc, p_mat, n_mat_ref, nel, nnp, nen)
    dBC = [1 4 2 1 2; 2 2 2 1 2]
    bc_per_dof = dirichlet_bc_control_points(p_mat, n_mat_ref, KV_ref, P_ref,
                                              npd, nnp, ned, dBC)
    neq, ID = build_id(bc_per_dof, ned, ncp)
    LM      = build_lm(nen, ned, npc, nel, ID, IEN, P_ref)
    mats = [LinearElastic(E, nu, :plane_strain), LinearElastic(E, nu, :plane_strain)]
    INC  = [build_inc(n_mat_ref[pc, :]) for pc in 1:npc]
    K, _ = build_stiffness_matrix(npc, nsd, npd, ned, neq,
                                   p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
                                   zeros(ncp, nsd),
                                   nen, nel, IEN, INC, LM, mats, NQUAD, thickness)
    K_bc, _ = enforce_dirichlet(Tuple{Int,Float64}[], K, zeros(neq))

    epss_use = epss > 0.0 ? epss : 1e4
    pairs_tm = [InterfacePair(1, 3, 2, 1), InterfacePair(2, 1, 1, 3)]
    pairs_sp = [InterfacePair(1, 3, 2, 1)]
    pairs = formulation isa SinglePassFormulation ? pairs_sp : pairs_tm
    Pc = build_interface_cps(pairs, p_mat, n_mat_ref, KV_ref, P_ref, npd, nnp, formulation)
    C, Z = build_mortar_coupling(Pc, pairs, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
                                  ID, nnp, ned, nsd, npd, neq, NQUAD_mortar, epss_use,
                                  strategy, formulation)
    return K_bc, C, Z
end

"""
    run_kappa_study(; exp_range, epss) — print table of κ(A) for all four configs.
"""
function run_kappa_study(; exp_range::UnitRange = 0:3, epss::Float64 = 1e9)
    configs = [
        ("SP-Seg", SinglePassFormulation(), SegmentBasedIntegration()),
        ("SP-Elm", SinglePassFormulation(), ElementBasedIntegration()),
        ("TM-Seg", TwinMortarFormulation(), SegmentBasedIntegration()),
        ("TM-Elm", TwinMortarFormulation(), ElementBasedIntegration()),
    ]
    println("\n=== Condition number κ(A) of augmented system [K, C; Cᵀ, -Z] ===")
    println("(SP: Z=0 ⟹ effectively singular; TM: Z≠0 regularises the system)")
    @printf("%-6s", "exp")
    for (l,_,_) in configs; @printf("  %12s", "κ($l)"); end
    println()
    @printf("%s\n", "-"^(6 + 14*length(configs)))
    for e in exp_range
        h = 0.5 / 2^e
        @printf("exp=%-2d  h=%.4f", e, h)
        for (_, form, strat) in configs
            κ, _ = cylinder_p1_kappa(e; formulation=form, strategy=strat, epss=epss)
            @printf("  %12.3e", κ)
        end
        println()
    end
end

"""
    run_formulation_comparison_p1(; exp_range, epss_tm, epss_sp, kwargs...)

Four-way convergence comparison for p=1 on the concentric-cylinder benchmark:
  1. Single-pass + segment-based integration  (exact single-pass reference)
  2. Single-pass + element-based integration  (variational crime, no cancellation)
  3. Twin Mortar + segment-based integration  (exact dual-pass reference)
  4. Twin Mortar + element-based integration  (dual-pass with error cancellation)

Reports relative L² stress error and energy-norm error with convergence rates.
"""
function run_formulation_comparison_p1(;
    exp_range::UnitRange  = 0:5,
    epss_tm::Float64      = 1e9,   # Twin Mortar ε (avoids resonance for p=1)
    epss_sp::Float64      = 0.0,   # unused for single-pass (Z=0), kept for signature
    NQUAD_mortar::Int     = 10,
    kwargs...
)
    configs = [
        ("SP-Seg", SinglePassFormulation(), SegmentBasedIntegration(), epss_tm),
        ("SP-Elm", SinglePassFormulation(), ElementBasedIntegration(), epss_tm),
        ("TM-Seg", TwinMortarFormulation(), SegmentBasedIntegration(), epss_tm),
        ("TM-Elm", TwinMortarFormulation(), ElementBasedIntegration(), epss_tm),
    ]

    exps = collect(exp_range)

    for (label, form, strat, eps_val) in configs
        println("\n=== $label (p=1) ===")
        @printf("%-6s  %12s  %6s  %12s  %6s\n",
                "exp", "L2-rel", "rate", "E-norm-rel", "rate")
        @printf("%s\n", "-"^50)

        prev_l2 = NaN;  prev_en = NaN
        for exp in exps
            n_ang_p2 = 3 * 2^exp
            n_rad    = 2 * 2^exp
            l2_rel, _, _, _, en_rel, _ = solve_cylinder_p1(exp;
                strategy=strat, formulation=form,
                epss=eps_val, NQUAD_mortar=NQUAD_mortar, kwargs...)
            rate_l2 = isnan(prev_l2) ? NaN : log2(prev_l2 / l2_rel)
            rate_en = isnan(prev_en) ? NaN : log2(prev_en / en_rel)
            @printf("exp=%-2d  %12.4e  %6.2f  %12.4e  %6.2f\n",
                    exp, l2_rel, rate_l2, en_rel, rate_en)
            prev_l2 = l2_rel;  prev_en = en_rel
        end
    end
end

# ─────────────────────── Performance / cost study ────────────────────────────

"""
    _cyl_setup(p_ord, exp_level; kwargs...) -> NamedTuple

Build the full cylinder mesh and system matrices (stiffness, load, Dirichlet
enforcement, interface connectivity) without the mortar assembly step.
Returns all arguments needed by `build_mortar_coupling` and `solve_mortar`,
plus pre-computed ancillary data (mesh size, error-checking helpers).

Used by `run_cost_study` to time only the mortar assembly in isolation.
"""
function _cyl_setup(
    p_ord::Int,
    exp_level::Int;
    r_i::Float64 = 1.0,
    r_c::Float64 = 1.5,
    r_o::Float64 = 2.0,
    E::Float64   = 100.0,
    nu::Float64  = 0.3,
    p_o::Float64 = 1.0,
    epss::Float64         = 1e6,
    NQUAD::Int            = p_ord + 1,
    NQUAD_mortar::Int     = p_ord + 2,
    n_ang_p2_base::Int    = 3,
    n_ang_p1_base::Int    = 6,
)
    nsd = 2; npd = 2; ned = 2; npc = 2
    thickness = 1.0

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

    pairs_tm = [InterfacePair(1, 3, 2, 1), InterfacePair(2, 1, 1, 3)]
    pairs_sp = [InterfacePair(1, 3, 2, 1)]

    # Angular element counts at the interface for QP counting
    n_iface_inner = n_ang_p1_base * 2^exp_level   # inner patch (finer), slave of pair 1
    n_iface_outer = n_ang_p2_base * 2^exp_level   # outer patch (coarser), slave of pair 2

    return (
        p_mat=p_mat, n_mat_ref=n_mat_ref, KV_ref=KV_ref, P_ref=P_ref, B_ref=B_ref,
        ID=ID, nnp=nnp, ned=ned, nsd=nsd, npd=npd, neq=neq,
        epss=epss, NQUAD_mortar=NQUAD_mortar,
        pairs_tm=pairs_tm, pairs_sp=pairs_sp,
        K_bc=K_bc, F_bc=F_bc,
        n_iface_inner=n_iface_inner, n_iface_outer=n_iface_outer,
        IEN=IEN, INC=INC, mats=mats, nel=nel, nen=nen, NQUAD=NQUAD,
        thickness=thickness, stress_fn=stress_fn,
    )
end

"""
    _count_seg_qp(d, pairs, NQUAD_mortar) -> Int

Count total interface quadrature points for segment-based integration.
Calls `find_interface_segments_1d` for each pair and returns
sum of (n_segments × NQUAD_mortar).
"""
function _count_seg_qp(d, pairs::Vector{InterfacePair}, NQUAD_mortar::Int)
    total = 0
    for pair in pairs
        spc = pair.slave_patch; sfacet = pair.slave_facet
        mpc = pair.master_patch; mfacet = pair.master_facet

        ps_s_vec, ns_s_vec, KVs, Ps, = get_segment_patch(
            d.p_mat[spc,:], d.n_mat_ref[spc,:], d.KV_ref[spc],
            d.P_ref[spc], d.npd, d.nnp[spc], sfacet)
        ps_m_vec, ns_m_vec, KVm, Pm, = get_segment_patch(
            d.p_mat[mpc,:], d.n_mat_ref[mpc,:], d.KV_ref[mpc],
            d.P_ref[mpc], d.npd, d.nnp[mpc], mfacet)

        breaks = find_interface_segments_1d(
            ps_s_vec[1], ns_s_vec[1], KVs[1],
            ps_m_vec[1], ns_m_vec[1], KVm[1],
            d.B_ref, Ps, Pm, d.nsd)
        total += (length(breaks) - 1) * NQUAD_mortar
    end
    return total
end

"""
    run_cost_study(; degrees, exp_levels, n_repeats, kwargs...)

Interface-assembly cost comparison for the concentric-cylinder benchmark.

For each (p, exp_level) combination, builds the mesh and system matrices once
(untimed), then measures the wall-clock time of `build_mortar_coupling` for
three method configurations:
  - TM-Elm : Twin Mortar, element-based integration  (two half-passes)
  - SP-Elm : Single Pass, element-based integration  (one pass)
  - SP-Seg : Single Pass, segment-based integration  (one pass + clipping)

The first call for each configuration is discarded (JIT warm-up). Subsequent
calls (n_repeats) are timed with `@elapsed`; the minimum is reported.

Prints a table with: p, exp, n_interface_elem, method, QP_count,
assembly_time_ms, and the relative L²-stress error (TM-Elm only).
"""
function run_cost_study(;
    degrees::Vector{Int}   = [2, 3, 4],
    exp_levels::Vector{Int} = [2, 3],
    n_repeats::Int         = 5,
    epss::Float64          = 1e6,
    kwargs...
)
    configs = [
        ("TM-Elm", TwinMortarFormulation(), ElementBasedIntegration()),
        ("SP-Elm", SinglePassFormulation(), ElementBasedIntegration()),
        ("SP-Seg", SinglePassFormulation(), SegmentBasedIntegration()),
    ]

    hdr = @sprintf("%-5s  %-3s  %-8s  %-7s  %-6s  %-12s  %-10s",
                   "p", "exp", "n_iface", "method", "n_QP", "t_assem_ms", "L2_rel")
    println("\n=== Interface assembly cost study (concentric cylinders) ===")
    println(hdr)
    println("-"^length(hdr))

    for p_ord in degrees
        for exp in exp_levels
            NQUAD_mortar = p_ord + 2
            d = _cyl_setup(p_ord, exp; epss=epss, NQUAD_mortar=NQUAD_mortar, kwargs...)

            # Reference L2 error (TM-Elm, one timed solve)
            Pc_tm = build_interface_cps(d.pairs_tm, d.p_mat, d.n_mat_ref,
                                        d.KV_ref, d.P_ref, d.npd, d.nnp,
                                        TwinMortarFormulation())
            C_tm, Z_tm = build_mortar_coupling(Pc_tm, d.pairs_tm,
                d.p_mat, d.n_mat_ref, d.KV_ref, d.P_ref, d.B_ref,
                d.ID, d.nnp, d.ned, d.nsd, d.npd, d.neq,
                NQUAD_mortar, d.epss, ElementBasedIntegration(), TwinMortarFormulation())
            U_tm, _ = solve_mortar(d.K_bc, C_tm, Z_tm, d.F_bc)
            s_abs, s_ref = l2_stress_error_cyl(
                U_tm, d.ID, 2, d.nsd, d.npd, d.p_mat, d.n_mat_ref,
                d.KV_ref, d.P_ref, d.B_ref, d.nen, d.nel,
                d.IEN, d.INC, d.mats, d.NQUAD, d.thickness, d.stress_fn)
            l2_rel = s_abs / s_ref

            for (label, form, strat) in configs
                pairs = form isa TwinMortarFormulation ? d.pairs_tm : d.pairs_sp
                Pc = build_interface_cps(pairs, d.p_mat, d.n_mat_ref,
                                         d.KV_ref, d.P_ref, d.npd, d.nnp, form)

                # QP count
                n_qp = if strat isa ElementBasedIntegration
                    if form isa TwinMortarFormulation
                        (d.n_iface_inner + d.n_iface_outer) * NQUAD_mortar
                    else
                        d.n_iface_inner * NQUAD_mortar
                    end
                else
                    _count_seg_qp(d, pairs, NQUAD_mortar)
                end

                # JIT warm-up
                build_mortar_coupling(Pc, pairs, d.p_mat, d.n_mat_ref, d.KV_ref,
                    d.P_ref, d.B_ref, d.ID, d.nnp, d.ned, d.nsd, d.npd, d.neq,
                    NQUAD_mortar, d.epss, strat, form)

                # Timed runs — minimum over n_repeats
                t_min = minimum(
                    @elapsed(build_mortar_coupling(Pc, pairs,
                        d.p_mat, d.n_mat_ref, d.KV_ref, d.P_ref, d.B_ref,
                        d.ID, d.nnp, d.ned, d.nsd, d.npd, d.neq,
                        NQUAD_mortar, d.epss, strat, form))
                    for _ in 1:n_repeats)

                l2_str = label == "TM-Elm" ? @sprintf("%.2e", l2_rel) : "—"
                n_iface = d.n_iface_inner   # interface = inner patch facet boundary
                @printf("%-5d  %-3d  %-8d  %-7s  %-6d  %-12.2f  %-10s\n",
                        p_ord, exp, n_iface, label, n_qp, t_min * 1000, l2_str)
            end
            println()
        end
    end
end

# ─────────────────────── C–Z cancellation test ──────────────────────────────

"""
    run_cz_cancellation(p_ord, exp_level; nquad_range, epss)

Solution-level test of the integration-error cancellation mechanism (§4.5).

**Idea**: Quadrature error in M enters both C and Z of the Twin Mortar
system.  When multipliers are eliminated, the error partially cancels in
C Z⁻¹ Cᵀ because the perturbation appears in both "numerator" and
"denominator".  For single-pass mortar (Z=0), no such cancellation exists.

**Method**:
1. Build reference solutions with segment-based integration (no quadrature
   error) for both TM and SP formulations.
2. For each NQUAD in `nquad_range`, solve with element-based integration.
3. Report:
   - `δC`     : relative Frobenius-norm error in coupling matrix C
   - `δU_TM`  : relative L2 displacement error vs segment-based TM reference
   - `δU_SP`  : relative L2 displacement error vs segment-based SP reference
   - `gain`   : δU_SP / δU_TM  (>1 means TM cancellation helps)

The displacement errors isolate the quadrature contribution by comparing
element-based to segment-based solutions (same discretization, same ε).
"""
function run_cz_cancellation(
    p_ord::Int     = 2,
    exp_level::Int = 3;
    nquad_range    = 1:p_ord+5,
    epss::Float64  = 1e6,
)
    d = _cyl_setup(p_ord, exp_level; epss=epss)

    # ── Helper: build C, Z and solve for a given (formulation, strategy, NQUAD) ──
    function _build_solve(form, strat, nq)
        pairs = form isa SinglePassFormulation ? d.pairs_sp : d.pairs_tm
        Pc = build_interface_cps(pairs, d.p_mat, d.n_mat_ref, d.KV_ref,
                                  d.P_ref, d.npd, d.nnp, form)
        C, Z = build_mortar_coupling(
            Pc, pairs, d.p_mat, d.n_mat_ref, d.KV_ref, d.P_ref, d.B_ref,
            d.ID, d.nnp, d.ned, d.nsd, d.npd, d.neq, nq, d.epss,
            strat, form)
        U, _ = solve_mortar(d.K_bc, C, Z, d.F_bc)
        return U, C
    end

    # ── Reference solutions: segment-based (exact integration) ───────────
    U_tm_ref, C_tm_ref = _build_solve(TwinMortarFormulation(),  SegmentBasedIntegration(), p_ord+2)
    U_sp_ref, C_sp_ref = _build_solve(SinglePassFormulation(),  SegmentBasedIntegration(), p_ord+2)
    nC_tm_ref = norm(C_tm_ref)
    nC_sp_ref = norm(C_sp_ref)

    # ── Sweep NQUAD with element-based integration ───────────────────────
    dC_tm_vec  = Float64[]
    dC_sp_vec  = Float64[]
    dU_tm_vec  = Float64[]
    dU_sp_vec  = Float64[]
    gain_vec   = Float64[]

    println("\n=== C–Z cancellation test: p=$p_ord, exp=$exp_level, ε=$epss ===")
    @printf("%-6s  %10s  %10s  %10s  %10s  %10s\n",
            "NQUAD", "δC_TM", "δC_SP", "δU_TM", "δU_SP", "gain")
    @printf("%s\n", "-"^68)

    for nq in nquad_range
        U_tm, C_tm = _build_solve(TwinMortarFormulation(),  ElementBasedIntegration(), nq)
        U_sp, C_sp = _build_solve(SinglePassFormulation(),  ElementBasedIntegration(), nq)

        dC_tm = norm(C_tm - C_tm_ref) / nC_tm_ref
        dC_sp = norm(C_sp - C_sp_ref) / nC_sp_ref
        dU_tm = norm(U_tm - U_tm_ref) / norm(U_tm_ref)
        dU_sp = norm(U_sp - U_sp_ref) / norm(U_sp_ref)
        gain  = dU_sp > 0 ? dU_sp / dU_tm : NaN

        push!(dC_tm_vec, dC_tm);  push!(dC_sp_vec, dC_sp)
        push!(dU_tm_vec, dU_tm);  push!(dU_sp_vec, dU_sp)
        push!(gain_vec, gain)

        @printf("%-6d  %10.2e  %10.2e  %10.2e  %10.2e  %10.2f\n",
                nq, dC_tm, dC_sp, dU_tm, dU_sp, gain)
    end

    return (nquad=collect(nquad_range),
            dC_tm=dC_tm_vec, dC_sp=dC_sp_vec,
            dU_tm=dU_tm_vec, dU_sp=dU_sp_vec,
            gain=gain_vec)
end

# ─────────────────────── Force-moment helper ──────────────────────────────────

"""
    compute_force_moments(D, M, s_cps, m_cps, B; dim=2)

Compute δ_0, δ_1, δ_2 with uniform test multiplier λ = 1.
See cz_cancellation.jl for detailed documentation.
"""
function compute_force_moments(D, M, s_cps, m_cps, B; dim::Int = 2)
    Dd = Matrix(D);  Md = Matrix(M)
    ns = length(s_cps)
    λ = ones(ns)
    ones_s = ones(ns);  ones_m = ones(length(m_cps))
    y_s  = [B[cp, dim] for cp in s_cps];  y_m  = [B[cp, dim] for cp in m_cps]
    y2_s = y_s .^ 2;                       y2_m = y_m .^ 2
    δ_0 = dot(λ, Dd * ones_s - Md * ones_m)
    δ_1 = dot(λ, Dd * y_s    - Md * y_m)
    δ_2 = dot(λ, Dd * y2_s   - Md * y2_m)
    return (δ_0 = δ_0, δ_1 = δ_1, δ_2 = δ_2)
end

# ─────────────────────── Force-moment analysis (curved interface) ─────────────

"""
    run_moment_table_cyl(p_ord, exp_level; nquad_range, epss)

Force-moment equilibrium errors δ_0, δ_1, δ_2 on the curved cylinder
interface.  Uses `build_mortar_mass_matrices` (element-based only) with
varying NQUAD and a high-NQUAD reference.

The coordinate dimension for the moments is the arc-length (y-coordinate
of CPs in physical space).  Both spatial dimensions (x, y) are tested
and the maximum absolute moment is reported.
"""
function run_moment_table_cyl(
    p_ord::Int     = 2,
    exp_level::Int = 2;
    nquad_range    = 1:p_ord+3,
    epss::Float64  = 1e6,
)
    d = _cyl_setup(p_ord, exp_level; epss=epss)

    pair1 = d.pairs_tm[1]   # pass 1: slave=inner, master=outer
    pair2 = d.pairs_tm[2]   # pass 2: slave=outer, master=inner

    strat = ElementBasedIntegration()
    nquad_ref = 20

    println("\n=== Force-moment analysis: cylinder p=$p_ord, exp=$exp_level ===")
    println("Moments with uniform test multiplier λ = 1")
    println("Reporting max |δ_k| over spatial dimensions x, y\n")

    for nq in nquad_range
        println("─── NQUAD = $nq ───")
        @printf("%-8s  %12s  %12s  %14s  %14s  %14s\n",
                "Method", "δ₀", "δ₁", "δ₂ (pass 1)", "δ₂ (pass 2)", "δ₂ (sum)")
        @printf("%s\n", "─"^80)

        for (label, nq_use, two_pass) in [
            ("SP-ref",  nquad_ref, false),
            ("SPME",    nq,        false),
            ("DP-ref",  nquad_ref, true),
            ("TM",      nq,        true),
        ]
            D1, M12, s1, m1 = build_mortar_mass_matrices(
                pair1, d.p_mat, d.n_mat_ref, d.KV_ref, d.P_ref, d.B_ref,
                d.nnp, d.nsd, d.npd, nq_use, strat)

            # Compute moments for both spatial dimensions, take max
            mom1_x = compute_force_moments(D1, M12, s1, m1, d.B_ref; dim=1)
            mom1_y = compute_force_moments(D1, M12, s1, m1, d.B_ref; dim=2)
            δ0 = max(abs(mom1_x.δ_0), abs(mom1_y.δ_0))
            δ1 = max(abs(mom1_x.δ_1), abs(mom1_y.δ_1))
            δ2_1 = max(abs(mom1_x.δ_2), abs(mom1_y.δ_2))

            if two_pass
                D2, M21, s2, m2 = build_mortar_mass_matrices(
                    pair2, d.p_mat, d.n_mat_ref, d.KV_ref, d.P_ref, d.B_ref,
                    d.nnp, d.nsd, d.npd, nq_use, strat)
                mom2_x = compute_force_moments(D2, M21, s2, m2, d.B_ref; dim=1)
                mom2_y = compute_force_moments(D2, M21, s2, m2, d.B_ref; dim=2)
                δ2_2 = max(abs(mom2_x.δ_2), abs(mom2_y.δ_2))
                # Sum: use signed values for the dominant dimension
                sum_x = mom1_x.δ_2 + mom2_x.δ_2
                sum_y = mom1_y.δ_2 + mom2_y.δ_2
                δ2_sum = max(abs(sum_x), abs(sum_y))

                @printf("%-8s  %12.2e  %12.2e  %14.4e  %14.4e  %14.4e\n",
                        label, δ0, δ1, δ2_1, δ2_2, δ2_sum)
            else
                @printf("%-8s  %12.2e  %12.2e  %14.4e  %14s  %14s\n",
                        label, δ0, δ1, δ2_1, "—", "—")
            end
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
