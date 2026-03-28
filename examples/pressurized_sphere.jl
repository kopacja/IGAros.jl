# examples/pressurized_sphere.jl
#
# Internally pressurized thick sphere — 3D Twin Mortar benchmark.
# Two-patch solid sphere octant (first octant x,y,z ≥ 0).
# Non-conforming mesh: Patch 1 (inner, r_i→r_c) has mesh_ratio× as many angular
# elements as Patch 2 (outer, r_c→r_o) in both angular directions.
# Default parameters match Puso & Solberg (2020) §10.3: Ri=1, Ro=1.4, E=1, ν=0.3.
#
# Exact Lamé solution (radially symmetric): σ_rr, σ_θθ=σ_φφ.
# L2 stress error vs h convergence for degrees p = 2, 3, 4.
#
# Parametrization convention per patch:
#   ξ (dir 1): polar angle θ from equatorial x-axis to z-axis  (0 → π/2)
#   η (dir 2): azimuthal angle φ from xz-plane to yz-plane     (0 → π/2)
#   ζ (dir 3): radial coordinate r from r_inner to r_outer
#
# Facet labels used:
#   Face 1 (ζ=1, inner radial):  Patch 1 inner-radius sphere surface
#   Face 6 (ζ=n₃, outer radial): Patch 1 interface (at r_c); Patch 2 outer load
#   Face 1 (ζ=1, inner radial):  Patch 2 interface (at r_c)
#
# Symmetry BCs (zero normal displacement on the 3 coordinate planes):
#   Face 5 (η=1, φ=0, xz-plane):  uy = 0  (on both patches)
#   Face 3 (η=n₂, φ=π/2, yz-plane): ux = 0 (on both patches)
#   Face 2 (ξ=n₁, θ=π/2, z-axis):  ux=uy=0 (degenerate — handled via geometry)
#   Actually: we need ux=0 on the yz-plane, uy=0 on the xz-plane, uz=0 on the xy-plane.
#
# NOTE: The degenerate pole edge (θ=π/2, z-axis) is handled naturally by NURBS.

import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using IGAros
using LinearAlgebra, SparseArrays, Printf

# ─────────────────────── Lamé exact solution (3D sphere) ─────────────────────

"""
    lame_stress_sphere(x, y, z; p_i, r_i, r_o) -> Matrix{Float64}(3×3)

Exact Lamé Cauchy stress for a thick sphere under internal pressure p_i
(zero external pressure). Returns the 3×3 Cartesian stress tensor.
"""
function lame_stress_sphere(x::Real, y::Real, z::Real;
                            p_i::Float64, r_i::Float64, r_o::Float64)::Matrix{Float64}
    r  = sqrt(x^2 + y^2 + z^2)
    r  = max(r, 1e-14)   # guard against degenerate pole
    A  = p_i * r_i^3 / (r_o^3 - r_i^3)
    σ_rr = A * (1.0 - r_o^3 / r^3)
    σ_tt = A * (1.0 + r_o^3 / (2.0 * r^3))   # σ_θθ = σ_φφ (hoop)

    # Cartesian: σ = σ_rr r̂⊗r̂ + σ_tt (I − r̂⊗r̂)
    rx, ry, rz = x/r, y/r, z/r
    R = [rx; ry; rz]
    σ = σ_tt * I(3) + (σ_rr - σ_tt) * (R * R')
    return σ
end

"""
    lame_displacement_sphere(x, y, z; p_i, r_i, r_o, E, nu) -> (ux, uy, uz)

Exact Lamé radial displacement (3D sphere, internal pressure).
"""
function lame_displacement_sphere(x::Real, y::Real, z::Real;
                                  p_i::Float64, r_i::Float64, r_o::Float64,
                                  E::Float64,   nu::Float64)
    r   = sqrt(x^2 + y^2 + z^2)
    r   = max(r, 1e-14)
    den = E * (r_o^3 - r_i^3)
    C1  = p_i * r_i^3 * (1 - 2nu) / den    # coefficient of r
    C2  = p_i * r_i^3 * r_o^3 * (1 + nu) / (2den) # coefficient of 1/r²
    u_r = C1 * r + C2 / r^2
    return u_r * x/r, u_r * y/r, u_r * z/r
end

# ─────────────────────── Bezier degree elevation ──────────────────────────────

"""
    bezier_elevate_3d(Bh) -> Bh_elevated

Elevate a Bezier curve by one degree. Works identically to `bezier_elevate`
in concentric_cylinders.jl; defined here to keep the sphere example self-contained.
"""
function bezier_elevate_3d(Bh::Matrix{Float64})::Matrix{Float64}
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

# ─────────────────────── Sphere octant geometry ───────────────────────────────

"""
    sphere_geometry(p_ord; r_i, r_c, r_o) -> (B, P)

Build the two-patch NURBS geometry for a solid sphere octant.

  Patch 1: r ∈ [r_i, r_c], first octant (x,y,z ≥ 0)
  Patch 2: r ∈ [r_c, r_o], first octant (x,y,z ≥ 0)

Parametrization:
  ξ (dir 1): θ from 0 (equatorial x-direction) to π/2 (z-pole)
  η (dir 2): φ from 0 (xz-plane) to π/2 (yz-plane)
  ζ (dir 3): r from r_inner to r_outer (linear, elevated to p_ord)

CP ordering: ξ inner, η middle, ζ outer (consistent with nurbs_coords convention).

Returns B (ncp × 4, [x, y, z, w]) and P (patch-to-global CP index list).
"""
function sphere_geometry(p_ord::Int;
                         r_i::Float64 = 1.0,
                         r_c::Float64 = 1.2,
                         r_o::Float64 = 1.4)::Tuple{Matrix{Float64}, Vector{Vector{Int}}}

    ws = 1.0 / sqrt(2.0)   # weight for 45° arc midpoint (exact NURBS circle)

    # ── Quadratic sphere surface at radius r (3×3 CPs, ξ-inner η-outer) ──────
    # CP ordering: flat index = (η_idx-1)*3 + ξ_idx, η_idx=1..3, ξ_idx=1..3
    function base_surface_q2(r)
        # Profile arc (xz-plane, θ from 0 to π/2):
        # i=1: (x=0,z=r) θ=0 → north-pole edge; i=2: (x=r,z=r) corner; i=3: (x=r,z=0)
        arc_xz = [(0.0, r, 1.0),   # (x, z, w_ξ)
                  (r,   r, ws),
                  (r,   0.0, 1.0)]
        # Revolution weights (φ from 0 to π/2):
        # j=1: (cos=1,sin=0,w=1); j=2: (cos=1,sin=1,w=ws) corner; j=3: (cos=0,sin=1,w=1)
        arc_phi = [(1.0, 0.0, 1.0),   # (cφ, sφ, w_η)
                   (1.0, 1.0, ws),
                   (0.0, 1.0, 1.0)]

        B = zeros(9, 4)
        for j in 1:3
            cφ, sφ, wη = arc_phi[j]
            for i in 1:3
                xr, zr, wξ = arc_xz[i]
                k = (j-1)*3 + i
                B[k, 1] = xr * cφ   # x = r*sin(θ)*cos(φ)
                B[k, 2] = xr * sφ   # y = r*sin(θ)*sin(φ)
                B[k, 3] = zr         # z = r*cos(θ)
                B[k, 4] = wξ * wη   # weight
            end
        end
        return B  # (9, 4): [x, y, z, w]
    end

    # ── Elevate surface from (2,2) to (p_ord,p_ord) ──────────────────────────
    function elevate_surface(B3x3)
        n_ang = p_ord + 1   # CPs per angular direction after elevation

        # Step 1: elevate in ξ direction (rows of 3 CPs for each η level)
        B_xi = zeros(n_ang * 3, 4)
        for j in 1:3
            row = B3x3[(j-1)*3+1 : j*3, :]
            Bh  = copy(row)
            Bh[:, 1:3] .*= Bh[:, 4:4]          # to homogeneous
            for _ in 3:p_ord; Bh = bezier_elevate_3d(Bh); end
            Bh[:, 1:3] ./= Bh[:, 4:4]          # back to Euclidean
            B_xi[(j-1)*n_ang+1 : j*n_ang, :] = Bh
        end
        # Now B_xi has ξ-inner ordering: index = (η-1)*n_ang + ξ, η ∈ 1:3

        # Step 2: elevate in η direction (columns of 3 CPs for each ξ level)
        B_full = zeros(n_ang * n_ang, 4)
        for i in 1:n_ang
            # Extract η-column at ξ=i: rows i, n_ang+i, 2*n_ang+i
            col = B_xi[i : n_ang : 2*n_ang+i, :]   # (3, 4)
            Bh  = copy(col)
            Bh[:, 1:3] .*= Bh[:, 4:4]
            for _ in 3:p_ord; Bh = bezier_elevate_3d(Bh); end
            Bh[:, 1:3] ./= Bh[:, 4:4]
            for j in 1:n_ang
                B_full[(j-1)*n_ang + i, :] = Bh[j, :]
            end
        end
        return B_full  # (n_ang^2, 4): ξ-inner, η-outer
    end

    # ── Build solid patch by blending two sphere surfaces in ζ (radial) ──────
    # Linear blend in Euclidean coords is exact (weights identical at both radii,
    # physical coords scale linearly with radius → zero quadrature error).
    function build_solid_patch(surf_a, surf_b)
        n_ang = p_ord + 1
        n_rad = p_ord + 1
        n_surf = n_ang^2
        B_solid = zeros(n_surf * n_rad, 4)
        for k in 0:n_rad-1
            t = (n_rad > 1) ? k / (n_rad - 1) : 0.0
            B_solid[k*n_surf+1 : (k+1)*n_surf, :] =
                (1 - t) .* surf_a .+ t .* surf_b
        end
        return B_solid  # (n_ang^2 * n_rad, 4): ξ-inner, η-mid, ζ-outer
    end

    # ── Assemble surfaces and patches ────────────────────────────────────────
    surf_i = elevate_surface(base_surface_q2(r_i))   # inner radius r_i
    surf_c = elevate_surface(base_surface_q2(r_c))   # interface r_c
    surf_o = elevate_surface(base_surface_q2(r_o))   # outer radius r_o

    B1_solid = build_solid_patch(surf_i, surf_c)   # Patch 1: r_i → r_c
    B2_solid = build_solid_patch(surf_c, surf_o)   # Patch 2: r_c → r_o

    ncp1 = size(B1_solid, 1)
    ncp2 = size(B2_solid, 1)
    B_out = vcat(B1_solid, B2_solid)   # (ncp1+ncp2, 4): [x, y, z, w]
    P = [collect(1:ncp1), collect(ncp1+1:ncp1+ncp2)]
    return B_out, P
end

# ─────────────────────── L2 stress error (3D) ────────────────────────────────

"""
    l2_stress_error_sphere(U, ID, npc, nsd, npd, p, n, KV, P, B,
                           nen, nel, IEN, INC, materials, NQUAD,
                           stress_fn) -> (err_abs, err_ref)

Compute L2 stress error for the sphere example. `stress_fn(x,y,z)` returns
the exact 3×3 Cauchy stress matrix.
"""
function l2_stress_error_sphere(
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
    stress_fn::Function   # (x,y,z) -> 3×3 exact stress matrix
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

                gwJ = gw * detJ

                # FEM stress (Voigt): [σ_xx, σ_yy, σ_zz, σ_xy, σ_yz, σ_zx]
                B0    = strain_displacement_matrix(nsd, nen[pc], dR_dx')
                Ue    = vec(Ub[P[pc][ien[el,:]], 1:nsd]')
                σ_h_v = D * (B0 * Ue)

                # FEM stress as 3×3 matrix
                σ_h_m = [σ_h_v[1] σ_h_v[4] σ_h_v[6];
                          σ_h_v[4] σ_h_v[2] σ_h_v[5];
                          σ_h_v[6] σ_h_v[5] σ_h_v[3]]

                # Exact stress at physical coordinates
                Xe = B[P[pc][ien[el,:]], :]
                X  = Xe' * R_s
                σ_ex = stress_fn(X[1], X[2], X[3])

                diff_m = σ_h_m - σ_ex
                err2 += dot(diff_m, diff_m) * gwJ
                ref2 += dot(σ_ex,   σ_ex)   * gwJ
            end
        end
    end

    return sqrt(err2), sqrt(ref2)
end

# ─────────────────────── L2 displacement error ───────────────────────────────

"""
    l2_disp_error_sphere(U, ID, npc, nsd, npd, p, n, KV, P, B,
                         nen, nel, IEN, INC, NQUAD, disp_fn) -> (err_abs, err_ref)

L2 displacement error: err² = ∫ ||u_h − u_ex||² dΩ.
disp_fn(x,y,z) must return (ux,uy,uz).
"""
function l2_disp_error_sphere(
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
    disp_fn::Function
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
            anchor = ien[el, 1]; n0 = inc[anchor]
            for (gp, gw) in GPW
                R_s, _, _, detJ, _ = shape_function(
                    p[pc,:], n[pc,:], KV[pc], B, P[pc], gp,
                    nen[pc], nsd, npd, el, n0, ien, inc)
                detJ <= 0 && continue
                gwJ = gw * detJ
                u_h = (Ub[P[pc][ien[el,:]], 1:nsd])' * R_s
                Xe  = B[P[pc][ien[el,:]], :]
                X   = Xe' * R_s
                ux_ex, uy_ex, uz_ex = disp_fn(X[1], X[2], X[3])
                u_ex = [ux_ex, uy_ex, uz_ex]
                diff = u_h[1:nsd] - u_ex
                err2 += dot(diff, diff) * gwJ
                ref2 += dot(u_ex, u_ex) * gwJ
            end
        end
    end
    return sqrt(err2), sqrt(ref2)
end

# ─────────────────────── Energy-norm error ───────────────────────────────────

"""
    energy_error_sphere(U, ID, npc, nsd, npd, p, n, KV, P, B,
                        nen, nel, IEN, INC, materials, NQUAD,
                        stress_fn) -> (err_abs, err_ref)

Energy-norm error: err² = ∫ (σ_h − σ_ex) : D⁻¹ : (σ_h − σ_ex) dΩ.
stress_fn(x,y,z) must return a 3×3 symmetric stress tensor.
"""
function energy_error_sphere(
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
    stress_fn::Function
)::Tuple{Float64, Float64}

    ncp = size(B, 1)
    Ub  = zeros(ncp, nsd)
    for A in 1:ncp, i in 1:nsd
        eq = ID[i, A]; eq != 0 && (Ub[A, i] = U[eq])
    end

    err2 = 0.0; ref2 = 0.0
    GPW  = gauss_product(NQUAD, npd)

    for pc in 1:npc
        ien  = IEN[pc]; inc = INC[pc]
        D    = elastic_constants(materials[pc], nsd)
        Dinv = inv(D)
        for el in 1:nel[pc]
            anchor = ien[el, 1]; n0 = inc[anchor]
            for (gp, gw) in GPW
                R_s, dR_dx, _, detJ, _ = shape_function(
                    p[pc,:], n[pc,:], KV[pc], B, P[pc], gp,
                    nen[pc], nsd, npd, el, n0, ien, inc)
                detJ <= 0 && continue
                gwJ = gw * detJ
                B0    = strain_displacement_matrix(nsd, nen[pc], dR_dx')
                Ue    = vec(Ub[P[pc][ien[el,:]], 1:nsd]')
                σ_h_v = D * (B0 * Ue)
                Xe = B[P[pc][ien[el,:]], :]
                X  = Xe' * R_s
                σ_ex   = stress_fn(X[1], X[2], X[3])
                σ_ex_v = [σ_ex[1,1], σ_ex[2,2], σ_ex[3,3],
                           σ_ex[1,2], σ_ex[2,3], σ_ex[1,3]]
                Δσ_v = σ_h_v - σ_ex_v
                err2 += dot(Δσ_v, Dinv * Δσ_v) * gwJ
                ref2 += dot(σ_ex_v, Dinv * σ_ex_v) * gwJ
            end
        end
    end
    return sqrt(err2), sqrt(ref2)
end

# ─────────────────────── Diagnostic solver ───────────────────────────────────

"""
    solve_sphere_diag(p_ord, exp_level; kwargs...) -> NamedTuple

Like `solve_sphere` but returns L2-displacement, L2-stress, and energy-norm
errors, plus the KKT components (K, C, Z).
"""
function solve_sphere_diag(
    p_ord::Int, exp_level::Int;
    r_i::Float64 = 1.0, r_c::Float64 = 1.2, r_o::Float64 = 1.4,
    E::Float64 = 1.0, nu::Float64 = 0.3, p_i::Float64 = 0.01,
    mesh_ratio::Float64 = 2.0,
    epss::Float64 = 0.0,
    NQUAD::Int = p_ord + 1, NQUAD_mortar::Int = p_ord + 2,
    formulation::FormulationStrategy = TwinMortarFormulation(),
    strategy::IntegrationStrategy = ElementBasedIntegration(),
)
    nsd = 3; npd = 3; ned = 3; npc = 2

    if p_ord == 1
        n_ang_o = 2^(exp_level + 1); n_ang_i = round(Int, mesh_ratio * n_ang_o); n_rad = 2^(exp_level + 1)
        B_ref, P_ref = sphere_geometry_direct_p1(n_ang_i, n_ang_o, n_rad;
                                                   r_i=r_i, r_c=r_c, r_o=r_o)
        ncp = size(B_ref, 1)
        p_mat = fill(1, npc, npd)
        n_mat_ref = [n_ang_i+1  n_ang_i+1  n_rad+1;
                     n_ang_o+1  n_ang_o+1  n_rad+1]
        kv_i   = open_uniform_kv(n_ang_i, 1)
        kv_o   = open_uniform_kv(n_ang_o, 1)
        kv_rad = open_uniform_kv(n_rad,   1)
        KV_ref = Vector{Vector{Vector{Float64}}}([
            [kv_i, kv_i, kv_rad], [kv_o, kv_o, kv_rad]])
        epss_use = epss > 0.0 ? epss : 100.0
    else
        B0, P  = sphere_geometry(p_ord; r_i=r_i, r_c=r_c, r_o=r_o)
        p_mat  = fill(p_ord, npc, npd)
        n_mat  = fill(p_ord + 1, npc, npd)
        KV     = generate_knot_vectors(npc, npd, p_mat, n_mat)
        n_ang = 2^exp_level; n_rad = 2^exp_level; n_ang_inner = round(Int, mesh_ratio * n_ang)
        u_ang_o = Float64[i/n_ang       for i in 1:n_ang-1]
        u_ang_i = Float64[i/n_ang_inner for i in 1:n_ang_inner-1]
        u_rad   = Float64[i/n_rad       for i in 1:n_rad-1]
        kref_data = Vector{Float64}[
            vcat([1.0,1.0],u_ang_i), vcat([1.0,2.0],u_ang_i), vcat([1.0,3.0],u_rad),
            vcat([2.0,1.0],u_ang_o), vcat([2.0,2.0],u_ang_o), vcat([2.0,3.0],u_rad)]
        n_mat_ref, _, KV_ref, B_ref, P_ref = krefinement(
            nsd, npd, npc, n_mat, p_mat, KV, B0, P, kref_data)
        ncp = size(B_ref, 1)
        epss_use = epss > 0.0 ? epss : 100.0
    end

    nel, nnp, nen = patch_metrics(npc, npd, p_mat, n_mat_ref)
    IEN = build_ien(nsd, npd, npc, p_mat, n_mat_ref, nel, nnp, nen)
    INC = [build_inc(n_mat_ref[pc, :]) for pc in 1:npc]

    dBC = [1 3 2 1 2; 1 4 2 1 2; 2 4 2 1 2; 2 5 2 1 2; 3 2 2 1 2]
    bc_per_dof = dirichlet_bc_control_points(p_mat, n_mat_ref, KV_ref, P_ref,
                                              npd, nnp, ned, dBC)
    neq, ID = build_id(bc_per_dof, ned, ncp)
    LM      = build_lm(nen, ned, npc, nel, ID, IEN, P_ref)

    mats = [LinearElastic(E, nu, :three_d), LinearElastic(E, nu, :three_d)]
    Ub0  = zeros(ncp, nsd)
    K, _ = build_stiffness_matrix(npc, nsd, npd, ned, neq,
                                   p_mat, n_mat_ref, KV_ref, P_ref, B_ref, Ub0,
                                   nen, nel, IEN, INC, LM, mats, NQUAD, 1.0)

    stress_fn = (x, y, z) -> lame_stress_sphere(x, y, z; p_i=p_i, r_i=r_i, r_o=r_o)
    F = zeros(neq)
    F = segment_load(n_mat_ref[1,:], p_mat[1,:], KV_ref[1], P_ref[1], B_ref,
                     nnp[1], nen[1], nsd, npd, ned,
                     Int[], 1, ID, F, stress_fn, 1.0, NQUAD)
    K_bc, F_bc = enforce_dirichlet(Tuple{Int,Float64}[], K, F)

    pairs = formulation isa SinglePassFormulation ?
        [InterfacePair(1,6,2,1)] :
        [InterfacePair(1,6,2,1), InterfacePair(2,1,1,6)]
    Pc = build_interface_cps(pairs, p_mat, n_mat_ref, KV_ref, P_ref, npd, nnp, formulation)
    C, Z = build_mortar_coupling(Pc, pairs, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
                                  ID, nnp, ned, nsd, npd, neq, NQUAD_mortar, epss_use,
                                  strategy, formulation)
    U, lam = solve_mortar(K_bc, C, Z, F_bc)

    # L2 stress error
    σ_abs, σ_ref = l2_stress_error_sphere(
        U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
        nen, nel, IEN, INC, mats, NQUAD, stress_fn)

    # L2 displacement error
    disp_fn = (x, y, z) -> lame_displacement_sphere(x, y, z;
        p_i=p_i, r_i=r_i, r_o=r_o, E=E, nu=nu)
    l2_abs, l2_ref = l2_disp_error_sphere(
        U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
        nen, nel, IEN, INC, NQUAD, disp_fn)

    # Energy-norm error
    en_abs, en_ref = energy_error_sphere(
        U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
        nen, nel, IEN, INC, mats, NQUAD, stress_fn)

    return (U=U, lam=lam, K=K_bc, C=C, Z=Z, F=F_bc,
            σ_abs=σ_abs, σ_ref=σ_ref, σ_rel=σ_abs/σ_ref,
            l2_abs=l2_abs, l2_ref=l2_ref, l2_rel=l2_abs/l2_ref,
            en_abs=en_abs, en_ref=en_ref, en_rel=en_abs/en_ref)
end

# ─────────────────────── Single-level solve ───────────────────────────────────

"""
    solve_sphere(p_ord, exp_level; conforming, mesh_ratio, epss, ...) -> NamedTuple

Run one refinement level of the pressurized-sphere benchmark (3D Twin Mortar).
Default parameters match Puso & Solberg (2020) §10.3: Ri=1, Ro=1.4, E=1, ν=0.3.
Returns `(σ_rel, σ_abs, l2_rel, l2_abs, en_rel, en_abs)`.
"""
function solve_sphere(
    p_ord::Int,
    exp_level::Int;
    conforming::Bool           = false,
    mesh_ratio::Float64        = 2.0,
    r_i::Float64               = 1.0,
    r_c::Float64               = 1.2,
    r_o::Float64               = 1.4,
    E::Float64                 = 1.0,
    nu::Float64                = 0.3,
    p_i::Float64               = 0.01,
    epss::Float64              = 0.0,     # 0 → auto: 1.0
    NQUAD::Int                 = p_ord + 1,
    NQUAD_mortar::Int          = p_ord + 2,
    strategy::IntegrationStrategy = ElementBasedIntegration(),
    formulation::FormulationStrategy = TwinMortarFormulation(),
    vtk_prefix::String         = "",
    n_vis::Int                 = 4
)

    nsd = 3; npd = 3; ned = 3; npc = 2

    # ── Initial coarse geometry ───────────────────────────────────────────────
    B0, P = sphere_geometry(p_ord; r_i=r_i, r_c=r_c, r_o=r_o)
    p_mat = fill(p_ord, npc, npd)
    n_mat = fill(p_ord + 1, npc, npd)
    KV    = generate_knot_vectors(npc, npd, p_mat, n_mat)

    # ── h-refinement ─────────────────────────────────────────────────────────
    # Angular: n_ang elements in outer, mesh_ratio*n_ang in inner (non-conforming)
    # Radial:  n_rad elements in both patches
    n_ang = 2^(exp_level + 2)   # 4 at exp=0 (matches P&S §10.3 baseline)
    n_rad = 2^exp_level         # 1 at exp=0 (1 element through thickness per patch)
    n_ang_inner = conforming ? n_ang : round(Int, mesh_ratio * n_ang)

    u_ang_o = Float64[i/n_ang       for i in 1:n_ang-1]
    u_ang_i = Float64[i/n_ang_inner for i in 1:n_ang_inner-1]
    u_rad   = Float64[i/n_rad       for i in 1:n_rad-1]

    kref_data = Vector{Float64}[
        vcat([1.0, 1.0], u_ang_i),   # Patch 1, ξ direction (angular)
        vcat([1.0, 2.0], u_ang_i),   # Patch 1, η direction (angular)
        vcat([1.0, 3.0], u_rad),     # Patch 1, ζ direction (radial)
        vcat([2.0, 1.0], u_ang_o),   # Patch 2, ξ direction (angular)
        vcat([2.0, 2.0], u_ang_o),   # Patch 2, η direction (angular)
        vcat([2.0, 3.0], u_rad),     # Patch 2, ζ direction (radial)
    ]

    n_mat_ref, _, KV_ref, B_ref, P_ref = krefinement(
        nsd, npd, npc, n_mat, p_mat, KV, B0, P, kref_data
    )
    ncp = size(B_ref, 1)
    epss_use = epss > 0.0 ? epss : 100.0

    # ── Connectivity ──────────────────────────────────────────────────────────
    nel, nnp, nen = patch_metrics(npc, npd, p_mat, n_mat_ref)
    IEN = build_ien(nsd, npd, npc, p_mat, n_mat_ref, nel, nnp, nen)
    INC = [build_inc(n_mat_ref[pc, :]) for pc in 1:npc]

    # ── Dirichlet BCs (symmetry planes for first-octant sphere) ─────────────
    # Parametrization: ξ goes from z-axis pole (θ=0, ξ=1) to equatorial (θ=π/2, ξ=n₁)
    #                  η goes from xz-plane (φ=0, η=1) to yz-plane (φ=π/2, η=n₂)
    #
    # Symmetry planes:
    #   z=0 (equatorial, xy-plane): uz=0  → facet 2 (ξ=n₁)
    #   y=0 (xz-plane):             uy=0  → facet 5 (η=1)
    #   x=0 (yz-plane):             ux=0  → facet 3 (η=n₂)
    #
    # Additionally, the z-axis edge (facet 4, ξ=1) has x=y=0 by geometry.
    # Interior z-axis CPs (1 < η < n₂) are not covered by facets 3 or 5 alone,
    # so we must explicitly apply ux=uy=0 on the entire z-axis face (facet 4).
    dBC = [1 3 2 1 2;    # ux=0 on yz-plane:   facet 3 (η=n₂), patches 1&2
           1 4 2 1 2;    # ux=0 on z-axis face: facet 4 (ξ=1),  patches 1&2
           2 4 2 1 2;    # uy=0 on z-axis face: facet 4 (ξ=1),  patches 1&2
           2 5 2 1 2;    # uy=0 on xz-plane:   facet 5 (η=1),   patches 1&2
           3 2 2 1 2]    # uz=0 on equatorial:  facet 2 (ξ=n₁),  patches 1&2
    bc_per_dof = dirichlet_bc_control_points(p_mat, n_mat_ref, KV_ref, P_ref,
                                              npd, nnp, ned, dBC)
    neq, ID = build_id(bc_per_dof, ned, ncp)
    LM      = build_lm(nen, ned, npc, nel, ID, IEN, P_ref)

    # ── Stiffness matrix ──────────────────────────────────────────────────────
    mats = [LinearElastic(E, nu, :three_d), LinearElastic(E, nu, :three_d)]
    Ub0  = zeros(ncp, nsd)
    K, _ = build_stiffness_matrix(npc, nsd, npd, ned, neq,
                                   p_mat, n_mat_ref, KV_ref, P_ref, B_ref, Ub0,
                                   nen, nel, IEN, INC, LM, mats, NQUAD, 1.0)

    # ── Load: internal pressure on Patch 1 face 1 (ζ=1, r=r_i) ─────────────
    # The exact traction on the inner surface is t = σ · (-n̂) = -σ_rr r̂
    # (inward normal → outward traction is compressive for positive p_i)
    stress_fn = (x, y, z) -> lame_stress_sphere(x, y, z; p_i=p_i, r_i=r_i, r_o=r_o)
    F = zeros(neq)
    # Patch 1, facet 1 (ζ=1, inner radius surface)
    F = segment_load(n_mat_ref[1,:], p_mat[1,:], KV_ref[1], P_ref[1], B_ref,
                     nnp[1], nen[1], nsd, npd, ned,
                     Int[], 1, ID, F, stress_fn, 1.0, NQUAD)

    K_bc, F_bc = enforce_dirichlet(Tuple{Int,Float64}[], K, F)

    # ── Mortar coupling at spherical interface r = r_c ──────────────────────
    # Patch 1 face 6 (ζ=n₃, outer radial) ↔ Patch 2 face 1 (ζ=1, inner radial)
    pairs = formulation isa SinglePassFormulation ?
        [InterfacePair(1, 6, 2, 1)] :
        [InterfacePair(1, 6, 2, 1), InterfacePair(2, 1, 1, 6)]
    Pc = build_interface_cps(pairs, p_mat, n_mat_ref, KV_ref, P_ref, npd, nnp, formulation)

    C, Z = build_mortar_coupling(Pc, pairs, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
                                  ID, nnp, ned, nsd, npd, neq, NQUAD_mortar, epss_use,
                                  strategy, formulation)

    # ── Solve ─────────────────────────────────────────────────────────────────
    U, _ = solve_mortar(K_bc, C, Z, F_bc)

    # ── L2 stress error ───────────────────────────────────────────────────────
    σ_abs, σ_ref = l2_stress_error_sphere(
        U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
        nen, nel, IEN, INC, mats, NQUAD, stress_fn
    )

    # ── L2 displacement error ─────────────────────────────────────────────────
    disp_fn = (x, y, z) -> lame_displacement_sphere(x, y, z;
        p_i=p_i, r_i=r_i, r_o=r_o, E=E, nu=nu)
    l2_abs, l2_ref = l2_disp_error_sphere(
        U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
        nen, nel, IEN, INC, NQUAD, disp_fn
    )

    # ── Energy-norm error ─────────────────────────────────────────────────────
    en_abs, en_ref = energy_error_sphere(
        U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
        nen, nel, IEN, INC, mats, NQUAD, stress_fn
    )

    # ── Optional VTK export ────────────────────────────────────────────────
    if !isempty(vtk_prefix)
        write_vtk_sphere(vtk_prefix, U, ID, npc, nsd, npd,
                          p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
                          nen, IEN, INC, E, nu;
                          n_vis=n_vis, p_i=p_i, r_i=r_i, r_o=r_o)
    end

    return (σ_rel=σ_abs/σ_ref, σ_abs=σ_abs,
            l2_rel=l2_abs/l2_ref, l2_abs=l2_abs,
            en_rel=en_abs/en_ref, en_abs=en_abs)
end

# ─── VTK postprocessing ───────────────────────────────────────────────────────

# Sample a knot vector at element boundaries + n_per_span interior points per span.
if !@isdefined(_kv_sample)
    function _kv_sample(kv_vec::AbstractVector{Float64}, n_per_span::Int)::Vector{Float64}
        breaks = unique(kv_vec)
        pts = Float64[]
        for i in 1:length(breaks)-1
            append!(pts, range(breaks[i], breaks[i+1]; length = n_per_span + 1)[1:end-1])
        end
        push!(pts, breaks[end])
        return pts
    end
end

"""
    write_vtk_sphere(prefix, U, ID, npc, nsd, npd, p_mat, n_mat, KV, P, B,
                     nen_vec, IEN, INC, E, nu; n_vis=4,
                     p_i=0.01, r_i=1.0, r_o=1.4)

Write one VTK STRUCTURED_GRID file per patch:  `prefix_1.vtk`, `prefix_2.vtk`, ...

Point data: displacement (ux,uy,uz), stress components (σ_xx,σ_yy,σ_zz,τ_xy,τ_yz,τ_zx),
von Mises stress, radial stress σ_rr, and exact Lamé σ_rr for comparison.
"""
function write_vtk_sphere(
    prefix::String,
    U::Vector{Float64},
    ID::Matrix{Int},
    npc::Int, nsd::Int, npd::Int,
    p_mat::Matrix{Int}, n_mat::Matrix{Int},
    KV::Vector{<:AbstractVector{<:AbstractVector{Float64}}},
    P::Vector{Vector{Int}},
    B::Matrix{Float64},
    nen_vec::Vector{Int},
    IEN::Vector{Matrix{Int}},
    INC::Vector{<:AbstractVector{<:AbstractVector{Int}}},
    E::Float64, nu::Float64;
    n_vis::Int = 4,
    p_i::Float64 = 0.01,
    r_i::Float64 = 1.0,
    r_o::Float64 = 1.4
)
    ncp = size(B, 1)

    # Displacement at all control points
    Ub = zeros(ncp, nsd)
    for A in 1:ncp, i in 1:nsd
        eq = ID[i, A]; eq != 0 && (Ub[A, i] = U[eq])
    end

    mat = LinearElastic(E, nu, :three_d)
    D   = elastic_constants(mat, nsd)

    for pc in 1:npc
        pv   = p_mat[pc, :]
        nv   = n_mat[pc, :]
        kv   = KV[pc]
        Ppc  = P[pc]
        ien  = IEN[pc]
        inc  = INC[pc]
        nen  = nen_vec[pc]

        s1 = _kv_sample(collect(kv[1]), n_vis)
        s2 = _kv_sample(collect(kv[2]), n_vis)
        s3 = _kv_sample(collect(kv[3]), n_vis)
        nd1, nd2, nd3 = length(s1), length(s2), length(s3)
        n_pts = nd1 * nd2 * nd3

        pts  = zeros(3, n_pts)
        disp = zeros(3, n_pts)
        sxx  = zeros(n_pts); syy = zeros(n_pts); szz = zeros(n_pts)
        sxy  = zeros(n_pts); syz = zeros(n_pts); szx = zeros(n_pts)
        svm  = zeros(n_pts)
        s_rr     = zeros(n_pts)
        s_rr_ex  = zeros(n_pts)

        n_elem = [nv[d] - pv[d] for d in 1:npd]

        idx = 0
        for xi3 in s3, xi2 in s2, xi1 in s1
            idx += 1
            Xi = [xi1, xi2, xi3]

            n0 = [find_span(nv[d]-1, pv[d], Float64(Xi[d]), collect(kv[d])) for d in 1:npd]
            e = [n0[d] - pv[d] for d in 1:npd]
            el = (e[3]-1)*n_elem[1]*n_elem[2] + (e[2]-1)*n_elem[1] + e[1]

            xi_tilde = zeros(npd)
            for d in 1:npd
                kv_d = collect(kv[d])
                a, b = kv_d[n0[d]], kv_d[n0[d]+1]
                xi_tilde[d] = (b > a) ? clamp((2*Xi[d] - a - b) / (b - a), -1.0, 1.0) : 0.0
            end
            # Pull slightly away from ±1 to avoid degenerate-pole Jacobian singularity.
            # ξ direction (dir 1): pole at ξ=1 where x=y=0; safe to perturb for viz.
            for d in 1:npd
                xi_tilde[d] = clamp(xi_tilde[d], -1.0 + 1e-10, 1.0 - 1e-10)
            end

            local R, dR_dx, detJ
            el_nodes = Ppc[ien[el, :]]
            R, dR_dx, _, detJ, _ = shape_function(
                pv, nv, kv, B, Ppc, xi_tilde,
                nen, nsd, npd, el, n0, ien, inc
            )

            Xe = B[el_nodes, 1:nsd]
            X  = Xe' * R
            pts[1:nsd, idx] = X

            Ue = Ub[el_nodes, :]
            u  = Ue' * R
            disp[1:nsd, idx] = u

            if detJ > 0.0
                dN_dX = Matrix(dR_dx')
                B_mat = strain_displacement_matrix(nsd, nen, dN_dX)
                u_vec = Vector{Float64}(undef, nsd * nen)
                for a in 1:nen, d in 1:nsd; u_vec[(a-1)*nsd + d] = Ue[a, d]; end
                sig = D * (B_mat * u_vec)  # [σxx, σyy, σzz, τxy, τyz, τzx]
                sxx[idx] = sig[1]; syy[idx] = sig[2]; szz[idx] = sig[3]
                sxy[idx] = sig[4]; syz[idx] = sig[5]; szx[idx] = sig[6]
                svm[idx] = sqrt(max(0.0,
                    0.5*((sig[1]-sig[2])^2 + (sig[2]-sig[3])^2 + (sig[3]-sig[1])^2)
                    + 3*(sig[4]^2 + sig[5]^2 + sig[6]^2)))

                # Radial stress: σ_rr = r̂ᵀ σ r̂
                r = norm(X)
                if r > 1e-14
                    rhat = X / r
                    sig_mat = [sig[1] sig[4] sig[6];
                               sig[4] sig[2] sig[5];
                               sig[6] sig[5] sig[3]]
                    s_rr[idx] = dot(rhat, sig_mat * rhat)
                    # Exact Lamé
                    sig_ex = lame_stress_sphere(X[1], X[2], X[3];
                                               p_i=p_i, r_i=r_i, r_o=r_o)
                    rhat3 = [X[1], X[2], X[3]] / r
                    s_rr_ex[idx] = dot(rhat3, sig_ex * rhat3)
                end
            end
        end

        fname = "$(prefix)_$(pc).vtk"
        open(fname, "w") do f
            println(f, "# vtk DataFile Version 2.0")
            println(f, "Pressurized sphere patch $pc")
            println(f, "ASCII")
            println(f, "DATASET STRUCTURED_GRID")
            println(f, "DIMENSIONS $nd1 $nd2 $nd3")
            println(f, "POINTS $n_pts float")
            for i in 1:n_pts
                @printf f "%e\t%e\t%e\n" pts[1,i] pts[2,i] pts[3,i]
            end
            println(f, "POINT_DATA $n_pts")
            println(f, "VECTORS displacement float")
            for i in 1:n_pts
                @printf f "%e\t%e\t%e\n" disp[1,i] disp[2,i] disp[3,i]
            end
            for (name, arr) in [("stress_xx", sxx), ("stress_yy", syy), ("stress_zz", szz),
                                 ("stress_xy", sxy), ("stress_yz", syz), ("stress_zx", szx),
                                 ("von_mises", svm), ("stress_rr", s_rr), ("stress_rr_exact", s_rr_ex)]
                println(f, "SCALARS $name float 1")
                println(f, "LOOKUP_TABLE default")
                for i in 1:n_pts; @printf f "%e\n" arr[i]; end
            end
        end
        @printf "  Wrote %s  (%d×%d×%d grid)\n" fname nd1 nd2 nd3
    end

    # ── Control points and control net as VTK ──────────────────────────────
    for pc in 1:npc
        Ppc = P[pc]
        nv  = n_mat[pc, :]  # CPs per direction
        n1, n2, n3 = nv[1], nv[2], nv[3]
        n_cp = length(Ppc)

        # Write control points as STRUCTURED_GRID (ParaView shows the mesh/net)
        fname_cp = "$(prefix)_cp_$(pc).vtk"
        open(fname_cp, "w") do f
            println(f, "# vtk DataFile Version 2.0")
            println(f, "Control points patch $pc")
            println(f, "ASCII")
            println(f, "DATASET STRUCTURED_GRID")
            println(f, "DIMENSIONS $n1 $n2 $n3")
            println(f, "POINTS $n_cp float")
            # CP ordering: ξ fastest, η middle, ζ slowest — matches IGAros convention
            for k in 1:n3, j in 1:n2, i in 1:n1
                lin = (k-1)*n1*n2 + (j-1)*n1 + i
                A   = Ppc[lin]
                @printf f "%e\t%e\t%e\n" B[A,1] B[A,2] B[A,3]
            end
            # Deformed positions and weights as point data
            println(f, "POINT_DATA $n_cp")
            println(f, "SCALARS weight float 1")
            println(f, "LOOKUP_TABLE default")
            for k in 1:n3, j in 1:n2, i in 1:n1
                lin = (k-1)*n1*n2 + (j-1)*n1 + i
                A   = Ppc[lin]
                @printf f "%e\n" B[A, end]
            end
            println(f, "VECTORS cp_displacement float")
            for k in 1:n3, j in 1:n2, i in 1:n1
                lin = (k-1)*n1*n2 + (j-1)*n1 + i
                A   = Ppc[lin]
                @printf f "%e\t%e\t%e\n" Ub[A,1] Ub[A,2] Ub[A,3]
            end
        end
        @printf "  Wrote %s  (%d×%d×%d control net)\n" fname_cp n1 n2 n3
    end
end

# ─────────────────────── p=1 helpers ─────────────────────────────────────────

# Open uniform B-spline knot vector for n_elem elements of degree p.
# Defined here so pressurized_sphere.jl can be included standalone; if
# concentric_cylinders.jl was already included, this definition is shadowed.
if !@isdefined(open_uniform_kv)
    function open_uniform_kv(n_elem::Int, p::Int)::Vector{Float64}
        n_cp = n_elem + p
        kv   = zeros(n_cp + p + 1)
        kv[end-p:end] .= 1.0
        for i in 1:n_elem-1; kv[p+1+i] = i / n_elem; end
        return kv
    end
end

# ─────────────────────── p=1 direct-mesh geometry ────────────────────────────

"""
    sphere_geometry_direct_p1(n_ang_i, n_ang_o, n_rad; r_i, r_c, r_o)
        -> (B, P)

Build the two-patch **trilinear** (p=1) geometry for the solid sphere octant by
placing control points **directly on the sphere surface** at each refinement level
(not via knot insertion from a coarser patch).

Each CP at grid index (i, j, k) (1-based, ξ×η×ζ) maps to
    θ = (i-1)/n_ang · (π/2),   φ = (j-1)/n_ang · (π/2),   r = r_a + (k-1)/n_rad · (r_b - r_a)
    x = r sin(θ) cos(φ),  y = r sin(θ) sin(φ),  z = r cos(θ)

CP ordering: ξ inner (fastest), η middle, ζ outer — consistent with IGAros convention.
All weights = 1 (non-rational trilinear).

Facet correspondence (same as higher-p sphere):
  Face 1 (ζ=1):   inner radial surface (r=r_a)
  Face 6 (ζ=n₃):  outer radial surface (r=r_b)
  Face 4 (ξ=1):   z-axis pole (θ=0,  x=y=0)
  Face 2 (ξ=n₁):  equatorial plane (θ=π/2, z=0)
  Face 5 (η=1):   xz-plane (φ=0, y=0)
  Face 3 (η=n₂):  yz-plane (φ=π/2, x=0)
"""
function sphere_geometry_direct_p1(
    n_ang_i::Int, n_ang_o::Int, n_rad::Int;
    r_i::Float64 = 1.0,
    r_c::Float64 = 1.2,
    r_o::Float64 = 1.4
)::Tuple{Matrix{Float64}, Vector{Vector{Int}}}

    function build_patch(n_ang::Int, r_a::Float64, r_b::Float64)
        n1 = n_ang + 1;  n2 = n_ang + 1;  n3 = n_rad + 1
        B  = zeros(n1 * n2 * n3, 4)   # [x, y, z, w]
        for k in 1:n3                                    # ζ: radial (slowest)
            r = r_a + (k - 1) / n_rad * (r_b - r_a)
            for j in 1:n2                                # η: φ angle (middle)
                φ = (j - 1) / n_ang * (π / 2)
                for i in 1:n1                            # ξ: θ angle (fastest)
                    θ = (i - 1) / n_ang * (π / 2)
                    A = (k - 1)*n1*n2 + (j - 1)*n1 + i
                    B[A, 1] = r * sin(θ) * cos(φ)
                    B[A, 2] = r * sin(θ) * sin(φ)
                    B[A, 3] = r * cos(θ)
                    B[A, 4] = 1.0                        # unit weight
                end
            end
        end
        return B
    end

    B1   = build_patch(n_ang_i, r_i, r_c)
    B2   = build_patch(n_ang_o, r_c, r_o)
    ncp1 = size(B1, 1);  ncp2 = size(B2, 1)
    return vcat(B1, B2), [collect(1:ncp1), collect(ncp1+1:ncp1+ncp2)]
end

# ─────────────────────── p=1 single-level solve ───────────────────────────────

"""
    solve_sphere_p1(exp_level; conforming, epss, ...) -> (err_rel, err_abs)

Pressurized-sphere benchmark with **p=1 trilinear elements**.
CPs are placed directly on the sphere surface so geometry error is O(h²),
faster than the O(h¹) discretisation error.

Mesh sizes at refinement level `exp_level`:
  n_ang_o = 2^(exp_level+1)  (outer patch angular elements)
  n_ang_i = mesh_ratio·n_ang_o (inner patch, non-conforming)
  n_rad   = 2^(exp_level+1)  (both patches, radial elements)
"""
function solve_sphere_p1(
    exp_level::Int;
    conforming::Bool              = false,
    mesh_ratio::Float64           = 2.0,
    r_i::Float64                  = 1.0,
    r_c::Float64                  = 1.2,
    r_o::Float64                  = 1.4,
    E::Float64                    = 1.0,
    nu::Float64                   = 0.3,
    p_i::Float64                  = 0.01,
    epss::Float64                 = 0.0,
    NQUAD::Int                    = 2,
    NQUAD_mortar::Int             = 3,
    strategy::IntegrationStrategy  = ElementBasedIntegration(),
    formulation::FormulationStrategy = TwinMortarFormulation(),
    vtk_prefix::String             = "",
    n_vis::Int                     = 4
)

    nsd = 3; npd = 3; ned = 3; npc = 2
    p_ord = 1

    n_ang_o = 2^(exp_level + 2)       # 4 at exp=0 (matches P&S §10.3 baseline)
    n_ang_i = conforming ? n_ang_o : round(Int, mesh_ratio * n_ang_o)
    n_rad   = 2^exp_level             # 1 at exp=0 (1 element through thickness per patch)

    # ── Geometry (CPs directly on sphere surface) ─────────────────────────────
    B_ref, P_ref = sphere_geometry_direct_p1(n_ang_i, n_ang_o, n_rad;
                                              r_i=r_i, r_c=r_c, r_o=r_o)
    ncp = size(B_ref, 1)

    p_mat     = fill(p_ord, npc, npd)
    n_mat_ref = [n_ang_i+1  n_ang_i+1  n_rad+1;
                 n_ang_o+1  n_ang_o+1  n_rad+1]

    # p=1 open uniform knot vectors
    kv_i   = open_uniform_kv(n_ang_i, 1)
    kv_o   = open_uniform_kv(n_ang_o, 1)
    kv_rad = open_uniform_kv(n_rad,   1)
    KV_ref = Vector{Vector{Vector{Float64}}}([
        [kv_i, kv_i, kv_rad],
        [kv_o, kv_o, kv_rad]
    ])

    epss_use = epss > 0.0 ? epss : 100.0

    # ── Connectivity ──────────────────────────────────────────────────────────
    nel, nnp, nen = patch_metrics(npc, npd, p_mat, n_mat_ref)
    IEN = build_ien(nsd, npd, npc, p_mat, n_mat_ref, nel, nnp, nen)
    INC = [build_inc(n_mat_ref[pc, :]) for pc in 1:npc]

    # ── Dirichlet BCs (same symmetry planes as higher-p sphere) ──────────────
    dBC = [1 3 2 1 2;
           1 4 2 1 2;
           2 4 2 1 2;
           2 5 2 1 2;
           3 2 2 1 2]
    bc_per_dof = dirichlet_bc_control_points(p_mat, n_mat_ref, KV_ref, P_ref,
                                              npd, nnp, ned, dBC)
    neq, ID = build_id(bc_per_dof, ned, ncp)
    LM      = build_lm(nen, ned, npc, nel, ID, IEN, P_ref)

    # ── Stiffness matrix ──────────────────────────────────────────────────────
    mats = [LinearElastic(E, nu, :three_d), LinearElastic(E, nu, :three_d)]
    Ub0  = zeros(ncp, nsd)
    K, _ = build_stiffness_matrix(npc, nsd, npd, ned, neq,
                                   p_mat, n_mat_ref, KV_ref, P_ref, B_ref, Ub0,
                                   nen, nel, IEN, INC, LM, mats, NQUAD, 1.0)

    # ── Load: internal pressure on Patch 1 face 1 (ζ=1, r=r_i) ──────────────
    stress_fn = (x, y, z) -> lame_stress_sphere(x, y, z; p_i=p_i, r_i=r_i, r_o=r_o)
    F = zeros(neq)
    F = segment_load(n_mat_ref[1,:], p_mat[1,:], KV_ref[1], P_ref[1], B_ref,
                     nnp[1], nen[1], nsd, npd, ned,
                     Int[], 1, ID, F, stress_fn, 1.0, NQUAD)

    K_bc, F_bc = enforce_dirichlet(Tuple{Int,Float64}[], K, F)

    # ── Mortar coupling at r = r_c ──────────────────────────────────────────
    pairs = formulation isa SinglePassFormulation ?
        [InterfacePair(1, 6, 2, 1)] :
        [InterfacePair(1, 6, 2, 1), InterfacePair(2, 1, 1, 6)]
    Pc    = build_interface_cps(pairs, p_mat, n_mat_ref, KV_ref, P_ref, npd, nnp, formulation)
    C, Z  = build_mortar_coupling(Pc, pairs, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
                                   ID, nnp, ned, nsd, npd, neq, NQUAD_mortar, epss_use,
                                   strategy, formulation)

    # ── Solve ─────────────────────────────────────────────────────────────────
    U, _ = solve_mortar(K_bc, C, Z, F_bc)

    # ── L2 stress error ───────────────────────────────────────────────────────
    σ_abs, σ_ref = l2_stress_error_sphere(
        U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
        nen, nel, IEN, INC, mats, NQUAD, stress_fn
    )

    # ── L2 displacement error ─────────────────────────────────────────────────
    disp_fn = (x, y, z) -> lame_displacement_sphere(x, y, z;
        p_i=p_i, r_i=r_i, r_o=r_o, E=E, nu=nu)
    l2_abs, l2_ref = l2_disp_error_sphere(
        U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
        nen, nel, IEN, INC, NQUAD, disp_fn
    )

    # ── Energy-norm error ─────────────────────────────────────────────────────
    en_abs, en_ref = energy_error_sphere(
        U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
        nen, nel, IEN, INC, mats, NQUAD, stress_fn
    )

    # ── Optional VTK export ────────────────────────────────────────────────
    if !isempty(vtk_prefix)
        write_vtk_sphere(vtk_prefix, U, ID, npc, nsd, npd,
                          p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
                          nen, IEN, INC, E, nu;
                          n_vis=n_vis, p_i=p_i, r_i=r_i, r_o=r_o)
    end

    return (σ_rel=σ_abs/σ_ref, σ_abs=σ_abs,
            l2_rel=l2_abs/l2_ref, l2_abs=l2_abs,
            en_rel=en_abs/en_ref, en_abs=en_abs)
end

# ─────────────────────── Setup helpers for cost study ─────────────────────────

"""
    _sphere_setup(p_ord, exp_level; kwargs...) -> NamedTuple

Extract all arguments needed for `build_mortar_coupling` from the pressurized sphere
benchmark at polynomial degree `p_ord` (≥2) and refinement level `exp_level`, without
performing the mortar assembly or linear solve.  Used by `run_cost_study_sphere`.
"""
function _sphere_setup(
    p_ord::Int, exp_level::Int;
    r_i::Float64 = 1.0, r_c::Float64 = 1.2, r_o::Float64 = 1.4,
    E::Float64 = 1.0, nu::Float64 = 0.3, p_i::Float64 = 0.01,
    mesh_ratio::Float64 = 2.0,
    epss::Float64 = 0.0, NQUAD::Int = p_ord + 1, NQUAD_mortar::Int = p_ord + 2,
)
    nsd = 3;  npd = 3;  ned = 3;  npc = 2

    B0, P  = sphere_geometry(p_ord; r_i=r_i, r_c=r_c, r_o=r_o)
    p_mat  = fill(p_ord, npc, npd)
    n_mat  = fill(p_ord + 1, npc, npd)
    KV     = generate_knot_vectors(npc, npd, p_mat, n_mat)

    n_ang = 2^exp_level;  n_rad = 2^exp_level
    n_ang_inner = round(Int, mesh_ratio * n_ang)
    u_ang_o = Float64[i/n_ang       for i in 1:n_ang-1]
    u_ang_i = Float64[i/n_ang_inner for i in 1:n_ang_inner-1]
    u_rad   = Float64[i/n_rad       for i in 1:n_rad-1]
    kref_data = Vector{Float64}[
        vcat([1.0, 1.0], u_ang_i), vcat([1.0, 2.0], u_ang_i), vcat([1.0, 3.0], u_rad),
        vcat([2.0, 1.0], u_ang_o), vcat([2.0, 2.0], u_ang_o), vcat([2.0, 3.0], u_rad),
    ]
    n_mat_ref, _, KV_ref, B_ref, P_ref = krefinement(
        nsd, npd, npc, n_mat, p_mat, KV, B0, P, kref_data)
    ncp = size(B_ref, 1)
    epss_use = epss > 0.0 ? epss : 100.0

    nel, nnp, nen = patch_metrics(npc, npd, p_mat, n_mat_ref)
    IEN = build_ien(nsd, npd, npc, p_mat, n_mat_ref, nel, nnp, nen)

    dBC = [1 3 2 1 2; 1 4 2 1 2; 2 4 2 1 2; 2 5 2 1 2; 3 2 2 1 2]
    bc_per_dof = dirichlet_bc_control_points(p_mat, n_mat_ref, KV_ref, P_ref,
                                              npd, nnp, ned, dBC)
    neq, ID = build_id(bc_per_dof, ned, ncp)

    pairs_tm = [InterfacePair(1, 6, 2, 1), InterfacePair(2, 1, 1, 6)]
    pairs_sp = [InterfacePair(1, 6, 2, 1)]

    # Interface element count: facet 6 of Patch 1 (ζ=n₃) → n_elem_ξ × n_elem_η
    n_iface_slave = (n_mat_ref[1, 1] - p_ord) * (n_mat_ref[1, 2] - p_ord)

    return (
        p_mat=p_mat, n_mat_ref=n_mat_ref, KV_ref=KV_ref, P_ref=P_ref, B_ref=B_ref,
        ID=ID, nnp=nnp, ned=ned, nsd=nsd, npd=npd, neq=neq, ncp=ncp,
        NQUAD_mortar=NQUAD_mortar, epss=epss_use,
        pairs_tm=pairs_tm, pairs_sp=pairs_sp,
        n_iface_slave=n_iface_slave,
    )
end

"""
    _sphere_setup_p1(exp_level; kwargs...) -> NamedTuple

p=1 variant of `_sphere_setup` using direct CP placement on the sphere surface.
"""
function _sphere_setup_p1(
    exp_level::Int;
    r_i::Float64 = 1.0, r_c::Float64 = 1.2, r_o::Float64 = 1.4,
    E::Float64 = 1.0, nu::Float64 = 0.3,
    mesh_ratio::Float64 = 2.0,
    epss::Float64 = 0.0, NQUAD_mortar::Int = 3,
)
    nsd = 3;  npd = 3;  ned = 3;  npc = 2;  p_ord = 1

    n_ang_o = 2^(exp_level + 1)
    n_ang_i = round(Int, mesh_ratio * n_ang_o)
    n_rad   = 2^(exp_level + 1)

    B_ref, P_ref = sphere_geometry_direct_p1(n_ang_i, n_ang_o, n_rad;
                                              r_i=r_i, r_c=r_c, r_o=r_o)
    ncp   = size(B_ref, 1)
    p_mat = fill(p_ord, npc, npd)
    n_mat_ref = [n_ang_i+1  n_ang_i+1  n_rad+1;
                 n_ang_o+1  n_ang_o+1  n_rad+1]

    kv_i   = open_uniform_kv(n_ang_i, 1)
    kv_o   = open_uniform_kv(n_ang_o, 1)
    kv_rad = open_uniform_kv(n_rad,   1)
    KV_ref = Vector{Vector{Vector{Float64}}}([
        [kv_i, kv_i, kv_rad], [kv_o, kv_o, kv_rad]
    ])

    epss_use = epss > 0.0 ? epss : 100.0

    nel, nnp, nen = patch_metrics(npc, npd, p_mat, n_mat_ref)
    IEN = build_ien(nsd, npd, npc, p_mat, n_mat_ref, nel, nnp, nen)

    dBC = [1 3 2 1 2; 1 4 2 1 2; 2 4 2 1 2; 2 5 2 1 2; 3 2 2 1 2]
    bc_per_dof = dirichlet_bc_control_points(p_mat, n_mat_ref, KV_ref, P_ref,
                                              npd, nnp, ned, dBC)
    neq, ID = build_id(bc_per_dof, ned, ncp)

    pairs_tm = [InterfacePair(1, 6, 2, 1), InterfacePair(2, 1, 1, 6)]
    pairs_sp = [InterfacePair(1, 6, 2, 1)]

    n_iface_slave = n_ang_i * n_ang_i   # p=1: n_elem = n_cp - 1 = n_ang

    return (
        p_mat=p_mat, n_mat_ref=n_mat_ref, KV_ref=KV_ref, P_ref=P_ref, B_ref=B_ref,
        ID=ID, nnp=nnp, ned=ned, nsd=nsd, npd=npd, neq=neq, ncp=ncp,
        NQUAD_mortar=NQUAD_mortar, epss=epss_use,
        pairs_tm=pairs_tm, pairs_sp=pairs_sp,
        n_iface_slave=n_iface_slave,
    )
end

# ─────────────────────── Cost study ───────────────────────────────────────────

"""
    run_cost_study_sphere(; degrees, exp_levels, n_repeats, epss)

Time `build_mortar_coupling` for TM (ElementBased), SPMS (SinglePass+SegmentBased),
and DPM (DualPass+SegmentBased) on the 3D pressurized sphere benchmark.
Prints a table suitable for §7.6 of the paper.
"""
function run_cost_study_sphere(;
    degrees::Vector{Int}    = [1, 2],
    exp_levels::Vector{Int} = [1, 2],
    n_repeats::Int          = 5,
    epss::Float64           = 1e6,
    kwargs...
)
    configs = [
        ("TM",   TwinMortarFormulation(),  ElementBasedIntegration()),
        ("SPMS", SinglePassFormulation(),  SegmentBasedIntegration()),
        ("DPM",  DualPassFormulation(),    SegmentBasedIntegration()),
    ]

    hdr = @sprintf("%-4s  %-3s  %-10s  %-9s  %-7s  %-10s",
                   "p", "exp", "n_iface_s", "method", "t(ms)", "t/t_TM")
    println("\n=== Interface assembly cost study (3D pressurized sphere) ===")
    println(hdr)
    println("-"^length(hdr))

    for p_ord in degrees
        NQUAD_mortar = p_ord + 2
        for exp in exp_levels
            d = if p_ord == 1
                _sphere_setup_p1(exp; epss=epss, NQUAD_mortar=NQUAD_mortar, kwargs...)
            else
                _sphere_setup(p_ord, exp; epss=epss, NQUAD_mortar=NQUAD_mortar, kwargs...)
            end

            # Reference TM time (JIT warm-up + timing)
            Pc_ref = build_interface_cps(d.pairs_tm, d.p_mat, d.n_mat_ref,
                                          d.KV_ref, d.P_ref, d.npd, d.nnp,
                                          TwinMortarFormulation())
            build_mortar_coupling(Pc_ref, d.pairs_tm, d.p_mat, d.n_mat_ref,
                d.KV_ref, d.P_ref, d.B_ref, d.ID, d.nnp, d.ned, d.nsd, d.npd, d.neq,
                NQUAD_mortar, d.epss, ElementBasedIntegration(), TwinMortarFormulation())
            t_tm = minimum(
                @elapsed(build_mortar_coupling(Pc_ref, d.pairs_tm, d.p_mat, d.n_mat_ref,
                    d.KV_ref, d.P_ref, d.B_ref, d.ID, d.nnp, d.ned, d.nsd, d.npd, d.neq,
                    NQUAD_mortar, d.epss, ElementBasedIntegration(), TwinMortarFormulation()))
                for _ in 1:n_repeats)

            for (label, form, strat) in configs
                pairs = form isa TwinMortarFormulation ? d.pairs_tm : d.pairs_sp
                Pc = build_interface_cps(pairs, d.p_mat, d.n_mat_ref,
                                          d.KV_ref, d.P_ref, d.npd, d.nnp, form)

                # JIT warm-up
                build_mortar_coupling(Pc, pairs, d.p_mat, d.n_mat_ref, d.KV_ref,
                    d.P_ref, d.B_ref, d.ID, d.nnp, d.ned, d.nsd, d.npd, d.neq,
                    NQUAD_mortar, d.epss, strat, form)

                # Timed runs
                t_min = minimum(
                    @elapsed(build_mortar_coupling(Pc, pairs, d.p_mat, d.n_mat_ref,
                        d.KV_ref, d.P_ref, d.B_ref, d.ID, d.nnp, d.ned, d.nsd, d.npd, d.neq,
                        NQUAD_mortar, d.epss, strat, form))
                    for _ in 1:n_repeats)

                ratio = label == "TM" ? 1.0 : t_min / t_tm
                @printf("%-4d  %-3d  %-10d  %-9s  %-7.2f  %-10.2f\n",
                        p_ord, exp, d.n_iface_slave, label, t_min*1000, ratio)
            end
            println()
        end
    end
end

# ─────────────────────── p=1 convergence table ────────────────────────────────

"""
    run_convergence_sphere_p1(; exp_range, kwargs...) -> nothing

Print a convergence table for the p=1 pressurized sphere benchmark.
Reports L2-displacement, energy-norm, and L2-stress errors with rates.
"""
function run_convergence_sphere_p1(;
    exp_range::UnitRange{Int} = 0:3,
    kwargs...
)
    @printf("\n─── p = 1  (direct mesh) ──────────────────────────────────────────────────────────────\n")
    @printf("  %5s  %12s  %6s  %12s  %6s  %12s  %6s  %8s\n",
            "exp", "||e||_disp", "rate", "||e||_energy", "rate", "||e||_σ", "rate", "h")
    prev_d = prev_e = prev_s = NaN
    for e in exp_range
        h = 1.0 / 2^(e + 1)
        try
            r = solve_sphere_p1(e; kwargs...)
            rd = isnan(prev_d) ? NaN : log(prev_d/r.l2_abs) / log(2.0)
            re = isnan(prev_e) ? NaN : log(prev_e/r.en_abs) / log(2.0)
            rs = isnan(prev_s) ? NaN : log(prev_s/r.σ_abs)  / log(2.0)
            @printf("  %5d  %12.4e  %6.2f  %12.4e  %6.2f  %12.4e  %6.2f  %8.4f\n",
                    e, r.l2_abs, rd, r.en_abs, re, r.σ_abs, rs, h)
            prev_d = r.l2_abs; prev_e = r.en_abs; prev_s = r.σ_abs
        catch ex
            @printf("  %5d  ERROR: %s\n", e, string(ex)[1:min(60,end)])
            prev_d = prev_e = prev_s = NaN
        end
    end
end

# ─────────────────────── Convergence table ────────────────────────────────────

"""
    run_convergence_sphere(; degrees, exp_range, kwargs...) -> nothing

Print a convergence table for the pressurized sphere benchmark.
Reports L2-displacement, energy-norm, and L2-stress errors with rates.
"""
function run_convergence_sphere(;
    degrees::Vector{Int} = [2, 3, 4],
    exp_range::UnitRange{Int} = 0:3,
    kwargs...
)
    for p in degrees
        @printf("\n─── p = %d ─────────────────────────────────────────────────────────────────────────────\n", p)
        @printf("  %5s  %12s  %6s  %12s  %6s  %12s  %6s  %8s\n",
                "exp", "||e||_disp", "rate", "||e||_energy", "rate", "||e||_σ", "rate", "h")
        prev_d = prev_e = prev_s = NaN
        for e in exp_range
            h = 0.5^e
            try
                r = solve_sphere(p, e; kwargs...)
                rd = isnan(prev_d) ? NaN : log(prev_d/r.l2_abs) / log(2.0)
                re = isnan(prev_e) ? NaN : log(prev_e/r.en_abs) / log(2.0)
                rs = isnan(prev_s) ? NaN : log(prev_s/r.σ_abs)  / log(2.0)
                @printf("  %5d  %12.4e  %6.2f  %12.4e  %6.2f  %12.4e  %6.2f  %8.4f\n",
                        e, r.l2_abs, rd, r.en_abs, re, r.σ_abs, rs, h)
                prev_d = r.l2_abs; prev_e = r.en_abs; prev_s = r.σ_abs
            catch ex
                @printf("  %5d  ERROR: %s\n", e, string(ex)[1:min(60,end)])
                prev_d = prev_e = prev_s = NaN
            end
        end
    end
end

# ─────────────────────── ε-sensitivity study ──────────────────────────────────

"""
    run_eps_sweep_sphere(; degrees, exp_range, epss_range, mesh_ratios, kwargs...)

Sweep stabilization parameter ε for a range of mesh refinements and mesh ratios.
Matches the format of Puso & Solberg (2020) §10.3 Fig. 13.
Reports energy-norm and L2-displacement relative errors.
"""
function run_eps_sweep_sphere(;
    degrees::Vector{Int}       = [2, 3],
    exp_range::UnitRange{Int}  = 0:3,
    epss_range::Vector{Float64} = [0.01, 0.1, 1.0, 10.0, 100.0],
    mesh_ratios::Vector{Float64} = [2.0, 3.5],
    kwargs...
)
    for mr in mesh_ratios
        @printf("\n\n═══ mesh ratio = %.1f:1 ═══════════════════════════════════════════\n", mr)
        for p in degrees
            @printf("\n─── p = %d ─────────────────────────────────\n", p)
            @printf("  %5s", "exp")
            for eps in epss_range
                @printf("  ε=%-8.2g", eps)
            end
            println()

            # Energy norm
            @printf("  %5s", "")
            for _ in epss_range; @printf("  %-10s", "en_rel"); end
            println()

            for e in exp_range
                @printf("  %5d", e)
                for eps in epss_range
                    try
                        r = solve_sphere(p, e; epss=eps, mesh_ratio=mr, kwargs...)
                        @printf("  %10.3e", r.en_rel)
                    catch ex
                        @printf("  %10s", "ERR")
                    end
                end
                println()
            end

            # Displacement norm
            println()
            @printf("  %5s", "")
            for _ in epss_range; @printf("  %-10s", "l2_rel"); end
            println()

            for e in exp_range
                @printf("  %5d", e)
                for eps in epss_range
                    try
                        r = solve_sphere(p, e; epss=eps, mesh_ratio=mr, kwargs...)
                        @printf("  %10.3e", r.l2_rel)
                    catch ex
                        @printf("  %10s", "ERR")
                    end
                end
                println()
            end
        end
    end
end
