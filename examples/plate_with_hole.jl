# examples/plate_with_hole.jl
#
# Plate with circular hole — Twin Mortar convergence study (non-conforming IGA).
# Two-patch quarter-domain decomposition along the 135° diagonal.
# Exact Kirsch solution used for traction loading and L2 stress error.
#
# Ported from MATLAB: plate_with_hole_elastic_two_patches.m

import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using IGAros
using LinearAlgebra, SparseArrays, Printf

# ─────────────────────── Kirsch analytical solution ──────────────────────────

"""
    kirsch_stress(x, y; Tx, R) -> Matrix{Float64}(2×2)

Kirsch solution for a circular hole of radius R in an infinite plate under
uniaxial far-field tension σ_xx = Tx (plane stress).
Returns the Cauchy stress tensor in Cartesian form.
"""
function kirsch_stress(x::Real, y::Real; Tx::Float64, R::Float64)::Matrix{Float64}
    r  = sqrt(x^2 + y^2)
    θ  = atan(y, x)
    ρ2 = (R / r)^2;  ρ4 = ρ2^2
    c2, s2 = cos(2θ), sin(2θ)

    σ_rr =  0.5Tx*(1 - ρ2) + 0.5Tx*(1 - 4ρ2 + 3ρ4)*c2
    σ_θθ =  0.5Tx*(1 + ρ2) - 0.5Tx*(1 + 3ρ4)*c2
    σ_rθ = -0.5Tx*(1 + 2ρ2 - 3ρ4)*s2

    c, s = cos(θ), sin(θ)
    T = [c s; -s c]                          # rotation from Cartesian to polar
    return T' * [σ_rr σ_rθ; σ_rθ σ_θθ] * T  # σ_cart = T^T σ_polar T
end

"""
    kirsch_displacement(x, y; Tx, R, E, nu) -> (ux, uy)

Kirsch displacement solution for a circular hole of radius R in an infinite
plate under uniaxial far-field tension σ_xx = Tx (plane stress).

Derived by integrating the Kirsch stress field via Hooke's law (plane stress):
  u_r = Tx/(2E) { (1-ν)r + (1+ν)R²/r
                 + [(1+ν)r + 4R²/r - (1+ν)R⁴/r³] cos2θ }
  u_θ = -Tx/(2E) { (1+ν)r + 2(1-ν)R²/r + (1+ν)R⁴/r³ } sin2θ
"""
function kirsch_displacement(x::Real, y::Real;
                             Tx::Float64, R::Float64,
                             E::Float64,  nu::Float64)::Tuple{Float64,Float64}
    r  = sqrt(x^2 + y^2)
    θ  = atan(y, x)
    c2, s2 = cos(2θ), sin(2θ)
    a2 = R^2;  a4 = R^4

    C = Tx / (2E)

    u_r =  C * ( (1 - nu)*r + (1 + nu)*a2/r
               + ((1 + nu)*r + 4a2/r - (1 + nu)*a4/r^3) * c2 )
    u_θ = -C * ( (1 + nu)*r + 2(1 - nu)*a2/r + (1 + nu)*a4/r^3 ) * s2

    c, s = cos(θ), sin(θ)
    ux = u_r * c - u_θ * s
    uy = u_r * s + u_θ * c
    return ux, uy
end

# ─────────────────────── Bezier degree elevation ──────────────────────────────

"""
    bezier_elevate(Bh) -> Bh_elevated

Elevate a Bezier curve by one degree.  `Bh` is a (p+1)×ncols matrix of
homogeneous control points.  Returns (p+2)×ncols elevated points.
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
    plate_geometry(p_ord; R, a, b) -> (B, P)

Build the initial coarse two-patch NURBS mesh for a quarter plate with hole.

Domain: angular range 90°–180° in the second quadrant (x ≤ 0, y ≥ 0).
  - Patch 1: 90°–135° (upper left), slave/master facets 4 (ξ_min) / 2 (ξ_max).
  - Patch 2: 135°–180° (lower left), obtained by –90° rotation of Patch 1.

Boundaries:
  - Patch 1 facet 2 (ξ_max, x = 0): symmetry BC ux = 0.
  - Patch 2 facet 4 (ξ_min, y = 0): symmetry BC uy = 0.
  - Patch 1 facet 3 (η_max, y = b): outer boundary, Kirsch traction.
  - Patch 2 facet 3 (η_max, x = –a): outer boundary, Kirsch traction.
  - Patch 1 facet 4 / Patch 2 facet 2: shared interface (135° diagonal).

Returns B (ncp × 4, columns = [x, y, 0, w]) and P (patch-to-global CP index).
"""
function plate_geometry(p_ord::Int;
                        R::Float64 = 1.0,
                        a::Float64 = 4.0,
                        b::Float64 = 4.0)::Tuple{Matrix{Float64}, Vector{Vector{Int}}}

    # ── Hole arc (angular range 135°→90°), start from quadratic NURBS (exact circle) ──
    b1_xyw = [-R*cos(π/4)   R*sin(π/4)  1.0;     # at 135°
               -R*tan(π/8)   R           cos(π/8); # exact NURBS circle midpoint
                0.0           R           1.0]      # at  90°

    # Degree-elevate from p=2 to p_ord (homogeneous coordinates)
    Bh = copy(b1_xyw)
    Bh[:, 1:2] .*= Bh[:, 3:3]   # to homogeneous
    for _ in 3:p_ord
        Bh = bezier_elevate(Bh)
    end
    Bh[:, 1:2] ./= Bh[:, 3:3]   # back to Euclidean
    b1 = Bh                      # (p_ord+1) × 3  [x, y, w]

    # ── Outer boundary (straight line from (–a, b) to (0, b)), weight = 1 ──
    b2 = zeros(p_ord + 1, 3)
    for i in 0:p_ord
        t = i / p_ord
        b2[i+1, :] = [(1 - t) * (-a),  b,  1.0]
    end

    n_xi = p_ord + 1   # CPs in ξ direction (angular)
    n_et = p_ord + 1   # CPs in η direction (radial, single Bezier element)

    # Helper: bilinear Euclidean blend between hole arc and outer boundary
    function blend_patch(b1_p, b2_p)
        B = zeros(n_xi * n_et, 3)
        for j in 0:n_et-1
            t = j / (n_et - 1)
            for i in 1:n_xi
                B[j*n_xi + i, :] = (1 - t) .* b1_p[i, :] .+ t .* b2_p[i, :]
            end
        end
        return B
    end

    # ── Patch 1: 90°–135° ─────────────────────────────────────────────────────
    B1_xyw = blend_patch(b1, b2)   # (n_xi*n_et) × 3

    # ── Patch 2: 135°–180° via –90° rotation (x_new = –y, y_new = –x) ───────
    b1_rev = b1[end:-1:1, :]       # reverse ξ order (90°→135°)
    b2_rev = b2[end:-1:1, :]
    B2_tmp = blend_patch(b1_rev, b2_rev)

    B2_xyw = copy(B2_tmp)
    B2_xyw[:, 1] = -B2_tmp[:, 2]  # x_new = –y_old
    B2_xyw[:, 2] = -B2_tmp[:, 1]  # y_new = –x_old

    # ── Assemble global B in Julia format [x, y, z=0, w] ────────────────────
    ncp1 = n_xi * n_et
    ncp2 = n_xi * n_et
    B_out = zeros(ncp1 + ncp2, 4)
    B_out[1:ncp1,       [1,2,4]] = B1_xyw
    B_out[ncp1+1:end,   [1,2,4]] = B2_xyw

    P = [collect(1:ncp1), collect(ncp1+1:ncp1+ncp2)]
    return B_out, P
end

# ─────────────────────── L2 stress error ─────────────────────────────────────

"""
    l2_stress_error(U, ID, npc, nsd, npd, p, n, KV, P, B, nen, nel,
                    IEN, INC, materials, NQUAD, thickness; Tx, R)
        -> (err_abs, err_ref)

Compute the absolute and reference L2 stress norms over the whole domain:
  err_abs = sqrt( ∫ (σ_h - σ_exact) : (σ_h - σ_exact) dΩ )
  err_ref = sqrt( ∫  σ_exact : σ_exact dΩ )

The double contraction uses the Frobenius norm of the 2×2 stress tensor.
"""
function l2_stress_error(
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
    thickness::Float64;
    Tx::Float64,
    R::Float64
)::Tuple{Float64, Float64}

    # Reconstruct Ub (displacements at CPs)
    ncp = size(B, 1)
    Ub  = zeros(ncp, nsd)
    for A in 1:ncp, i in 1:nsd
        eq = ID[i, A]
        eq != 0 && (Ub[A, i] = U[eq])
    end
    # Use reference geometry B (not deformed Bu) for Jacobian and physical
    # coordinate evaluation — K was assembled with Ub0=zeros (reference B),
    # so we must be consistent. Using Bu here would evaluate kirsch_stress
    # at the wrong (displaced) position, creating an O(Tx/E·σ) error floor.

    err2 = 0.0
    ref2 = 0.0
    GPW  = gauss_product(NQUAD, npd)

    for pc in 1:npc
        ien = IEN[pc];  inc = INC[pc]
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

                # Numerical stress (Voigt: [σxx, σyy, σxy])
                B0    = strain_displacement_matrix(nsd, nen[pc], dR_dx')
                Ue    = vec(Ub[P[pc][ien[el,:]], 1:nsd]')
                ε_h   = B0 * Ue
                σ_h_v = D * ε_h

                # Physical coordinates of Gauss point (reference configuration)
                Xe = B[P[pc][ien[el,:]], :]
                X  = Xe' * R_s
                x_gp, y_gp = X[1], X[2]

                # Exact stress (2×2 matrix)
                σ_ex = kirsch_stress(x_gp, y_gp; Tx=Tx, R=R)

                # Numerical stress as 2×2 matrix
                σ_h_m = [σ_h_v[1] σ_h_v[3]; σ_h_v[3] σ_h_v[2]]

                # Frobenius inner product (note: shear counted once via symmetric matrix)
                diff_m = σ_h_m - σ_ex
                err2  += dot(diff_m, diff_m) * gwJ
                ref2  += dot(σ_ex,   σ_ex)   * gwJ
            end
        end
    end

    return sqrt(err2), sqrt(ref2)
end

# ─────────────────────── Energy-norm error ─────────────────────────────────────

"""
    energy_error_plate(U, ID, npc, nsd, npd, p, n, KV, P, B,
                       nen, nel, IEN, INC, materials, NQUAD, thickness;
                       Tx, R) -> (err_abs, err_ref)

Energy-norm error against the Kirsch solution:
  err_abs² = ∫_Ω (σ_h − σ_ex) : D⁻¹ : (σ_h − σ_ex) dΩ
  err_ref² = ∫_Ω σ_ex : D⁻¹ : σ_ex dΩ
where D is the plane-stress elasticity tensor (3×3 Voigt).
"""
function energy_error_plate(
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
    thickness::Float64;
    Tx::Float64,
    R::Float64
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
        Dinv = inv(D)   # 3×3 compliance matrix (plane-stress)

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
                σ_ex   = kirsch_stress(X[1], X[2]; Tx=Tx, R=R)
                σ_ex_v = [σ_ex[1,1], σ_ex[2,2], σ_ex[1,2]]

                Δσ_v = σ_h_v - σ_ex_v

                err2 += dot(Δσ_v, Dinv * Δσ_v) * gwJ
                ref2 += dot(σ_ex_v, Dinv * σ_ex_v) * gwJ
            end
        end
    end

    return sqrt(err2), sqrt(ref2)
end

# ─────────────────────── L2 displacement error ─────────────────────────────────

"""
    l2_disp_error_plate(U, ID, npc, nsd, npd, p, n, KV, P, B,
                        nen, nel, IEN, INC, NQUAD, thickness;
                        Tx, R, E, nu) -> (err_abs, err_ref)

L2 displacement error against the Kirsch analytical displacement field:
  err_abs² = ∫_Ω ||u_h − u_ex||² dΩ,   err_ref² = ∫_Ω ||u_ex||² dΩ
"""
function l2_disp_error_plate(
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
    thickness::Float64;
    Tx::Float64,
    R::Float64,
    E::Float64,
    nu::Float64
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
                u_h    = Ue_mat' * R_s                   # (nsd,)

                Xe     = B[P[pc][ien[el,:]], :]
                X      = Xe' * R_s
                ux_ex, uy_ex = kirsch_displacement(X[1], X[2];
                                                    Tx=Tx, R=R, E=E, nu=nu)
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
    solve_plate(p_ord, exp_level; conforming, Tx, R, a, b, E, nu, epss, NQUAD_mortar)
        -> (err_rel, err_abs)

Run one refinement level of the plate-with-hole convergence study.

- `exp_level`: refinement exponent (mesh size ∝ 1/2^exp_level).
- `conforming = true`: both patches get identical kRefData (matching interface).
- `conforming = false`: patch 1 gets one extra radial element (non-conforming).
"""
function solve_plate(
    p_ord::Int,
    exp_level::Int;
    conforming::Bool   = false,
    Tx::Float64        = 10.0,
    R::Float64         = 1.0,
    a::Float64         = 4.0,
    b::Float64         = 4.0,
    E::Float64         = 1e5,
    nu::Float64        = 0.3,
    epss::Float64      = 0.0,   # 0 = auto-scale: E * s2 (coarser radial step)
    NQUAD::Int         = p_ord + 1,
    NQUAD_mortar::Int  = 10,
    strategy::IntegrationStrategy    = ElementBasedIntegration(),
    formulation::FormulationStrategy = TwinMortarFormulation()
)::Tuple{Float64, Float64}

    nsd = 2;  npd = 2;  ned = 2;  npc = 2
    thickness = 1.0

    # ── Initial coarse geometry ───────────────────────────────────────────────
    B0, P = plate_geometry(p_ord; R=R, a=a, b=b)
    p_mat = fill(p_ord, npc, npd)
    n_mat = fill(p_ord + 1, npc, npd)  # one Bezier element per direction
    KV    = generate_knot_vectors(npc, npd, p_mat, n_mat)

    # ── k-refinement (h-refinement only; degree already set) ─────────────────
    # Angular (ξ, pd=1): uniform subdivision with step s1 = 1/(3·2^exp)
    # Radial  (η, pd=2): uniform subdivision with step s2 = 1/(2·2^exp)
    s1 = (1/3) / 2^exp_level
    s2 = (1/2) / 2^exp_level
    s2_nc = s2 / 2   # finer radial for patch 1 (non-conforming)

    # Adaptive epss scaling: E * s2 (coarser interface element size).
    # Large fixed epss (e.g. 1e6) over-stabilizes the KKT system at fine meshes
    # and kills convergence for p≥3. Scaling with h keeps Z a regularisation
    # perturbation rather than a dominant term.
    epss_use = epss > 0.0 ? epss : E * s2

    u_ang    = collect(s1    : s1    : 1.0 - s1/2)
    u_rad    = collect(s2    : s2    : 1.0 - s2/2)
    u_rad_nc = collect(s2_nc : s2_nc : 1.0 - s2_nc/2)

    # kref_data format: [pc, pd, knots...]
    kref_data = Vector{Float64}[
        vcat([1.0, 1.0], u_ang),
        vcat([2.0, 1.0], u_ang),
        vcat([1.0, 2.0], conforming ? u_rad : u_rad_nc),
        vcat([2.0, 2.0], u_rad),
    ]

    # Apply MATLAB-style offset hack: temporarily shift Patch 1's y-coords by
    # +1000 so that krefinement does NOT merge Patch 1 interface CPs with
    # Patch 2 interface CPs (which share the same physical position along the
    # 135° diagonal).  After refinement, restore y-coords via threshold > 100.
    B0_hack = copy(B0)
    B0_hack[P[1], 2] .+= 1000.0

    n_mat_ref, _, KV_ref, B_hack_ref, P_ref = krefinement(
        nsd, npd, npc, n_mat, p_mat, KV, B0_hack, P, kref_data
    )

    # Restore Patch 1 y-coords (all CPs with y > 100 belong to Patch 1)
    B_ref = copy(B_hack_ref)
    for i in axes(B_ref, 1)
        if B_ref[i, 2] > 100.0
            B_ref[i, 2] -= 1000.0
        end
    end

    ncp = size(B_ref, 1)

    # ── Connectivity ──────────────────────────────────────────────────────────
    nel, nnp, nen = patch_metrics(npc, npd, p_mat, n_mat_ref)
    IEN = build_ien(nsd, npd, npc, p_mat, n_mat_ref, nel, nnp, nen)
    INC = [build_inc(n_mat_ref[pc, :]) for pc in 1:npc]

    # ── Boundary conditions (homogeneous Dirichlet) ───────────────────────────
    # ux = 0 on facet 2 (ξ_max, x = 0) of patch 1
    # uy = 0 on facet 4 (ξ_min, y = 0) of patch 2
    dBC = [1 2 1 1;
           2 4 1 2]
    bc_per_dof = dirichlet_bc_control_points(p_mat, n_mat_ref, KV_ref, P_ref,
                                              npd, nnp, ned, dBC)
    neq, ID = build_id(bc_per_dof, ned, ncp)
    LM      = build_lm(nen, ned, npc, nel, ID, IEN, P_ref)

    # ── Stiffness matrix ──────────────────────────────────────────────────────
    mats = [LinearElastic(E, nu, :plane_stress), LinearElastic(E, nu, :plane_stress)]
    Ub0  = zeros(ncp, nsd)
    K, _ = build_stiffness_matrix(npc, nsd, npd, ned, neq,
                                   p_mat, n_mat_ref, KV_ref, P_ref, B_ref, Ub0,
                                   nen, nel, IEN, INC, LM, mats, NQUAD, thickness)

    # ── Load vector (Kirsch traction on outer boundaries) ────────────────────
    traction_fn = (x, y) -> kirsch_stress(x, y; Tx=Tx, R=R)

    F = zeros(neq)
    # Patch 1 facet 3 (η_max, y = b)
    F = segment_load(n_mat_ref[1,:], p_mat[1,:], KV_ref[1], P_ref[1], B_ref,
                     nnp[1], nen[1], nsd, npd, ned,
                     Int[], 3, ID, F, traction_fn, thickness, NQUAD)
    # Patch 2 facet 3 (η_max, x = –a)
    F = segment_load(n_mat_ref[2,:], p_mat[2,:], KV_ref[2], P_ref[2], B_ref,
                     nnp[2], nen[2], nsd, npd, ned,
                     Int[], 3, ID, F, traction_fn, thickness, NQUAD)

    K_bc, F_bc = enforce_dirichlet(Tuple{Int,Float64}[], K, F)

    # ── Mortar coupling ──────────────────────────────────────────────────────
    # Interface: patch 1 facet 4 (ξ_min, 135° line) ↔ patch 2 facet 2 (ξ_max)
    pairs_full = [InterfacePair(2, 2, 1, 4),    # slave=pc2(fac2), master=pc1(fac4)
                  InterfacePair(1, 4, 2, 2)]    # slave=pc1(fac4), master=pc2(fac2)
    pairs = formulation isa SinglePassFormulation ? pairs_full[1:1] : pairs_full
    Pc = build_interface_cps(pairs, p_mat, n_mat_ref, KV_ref, P_ref, npd, nnp,
                              formulation)

    C, Z = build_mortar_coupling(Pc, pairs, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
                                  ID, nnp, ned, nsd, npd, neq, NQUAD_mortar, epss_use,
                                  strategy, formulation)

    # ── Solve ─────────────────────────────────────────────────────────────────
    U, _ = solve_mortar(K_bc, C, Z, F_bc)

    # ── L2 stress error ───────────────────────────────────────────────────────
    err_abs, err_ref = l2_stress_error(
        U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
        nen, nel, IEN, INC, mats, NQUAD, thickness; Tx=Tx, R=R
    )

    return err_abs / err_ref, err_abs
end

"""
    solve_plate_full(p_ord, exp_level; kwargs...) -> NamedTuple

Like `solve_plate` but returns both L² stress and energy-norm errors.
Returns (l2_rel, l2_abs, en_rel, en_abs).
"""
function solve_plate_full(
    p_ord::Int,
    exp_level::Int;
    conforming::Bool   = false,
    Tx::Float64        = 10.0,
    R::Float64         = 1.0,
    a::Float64         = 4.0,
    b::Float64         = 4.0,
    E::Float64         = 1e5,
    nu::Float64        = 0.3,
    epss::Float64      = 0.0,
    NQUAD::Int         = p_ord + 1,
    NQUAD_mortar::Int  = 10,
    strategy::IntegrationStrategy    = ElementBasedIntegration(),
    formulation::FormulationStrategy = TwinMortarFormulation()
)
    nsd = 2;  npd = 2;  ned = 2;  npc = 2
    thickness = 1.0

    B0, P = plate_geometry(p_ord; R=R, a=a, b=b)
    p_mat = fill(p_ord, npc, npd)
    n_mat = fill(p_ord + 1, npc, npd)
    KV    = generate_knot_vectors(npc, npd, p_mat, n_mat)

    s1 = (1/3) / 2^exp_level
    s2 = (1/2) / 2^exp_level
    s2_nc = s2 / 2

    epss_use = epss > 0.0 ? epss : E * s2

    u_ang    = collect(s1    : s1    : 1.0 - s1/2)
    u_rad    = collect(s2    : s2    : 1.0 - s2/2)
    u_rad_nc = collect(s2_nc : s2_nc : 1.0 - s2_nc/2)

    kref_data = Vector{Float64}[
        vcat([1.0, 1.0], u_ang),
        vcat([2.0, 1.0], u_ang),
        vcat([1.0, 2.0], conforming ? u_rad : u_rad_nc),
        vcat([2.0, 2.0], u_rad),
    ]

    B0_hack = copy(B0)
    B0_hack[P[1], 2] .+= 1000.0

    n_mat_ref, _, KV_ref, B_hack_ref, P_ref = krefinement(
        nsd, npd, npc, n_mat, p_mat, KV, B0_hack, P, kref_data
    )

    B_ref = copy(B_hack_ref)
    for i in axes(B_ref, 1)
        if B_ref[i, 2] > 100.0
            B_ref[i, 2] -= 1000.0
        end
    end

    ncp = size(B_ref, 1)

    nel, nnp, nen = patch_metrics(npc, npd, p_mat, n_mat_ref)
    IEN = build_ien(nsd, npd, npc, p_mat, n_mat_ref, nel, nnp, nen)
    INC = [build_inc(n_mat_ref[pc, :]) for pc in 1:npc]

    dBC = [1 2 1 1;
           2 4 1 2]
    bc_per_dof = dirichlet_bc_control_points(p_mat, n_mat_ref, KV_ref, P_ref,
                                              npd, nnp, ned, dBC)
    neq, ID = build_id(bc_per_dof, ned, ncp)
    LM      = build_lm(nen, ned, npc, nel, ID, IEN, P_ref)

    mats = [LinearElastic(E, nu, :plane_stress), LinearElastic(E, nu, :plane_stress)]
    Ub0  = zeros(ncp, nsd)
    K, _ = build_stiffness_matrix(npc, nsd, npd, ned, neq,
                                   p_mat, n_mat_ref, KV_ref, P_ref, B_ref, Ub0,
                                   nen, nel, IEN, INC, LM, mats, NQUAD, thickness)

    traction_fn = (x, y) -> kirsch_stress(x, y; Tx=Tx, R=R)
    F = zeros(neq)
    F = segment_load(n_mat_ref[1,:], p_mat[1,:], KV_ref[1], P_ref[1], B_ref,
                     nnp[1], nen[1], nsd, npd, ned,
                     Int[], 3, ID, F, traction_fn, thickness, NQUAD)
    F = segment_load(n_mat_ref[2,:], p_mat[2,:], KV_ref[2], P_ref[2], B_ref,
                     nnp[2], nen[2], nsd, npd, ned,
                     Int[], 3, ID, F, traction_fn, thickness, NQUAD)

    K_bc, F_bc = enforce_dirichlet(Tuple{Int,Float64}[], K, F)

    pairs_full = [InterfacePair(2, 2, 1, 4), InterfacePair(1, 4, 2, 2)]
    pairs = formulation isa SinglePassFormulation ? pairs_full[1:1] : pairs_full
    Pc = build_interface_cps(pairs, p_mat, n_mat_ref, KV_ref, P_ref, npd, nnp,
                              formulation)

    C, Z = build_mortar_coupling(Pc, pairs, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
                                  ID, nnp, ned, nsd, npd, neq, NQUAD_mortar, epss_use,
                                  strategy, formulation)

    U, _ = solve_mortar(K_bc, C, Z, F_bc)

    l2_abs, l2_ref = l2_stress_error(
        U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
        nen, nel, IEN, INC, mats, NQUAD, thickness; Tx=Tx, R=R
    )

    en_abs, en_ref = energy_error_plate(
        U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
        nen, nel, IEN, INC, mats, NQUAD, thickness; Tx=Tx, R=R
    )

    d_abs, d_ref = l2_disp_error_plate(
        U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
        nen, nel, IEN, INC, NQUAD, thickness; Tx=Tx, R=R, E=E, nu=nu
    )

    return (l2_rel=l2_abs/l2_ref, l2_abs=l2_abs,
            en_rel=en_abs/en_ref, en_abs=en_abs,
            d_rel=d_abs/d_ref, d_abs=d_abs,
            K_bc=K_bc, C=C, Z=Z, neq=neq)
end

# ─────────────────────── Convergence study ───────────────────────────────────

function run_convergence(;
    degrees::Vector{Int}   = [2, 3, 4],
    exp_range::UnitRange   = 1:5,
    conforming::Bool       = false,
    kwargs...
)
    label = conforming ? "conforming" : "non-conforming"
    @printf("\n%-40s  %6s  %14s  %14s\n",
            "Case", "exp", "||e||_L2/||σ||", "||e||_L2")
    @printf("%s\n", "-"^80)

    for p_ord in degrees
        for exp_level in exp_range
            rel, abs_err = solve_plate(p_ord, exp_level; conforming=conforming, kwargs...)
            @printf("p=%d  %s  exp=%d          %6.4e      %14.8e\n",
                    p_ord, label, exp_level, rel, abs_err)
        end
    end
end

# ─────────────────────── Entry point ─────────────────────────────────────────

"""
    run_convergence_table(; degrees, exp_range, conforming, kwargs...)

Print a compact table of relative L2 errors and convergence rates.
"""
function run_convergence_table(;
    degrees::Vector{Int}   = [2, 3, 4],
    exp_range::UnitRange   = 0:5,
    conforming::Bool       = false,
    kwargs...
)
    label = conforming ? "conforming" : "non-conforming"
    exps  = collect(exp_range)
    ne    = length(exps)

    println("\n=== Relative L2 stress error  ($label, adaptive epss=E·s2) ===")
    @printf("%-5s", "p\\h")
    for e in exps;  @printf("  %10s", "exp=$e");  end
    println()
    @printf("%s\n", "-"^(5 + 12*ne))

    for p_ord in degrees
        @printf("p=%-2d", p_ord)
        for e in exps
            rel, _ = solve_plate(p_ord, e; conforming=conforming, kwargs...)
            @printf("  %10.3e", rel)
        end
        println()
    end

    println("\n--- Convergence rates (log₂ ratio of successive levels) ---")
    @printf("%-5s", "p\\h")
    for i in 1:ne-1;  @printf("  %10s", "$(exps[i])→$(exps[i+1])");  end
    println()
    @printf("%s\n", "-"^(5 + 12*(ne-1)))

    for p_ord in degrees
        errs = [solve_plate(p_ord, e; conforming=conforming, kwargs...)[2] for e in exps]
        @printf("p=%-2d", p_ord)
        for i in 1:ne-1
            @printf("  %10.2f", log2(errs[i]/errs[i+1]))
        end
        println()
    end
end

# ─────────────────────── p=1 direct-CP geometry ──────────────────────────────

"""
    open_uniform_kv(n_elem, p) -> Vector{Float64}

Open uniform knot vector for `n_elem` elements of degree `p`.
"""
function open_uniform_kv(n_elem::Int, p::Int)::Vector{Float64}
    n_cp = n_elem + p
    kv = zeros(n_cp + p + 1)
    kv[1:p+1] .= 0.0
    kv[end-p:end] .= 1.0
    for i in 1:n_elem-1
        kv[p+1+i] = i / n_elem
    end
    return kv
end

"""
    plate_geometry_direct_p1(n_ang_p1, n_ang_p2, n_rad; R, a, b)
        -> (B, P, p_mat, n_mat, KV)

Build two-patch **bilinear** (p=1) geometry for the plate with hole by placing
CPs **directly on the circle arc** (not via NURBS degree elevation).

Patch 1: angular 135°→90°, radial: hole arc (r=R) to outer edge (y=b)
Patch 2: angular 135°→180° (= Patch 1 mirrored: x_new = -y, y_new = -x),
         radial: hole arc (r=R) to outer edge (x=-a)

`n_ang_p1`, `n_ang_p2`: angular elements per patch (may differ for non-conforming).
`n_rad`: radial elements (same for both patches).
"""
function plate_geometry_direct_p1(
    n_ang_p1::Int, n_ang_p2::Int, n_rad::Int;
    R::Float64 = 1.0,
    a::Float64 = 4.0,
    b::Float64 = 4.0,
)::Tuple{Matrix{Float64}, Vector{Vector{Int}}, Matrix{Int}, Matrix{Int},
         Vector{Vector{Vector{Float64}}}}

    # Build CP array for Patch 1: θ from 3π/4 (135°) to π/2 (90°)
    # Inner arc: CPs on circle at radius R
    # Outer boundary: straight line from (-a, b) to (0, b)
    function build_patch1(n_ang)
        n_ang_cps = n_ang + 1
        n_rad_cps = n_rad + 1
        B = zeros(n_ang_cps * n_rad_cps, 3)   # [x, y, w]

        for η_idx in 1:n_rad_cps
            t_rad = (η_idx - 1) / n_rad   # 0 = inner, 1 = outer
            for ξ_idx in 1:n_ang_cps
                t_ang = (ξ_idx - 1) / n_ang   # 0 = 135°, 1 = 90°
                θ = (3π/4) * (1 - t_ang) + (π/2) * t_ang

                # Inner point on circle arc
                x_inner = R * cos(θ)
                y_inner = R * sin(θ)

                # Outer point on boundary line y = b
                x_outer = -a * (1 - t_ang)   # from -a (at 135°) to 0 (at 90°)
                y_outer = b

                # Bilinear blend
                cp = (η_idx - 1) * n_ang_cps + ξ_idx
                B[cp, 1] = (1 - t_rad) * x_inner + t_rad * x_outer
                B[cp, 2] = (1 - t_rad) * y_inner + t_rad * y_outer
                B[cp, 3] = 1.0   # unit weight
            end
        end
        return B
    end

    B1_xyw = build_patch1(n_ang_p1)

    # Patch 2: mirror of Patch 1.  Angular range 135°→180° is obtained by
    # reversing ξ direction of Patch 1 and applying (x_new, y_new) = (-y_old, -x_old).
    # But it's cleaner to just build directly:
    function build_patch2(n_ang)
        n_ang_cps = n_ang + 1
        n_rad_cps = n_rad + 1
        B = zeros(n_ang_cps * n_rad_cps, 3)

        for η_idx in 1:n_rad_cps
            t_rad = (η_idx - 1) / n_rad
            for ξ_idx in 1:n_ang_cps
                t_ang = (ξ_idx - 1) / n_ang   # 0 = 180°, 1 = 135°
                θ = π * (1 - t_ang) + (3π/4) * t_ang

                x_inner = R * cos(θ)
                y_inner = R * sin(θ)

                # Outer boundary: line x = -a, y from 0 (at 180°) to b (at 135°)
                x_outer = -a
                y_outer = b * t_ang   # from 0 to b

                cp = (η_idx - 1) * n_ang_cps + ξ_idx
                B[cp, 1] = (1 - t_rad) * x_inner + t_rad * x_outer
                B[cp, 2] = (1 - t_rad) * y_inner + t_rad * y_outer
                B[cp, 3] = 1.0
            end
        end
        return B
    end

    B2_xyw = build_patch2(n_ang_p2)

    # Global B in IGAros format [x, y, z=0, w]
    ncp1 = size(B1_xyw, 1);  ncp2 = size(B2_xyw, 1)
    B_out = zeros(ncp1 + ncp2, 4)
    B_out[1:ncp1,     [1,2,4]] = B1_xyw
    B_out[ncp1+1:end, [1,2,4]] = B2_xyw

    P = [collect(1:ncp1), collect(ncp1+1:ncp1+ncp2)]

    p_mat = fill(1, 2, 2)
    n_mat = [n_ang_p1+1  n_rad+1;
             n_ang_p2+1  n_rad+1]

    kv_ang_p1 = open_uniform_kv(n_ang_p1, 1)
    kv_ang_p2 = open_uniform_kv(n_ang_p2, 1)
    kv_rad    = open_uniform_kv(n_rad,     1)

    KV = Vector{Vector{Vector{Float64}}}([
        [kv_ang_p1, kv_rad],
        [kv_ang_p2, kv_rad],
    ])

    return B_out, P, p_mat, n_mat, KV
end

"""
    solve_plate_p1(exp_level; conforming, ...) -> (err_rel, err_abs)

Plate-with-hole benchmark with **p=1 bilinear elements**.
CPs placed directly on circle arc at each refinement level.
"""
function solve_plate_p1(
    exp_level::Int;
    conforming::Bool   = false,
    Tx::Float64        = 10.0,
    R::Float64         = 1.0,
    a::Float64         = 4.0,
    b::Float64         = 4.0,
    E::Float64         = 1e5,
    nu::Float64        = 0.3,
    epss::Float64      = 0.0,
    NQUAD::Int         = 2,
    NQUAD_mortar::Int  = 10,
    n_ang_base::Int    = 3,
)

    nsd = 2; npd = 2; ned = 2; npc = 2; thickness = 1.0

    n_ang_p1 = (conforming ? n_ang_base : 2 * n_ang_base) * 2^exp_level
    n_ang_p2 = n_ang_base * 2^exp_level
    n_rad    = 2 * 2^exp_level

    s_rad = 1.0 / n_rad
    epss_use = epss > 0.0 ? epss : E * s_rad

    B, P, p_mat, n_mat, KV = plate_geometry_direct_p1(
        n_ang_p1, n_ang_p2, n_rad; R=R, a=a, b=b)

    ncp = size(B, 1)

    nel, nnp, nen = patch_metrics(npc, npd, p_mat, n_mat)
    IEN = build_ien(nsd, npd, npc, p_mat, n_mat, nel, nnp, nen)
    INC = [build_inc(n_mat[pc, :]) for pc in 1:npc]

    # BCs: ux=0 on Patch 1 facet 2 (ξ_max, x≈0), uy=0 on Patch 2 facet 4 (ξ_min, y≈0)
    dBC = [1 2 1 1;
           2 4 1 2]
    bc_per_dof = dirichlet_bc_control_points(p_mat, n_mat, KV, P,
                                              npd, nnp, ned, dBC)
    neq, ID = build_id(bc_per_dof, ned, ncp)
    LM      = build_lm(nen, ned, npc, nel, ID, IEN, P)

    mats = [LinearElastic(E, nu, :plane_stress), LinearElastic(E, nu, :plane_stress)]
    Ub0  = zeros(ncp, nsd)
    K, _ = build_stiffness_matrix(npc, nsd, npd, ned, neq,
                                   p_mat, n_mat, KV, P, B, Ub0,
                                   nen, nel, IEN, INC, LM, mats, NQUAD, thickness)

    traction_fn = (x, y) -> kirsch_stress(x, y; Tx=Tx, R=R)
    F = zeros(neq)
    F = segment_load(n_mat[1,:], p_mat[1,:], KV[1], P[1], B,
                     nnp[1], nen[1], nsd, npd, ned,
                     Int[], 3, ID, F, traction_fn, thickness, NQUAD)
    F = segment_load(n_mat[2,:], p_mat[2,:], KV[2], P[2], B,
                     nnp[2], nen[2], nsd, npd, ned,
                     Int[], 3, ID, F, traction_fn, thickness, NQUAD)

    K_bc, F_bc = enforce_dirichlet(Tuple{Int,Float64}[], K, F)

    # Interface: Patch 2 facet 2 (ξ_max, at 135°) ↔ Patch 1 facet 4 (ξ_min, at 135°)
    pairs = [InterfacePair(2, 2, 1, 4), InterfacePair(1, 4, 2, 2)]
    Pc = build_interface_cps(pairs, p_mat, n_mat, KV, P, npd, nnp)

    C, Z = build_mortar_coupling(Pc, pairs, p_mat, n_mat, KV, P, B,
                                  ID, nnp, ned, nsd, npd, neq, NQUAD_mortar, epss_use,
                                  ElementBasedIntegration())

    U, _ = solve_mortar(K_bc, C, Z, F_bc)

    err_abs, err_ref = l2_stress_error(
        U, ID, npc, nsd, npd, p_mat, n_mat, KV, P, B,
        nen, nel, IEN, INC, mats, NQUAD, thickness; Tx=Tx, R=R)

    d_abs, d_ref = l2_disp_error_plate(
        U, ID, npc, nsd, npd, p_mat, n_mat, KV, P, B,
        nen, nel, IEN, INC, NQUAD, thickness; Tx=Tx, R=R, E=E, nu=nu)

    return (l2_rel=err_abs/err_ref, l2_abs=err_abs,
            d_rel=d_abs/d_ref, d_abs=d_abs)
end

# ─────────────────────── p=1 convergence ──────────────────────────────────────

function run_convergence_plate_p1(;
    exp_range::UnitRange = 0:4,
    kwargs...
)
    println("\n=== Plate with hole: p=1 convergence (non-conforming, direct CP) ===")
    @printf("%-6s  %14s  %14s  %10s\n", "exp", "||e||_L2/||σ||", "||e||_L2", "rate")
    @printf("%s\n", "-"^50)

    errs = Float64[]
    for e in exp_range
        res = solve_plate_p1(e; kwargs...)
        rate = length(errs) > 0 ? log2(errs[end] / res.l2_abs) : NaN
        push!(errs, res.l2_abs)
        @printf("%-6d  %14.4e  %14.4e  %10.2f\n", e, res.l2_rel, res.l2_abs, rate)
    end
end

# ─────────────────────── Force-moment evaluation ──────────────────────────────

"""
    compute_force_moments_plate(D, M, s_cps, m_cps, B; dim=1)

Compute force-moment equilibrium errors δ₀, δ₁, δ₂ for a single interface pass,
using uniform test multiplier λ = 1.  Returns signed values for dimension `dim`.
"""
function compute_force_moments_plate(D, M, s_cps, m_cps, B; dim::Int=1)
    Dd = Matrix(D);  Md = Matrix(M)
    ns = length(s_cps)
    nm = length(m_cps)
    λ  = ones(ns)

    φ_s  = [B[cp, dim] for cp in s_cps]
    φ_m  = [B[cp, dim] for cp in m_cps]
    φ2_s = φ_s .^ 2
    φ2_m = φ_m .^ 2

    δ_0 = dot(λ, Dd * ones(ns) - Md * ones(nm))
    δ_1 = dot(λ, Dd * φ_s  - Md * φ_m)
    δ_2 = dot(λ, Dd * φ2_s - Md * φ2_m)

    return (δ_0 = δ_0, δ_1 = δ_1, δ_2 = δ_2)
end

"""
    run_moment_table_plate(p_ord, exp_level; NQUAD_mortar)

Print force-moment equilibrium errors for the plate-with-hole benchmark.
Compares SPMS (high NQUAD), SPME, DPM (high NQUAD), and TM.
"""
function run_moment_table_plate(
    p_ord::Int = 2,
    exp_level::Int = 2;
    NQUAD_mortar::Int = p_ord + 1,
    R::Float64  = 1.0,
    a::Float64  = 4.0,
    b::Float64  = 4.0,
)
    nsd = 2; npd = 2; npc = 2

    # Build the plate mesh
    B0, P = plate_geometry(p_ord; R=R, a=a, b=b)
    p_mat = fill(p_ord, npc, npd)
    n_mat = fill(p_ord + 1, npc, npd)
    KV    = generate_knot_vectors(npc, npd, p_mat, n_mat)

    s1 = (1/3) / 2^exp_level
    s2 = (1/2) / 2^exp_level
    s2_nc = s2 / 2

    u_ang    = collect(s1    : s1    : 1.0 - s1/2)
    u_rad    = collect(s2    : s2    : 1.0 - s2/2)
    u_rad_nc = collect(s2_nc : s2_nc : 1.0 - s2_nc/2)

    kref_data = Vector{Float64}[
        vcat([1.0, 1.0], u_ang),
        vcat([2.0, 1.0], u_ang),
        vcat([1.0, 2.0], u_rad_nc),
        vcat([2.0, 2.0], u_rad),
    ]

    B0_hack = copy(B0)
    B0_hack[P[1], 2] .+= 1000.0

    n_mat_ref, _, KV_ref, B_hack_ref, P_ref = krefinement(
        nsd, npd, npc, n_mat, p_mat, KV, B0_hack, P, kref_data
    )

    B_ref = copy(B_hack_ref)
    for i in axes(B_ref, 1)
        if B_ref[i, 2] > 100.0
            B_ref[i, 2] -= 1000.0
        end
    end

    _, nnp, _ = patch_metrics(npc, npd, p_mat, n_mat_ref)

    pair1 = InterfacePair(2, 2, 1, 4)   # pass 1: slave=P2, master=P1
    pair2 = InterfacePair(1, 4, 2, 2)   # pass 2: slave=P1, master=P2

    strat = ElementBasedIntegration()
    nquad_ref = 20   # high NQUAD ≈ segment-based reference

    println("\n=== Force-moment analysis: plate with hole, p=$p_ord, exp=$exp_level ===")
    println("Moments computed with uniform test multiplier λ = 1, dim=1 (x)")
    println("NQUAD_mortar = $NQUAD_mortar\n")
    @printf("%-8s  %12s  %12s  %+14s  %+14s  %+14s\n",
            "Method", "δ₀", "δ₁", "δ₂ (pass 1)", "δ₂ (pass 2)", "δ₂ (sum)")
    @printf("%s\n", "─"^80)

    # ── SPMS: single-pass, high NQUAD ≈ segment-based ────────────────────
    D1_ref, M12_ref, s1_cps, m1_cps = build_mortar_mass_matrices(
        pair1, p_mat, n_mat_ref, KV_ref, P_ref, B_ref, nnp, nsd, npd, nquad_ref, strat)
    mom_ref = compute_force_moments_plate(D1_ref, M12_ref, s1_cps, m1_cps, B_ref; dim=1)
    @printf("%-8s  %12.2e  %12.2e  %+14.4e  %14s  %14s\n",
            "SPMS", mom_ref.δ_0, mom_ref.δ_1, mom_ref.δ_2, "—", "—")

    # ── SPME: single-pass, element-based ─────────────────────────────────
    D1_e, M12_e, _, _ = build_mortar_mass_matrices(
        pair1, p_mat, n_mat_ref, KV_ref, P_ref, B_ref, nnp, nsd, npd, NQUAD_mortar, strat)
    mom_sp = compute_force_moments_plate(D1_e, M12_e, s1_cps, m1_cps, B_ref; dim=1)
    @printf("%-8s  %12.2e  %12.2e  %+14.4e  %14s  %14s\n",
            "SPME", mom_sp.δ_0, mom_sp.δ_1, mom_sp.δ_2, "—", "—")

    # ── DPM: dual-pass, high NQUAD ──────────────────────────────────────
    D2_ref, M21_ref, s2_cps, m2_cps = build_mortar_mass_matrices(
        pair2, p_mat, n_mat_ref, KV_ref, P_ref, B_ref, nnp, nsd, npd, nquad_ref, strat)
    mom2_ref = compute_force_moments_plate(D2_ref, M21_ref, s2_cps, m2_cps, B_ref; dim=1)
    @printf("%-8s  %12.2e  %12.2e  %+14.4e  %+14.4e  %+14.4e\n",
            "DPM", mom_ref.δ_0, mom_ref.δ_1,
            mom_ref.δ_2, mom2_ref.δ_2, mom_ref.δ_2 + mom2_ref.δ_2)

    # ── TM: twin mortar, element-based ───────────────────────────────────
    D2_e, M21_e, _, _ = build_mortar_mass_matrices(
        pair2, p_mat, n_mat_ref, KV_ref, P_ref, B_ref, nnp, nsd, npd, NQUAD_mortar, strat)
    mom2_tm = compute_force_moments_plate(D2_e, M21_e, s2_cps, m2_cps, B_ref; dim=1)
    @printf("%-8s  %12.2e  %12.2e  %+14.4e  %+14.4e  %+14.4e\n",
            "TM", mom_sp.δ_0, mom_sp.δ_1,
            mom_sp.δ_2, mom2_tm.δ_2, mom_sp.δ_2 + mom2_tm.δ_2)

    println()
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_convergence_table(degrees=[2, 3, 4], exp_range=0:5, conforming=false)
    run_convergence_table(degrees=[2, 3, 4], exp_range=0:5, conforming=true)
end
