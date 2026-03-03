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
    NQUAD_mortar::Int  = 10
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

    # ── Twin Mortar coupling ──────────────────────────────────────────────────
    # Interface: patch 1 facet 4 (ξ_min, 135° line) ↔ patch 2 facet 2 (ξ_max)
    pairs = [InterfacePair(2, 2, 1, 4),    # slave=pc2(fac2), master=pc1(fac4)
             InterfacePair(1, 4, 2, 2)]    # slave=pc1(fac4), master=pc2(fac2)
    Pc = build_interface_cps(pairs, p_mat, n_mat_ref, KV_ref, P_ref, npd, nnp)

    C, Z = build_mortar_coupling(Pc, pairs, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
                                  ID, nnp, ned, nsd, npd, neq, NQUAD_mortar, epss_use)

    # ── Solve ─────────────────────────────────────────────────────────────────
    U, _ = solve_mortar(K_bc, C, Z, F_bc)

    # ── L2 stress error ───────────────────────────────────────────────────────
    err_abs, err_ref = l2_stress_error(
        U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
        nen, nel, IEN, INC, mats, NQUAD, thickness; Tx=Tx, R=R
    )

    return err_abs / err_ref, err_abs
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

if abspath(PROGRAM_FILE) == @__FILE__
    run_convergence_table(degrees=[2, 3, 4], exp_range=0:5, conforming=false)
    run_convergence_table(degrees=[2, 3, 4], exp_range=0:5, conforming=true)
end
