# examples/bending_beam.jl
#
# 3D rectangular beam with a curved B-spline arc interface.
# Non-conforming mesh tying via Twin Mortar (and Single-Pass / Dual-Pass) methods.
#
# PURPOSE
#   Demonstrate the variational crime of element-based integration (SP-Elm) on a
#   curved non-conforming interface:
#     SP-Seg → O(h^{p+1}) L2-displacement convergence  (optimal)
#     SP-Elm → O(h^p)     L2-displacement convergence  (sub-optimal)
#   TwinMortar and DualPass both recover optimal rates with both strategies.
#
# GEOMETRY  (quarter domain — symmetric in x and z)
#   Full beam: l_x=8, l_y=2, l_z=2.  Quarter model: x ∈ [0, l_x/2] = [0, 4], y ∈ [-l_y/2, l_y/2] = [-1, 1], z ∈ [0, l_z/2] = [0, 1]
#   Default: l_x=8, l_y=2, l_z=2  →  patch domain x ∈ [0, 4], y ∈ [-1, 1], z ∈ [0, 1]
#
#   Symmetry BCs:  ux = 0  on x = 0  (x-symmetry plane)
#                  uz = 0  on z = 0  (z-symmetry plane)
#                  uy = 0  at x = 0, y = -l_y/2, z = 0 (prevents rigid-body translation in y; on Patch 1)
#
#   Interface arc (quadratic B-spline in x-y, uniform in z):
#   arc_y(x) = (l_y/4)*[0.5*(2x/l_x) + 0.5*(2x/l_x)²]  (concave up, y=0 at x=0, y=l_y/4 at x=l_x/2)
#
# MATERIAL:  E = 1000, ν = 0
#
# LOADING:   Neumann traction t_x = 2p/(l_y)·y at x = l_x/2 face.
#            Consistent with eq.(53): σ_xx = 2p*y/l_y  (y is already centred).
#
# EXACT SOLUTION:
#   u_x = 2p/(E·l_y)·x·y,   u_y = p/(E·l_y)·(-x² -ν·y² +ν·z²),   u_z = -2pν/(E·l_y)·y·z
#   (y ∈ [-l_y/2, l_y/2] is the centred coordinate, as in main.tex eq.(53))
#
# PATCHES
#   Patch 1 (lower, slave/finer):    y ∈ [-l_y/2,       arc_y(x)]
#   Patch 2 (upper, master/coarser): y ∈ [arc_y(x), l_y/2]
#
# PARAMETRIZATION  (ξ = x fastest, η = y middle, ζ = z slowest)
#   Interface: Patch 1 facet 3 (η = n₂) ↔ Patch 2 facet 5 (η = 1)
#
# NON-CONFORMING MESH  configurable ratio in x via n_x_lower_base:n_x_upper_base
#   default 2:1  (n_x_lower_base=2, n_x_upper_base=1)
#   example 3:2  (n_x_lower_base=3, n_x_upper_base=2)

import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using IGAros
using LinearAlgebra, SparseArrays, Printf

# ─── Exact solution ────────────────────────────────────────────────────────────

"""
    beam_exact_disp(x, y, z; p_load=10.0, E=1000.0, l_x=8.0, l_y=4.0, nu=0.0) -> (ux, uy, uz)

Exact solution from main.tex eq.(53): pure bending, σ_xx = 2p*y/l_y.
  y ∈ [-l_y/2, l_y/2] is the centred coordinate,   traction t_x = 2p*y/l_y at x = l_x/2
  u_x = 2p/(E*l_y)*x*y,   u_y = p/(E*l_y)*(-x²-ν*y²+ν*z²),   u_z = -2pν/(E*l_y)*y*z
"""
function beam_exact_disp(x::Real, y::Real, z::Real;
                          p_load::Float64 = 10.0,
                          E::Float64      = 1000.0,
                          l_x::Float64    = 8.0,
                          l_y::Float64    = 2.0,
                          nu::Float64     = 0.0)
    C = p_load / E
    return 2C/l_y * x * y,
           C/l_y * (-x^2 - nu*y^2 + nu*z^2),
           -2C*nu/l_y * y * z
end

"""
    arc_y_beam(x; l_x=8.0, l_y=2.0) -> y

Centred y-coordinate of the arc interface at position x ∈ [0, l_x/2].
  arc_y = (l_y/4) * 0.5*(ξ + ξ²),  ξ = 2x/l_x ∈ [0,1]
Concave-up (d²y/dx² > 0): 0 at x=0, monotonically increasing to l_y/4 at x=l_x/2.
Amplitude scales with l_y so the upper patch never degenerates (arc < l_y/2 always).
"""
function arc_y_beam(x::Real; l_x::Float64 = 8.0, l_y::Float64 = 2.0)
    ξ = 2.0 * x / l_x   # ξ ∈ [0,1] as x ∈ [0, l_x/2]
    return (l_y/8) * (ξ + ξ^2)   # = (l_y/4)*0.5*(ξ+ξ²); reaches l_y/4 at ξ=1
end

# ─── Bezier degree-elevation helpers ──────────────────────────────────────────

# Single one-degree elevation of a (p+1)×ncols Bezier control-point row.
function _bezier_elev_beam(Bh::Matrix{Float64})::Matrix{Float64}
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

# Elevate a 3D patch in the ξ (x, fastest) direction n_elev times.
# CP ordering: A = (k-1)*n1*n2 + (j-1)*n1 + i  (i fastest, k slowest).
function _elevate_x_beam(B_flat::Matrix{Float64},
                          n1::Int, n2::Int, n3::Int, n_elev::Int)
    ncols = size(B_flat, 2)
    for _ in 1:n_elev
        n1n = n1 + 1
        Bn  = zeros(n1n * n2 * n3, ncols)
        for k in 1:n3, j in 1:n2
            s  = (k-1)*n1 *n2 + (j-1)*n1  + 1
            sn = (k-1)*n1n*n2 + (j-1)*n1n + 1
            Bh = copy(B_flat[s:s+n1-1, :])
            Bh[:, 1:ncols-1] .*= Bh[:, ncols:ncols]
            Bh = _bezier_elev_beam(Bh)
            Bh[:, 1:ncols-1] ./= Bh[:, ncols:ncols]
            Bn[sn:sn+n1n-1, :] = Bh
        end
        B_flat = Bn;  n1 = n1n
    end
    return B_flat, n1
end

# Elevate in the η (y, middle) direction n_elev times.
function _elevate_y_beam(B_flat::Matrix{Float64},
                          n1::Int, n2::Int, n3::Int, n_elev::Int)
    ncols = size(B_flat, 2)
    for _ in 1:n_elev
        n2n = n2 + 1
        Bn  = zeros(n1 * n2n * n3, ncols)
        for k in 1:n3, i in 1:n1
            fiber = B_flat[[(k-1)*n1*n2 + (j-1)*n1 + i for j in 1:n2], :]
            Bh = copy(fiber)
            Bh[:, 1:ncols-1] .*= Bh[:, ncols:ncols]
            Bh = _bezier_elev_beam(Bh)
            Bh[:, 1:ncols-1] ./= Bh[:, ncols:ncols]
            for j_new in 1:n2n
                Bn[(k-1)*n1*n2n + (j_new-1)*n1 + i, :] = Bh[j_new, :]
            end
        end
        B_flat = Bn;  n2 = n2n
    end
    return B_flat, n2
end

# Elevate in the ζ (z, slowest) direction n_elev times.
function _elevate_z_beam(B_flat::Matrix{Float64},
                          n1::Int, n2::Int, n3::Int, n_elev::Int)
    ncols = size(B_flat, 2)
    for _ in 1:n_elev
        n3n = n3 + 1
        Bn  = zeros(n1 * n2 * n3n, ncols)
        for j in 1:n2, i in 1:n1
            fiber = B_flat[[(k-1)*n1*n2 + (j-1)*n1 + i for k in 1:n3], :]
            Bh = copy(fiber)
            Bh[:, 1:ncols-1] .*= Bh[:, ncols:ncols]
            Bh = _bezier_elev_beam(Bh)
            Bh[:, 1:ncols-1] ./= Bh[:, ncols:ncols]
            for k_new in 1:n3n
                Bn[(k_new-1)*n1*n2 + (j-1)*n1 + i, :] = Bh[k_new, :]
            end
        end
        B_flat = Bn;  n3 = n3n
    end
    return B_flat, n3
end

# ─── Beam geometry (p ≥ 2) ────────────────────────────────────────────────────

"""
    beam_geometry(p_ord; l_x, l_y, l_z) -> (B, P)

Build the two-patch NURBS geometry for a rectangular beam with a curved arc interface.

  Patch 1 (lower):  y ∈ [-l_y/2,   arc_y(x)]   (slave, finer in x)
  Patch 2 (upper):  y ∈ [arc_y(x), l_y/2]       (master, coarser in x)

Arc is a quadratic B-spline with Bernstein CPs (0,0), (l_x/4,l_y/16), (l_x/2,l_y/4) in x-y.
Concave up (d²y/dx² > 0), monotonically increasing from y=0 at x=0 to y=l_y/4 at x=l_x/2.
Scales with l_y so the upper patch always has positive height (arc < l_y/2 everywhere).
The arc is polynomial of degree 2 and is represented exactly for p_ord ≥ 2.

Parametrization: ξ = x (fastest), η = y (middle), ζ = z (slowest).
Domain: x ∈ [0, l_x/2], y ∈ [-l_y/2, l_y/2], z ∈ [0, l_z/2] (quarter-beam by x- and z-symmetry).
Initial coarse geometry: p = [2,1,1], n = [3,2,2].
After degree elevation to p_ord: n = [p_ord+1, p_ord+1, p_ord+1] per patch.
"""
function beam_geometry(p_ord::Int;
                        l_x::Float64 = 8.0,
                        l_y::Float64 = 2.0,
                        l_z::Float64 = 2.0)::Tuple{Matrix{Float64}, Vector{Vector{Int}}}

    # Arc y-values at the 3 coarse x control points (x ∈ [0, l_x/2], centred coords)
    # Bernstein CPs of arc_y = (l_y/8)*(ξ+ξ²): [0, l_y/16, l_y/4]
    # Arc reaches l_y/4 at x=l_x/2 (scales with beam height, always < l_y/2)
    y_arc = [0.0, l_y/16, l_y/4]
    x_cps = [0.0, l_x/4, l_x/2]
    z_cps = [0.0, l_z/2]

    # Initial coarse geometry: p=[2,1,1], n1=3, n2=2, n3=2
    # CP ordering: A = (k-1)*n1*n2 + (j-1)*n1 + i  (i=x fastest, k=z slowest)
    n1, n2, n3 = 3, 2, 2

    # Patch 1 (lower): j=1 → y = -l_y/2 (bottom), j=2 → arc_y (top/interface)
    B1 = zeros(n1 * n2 * n3, 4)
    for k in 1:n3, j in 1:n2, i in 1:n1
        A = (k-1)*n1*n2 + (j-1)*n1 + i
        B1[A, 1] = x_cps[i]
        B1[A, 2] = (j == 1) ? -l_y/2 : y_arc[i]
        B1[A, 3] = z_cps[k]
        B1[A, 4] = 1.0
    end

    # Patch 2 (upper): j=1 → arc_y (bottom/interface), j=2 → l_y/2 (top)
    B2 = zeros(n1 * n2 * n3, 4)
    for k in 1:n3, j in 1:n2, i in 1:n1
        A = (k-1)*n1*n2 + (j-1)*n1 + i
        B2[A, 1] = x_cps[i]
        B2[A, 2] = (j == 1) ? y_arc[i] : l_y/2
        B2[A, 3] = z_cps[k]
        B2[A, 4] = 1.0
    end

    # Degree elevation: x (2 → p_ord), y (1 → p_ord), z (1 → p_ord)
    # Apply identical operations to B1 and B2 (same initial structure).
    B1, n1a = _elevate_x_beam(B1, n1, n2, n3, p_ord - 2)
    B2, _   = _elevate_x_beam(B2, n1, n2, n3, p_ord - 2)

    B1, n2a = _elevate_y_beam(B1, n1a, n2, n3, p_ord - 1)
    B2, _   = _elevate_y_beam(B2, n1a, n2, n3, p_ord - 1)

    B1, _   = _elevate_z_beam(B1, n1a, n2a, n3, p_ord - 1)
    B2, _   = _elevate_z_beam(B2, n1a, n2a, n3, p_ord - 1)

    ncp1 = size(B1, 1);  ncp2 = size(B2, 1)
    return vcat(B1, B2), [collect(1:ncp1), collect(ncp1+1:ncp1+ncp2)]
end

# ─── p=1 open uniform knot vector ─────────────────────────────────────────────

if !@isdefined(open_uniform_kv)
    function open_uniform_kv(n_elem::Int, p::Int)::Vector{Float64}
        n_cp = n_elem + p
        kv   = zeros(n_cp + p + 1)
        kv[end-p:end] .= 1.0
        for i in 1:n_elem-1; kv[p+1+i] = i / n_elem; end
        return kv
    end
end

# ─── Beam geometry p=1 direct mesh ────────────────────────────────────────────

"""
    beam_geometry_p1(n_x_lower, n_x_upper, n_y, n_z; l_x, l_y, l_z) -> (B, P)

Build the two-patch **bilinear** (p=1) geometry for a 3D beam with a FLAT interface
at y = l_y/2.  A flat interface is used for p=1 because p=1 cannot represent curved
surfaces exactly — different mesh resolutions produce different piecewise-linear
approximations of a curved arc, leading to a geometric gap that corrupts the mortar.
A flat mid-plane (y = l_y/2) is represented exactly by any p=1 mesh, so both patches
always share the same physical interface regardless of refinement.

  n_x_lower: number of x-elements in Patch 1 (finer, slave)
  n_x_upper: number of x-elements in Patch 2 (coarser, master)
  n_y:       number of y-elements in both patches
  n_z:       number of z-elements in both patches
"""
function beam_geometry_p1(
    n_x_lower::Int, n_x_upper::Int, n_y::Int, n_z::Int;
    l_x::Float64 = 8.0,
    l_y::Float64 = 2.0,
    l_z::Float64 = 2.0
)::Tuple{Matrix{Float64}, Vector{Vector{Int}}}

    l_x_h  = l_x / 2    # half-domain x-extent (x-symmetry)
    l_y_h  = l_y / 2    # half-domain y-extent (centred: y ∈ [-l_y/2, l_y/2])
    l_z_h  = l_z / 2    # half-domain z-extent (z-symmetry)

    # Patch 1 (lower): y from -l_y/2 to 0 (centred flat interface at y=0)
    n1_1 = n_x_lower + 1;  n2_1 = n_y + 1;  n3_1 = n_z + 1
    B1 = zeros(n1_1 * n2_1 * n3_1, 4)
    for k in 1:n3_1, j in 1:n2_1, i in 1:n1_1
        A = (k-1)*n1_1*n2_1 + (j-1)*n1_1 + i
        B1[A, 1] = (i - 1) / n_x_lower * l_x_h
        B1[A, 2] = -l_y_h + (j - 1) / n_y * l_y_h
        B1[A, 3] = (k - 1) / n_z * l_z_h
        B1[A, 4] = 1.0
    end

    # Patch 2 (upper): y from 0 to l_y/2
    n1_2 = n_x_upper + 1;  n2_2 = n_y + 1;  n3_2 = n_z + 1
    B2 = zeros(n1_2 * n2_2 * n3_2, 4)
    for k in 1:n3_2, j in 1:n2_2, i in 1:n1_2
        A = (k-1)*n1_2*n2_2 + (j-1)*n1_2 + i
        B2[A, 1] = (i - 1) / n_x_upper * l_x_h
        B2[A, 2] = (j - 1) / n_y * l_y_h
        B2[A, 3] = (k - 1) / n_z * l_z_h
        B2[A, 4] = 1.0
    end

    ncp1 = size(B1, 1);  ncp2 = size(B2, 1)
    return vcat(B1, B2), [collect(1:ncp1), collect(ncp1+1:ncp1+ncp2)]
end

"""
    beam_geometry_p1_curved(n_x_lower, n_x_upper, n_y, n_z; l_x, l_y, l_z) -> (B, P)

Build the two-patch **bilinear** (p=1) geometry for a 3D beam with a CURVED arc
interface.  CPs are placed exactly on arc_y_beam(x) at the interface rows so the
geometry error is O(h²) — smaller than the O(h) discretisation error.

Each patch samples the arc independently at its own x-mesh, so Patch 1 (3n or 2n
x-elements) and Patch 2 (2n or n x-elements) represent slightly different piecewise-
linear arcs.  This non-conforming curved interface is exactly the setting needed to
exhibit the variational crime of element-based integration (SP-Elm).
"""
function beam_geometry_p1_curved(
    n_x_lower::Int, n_x_upper::Int, n_y::Int, n_z::Int;
    l_x::Float64 = 8.0,
    l_y::Float64 = 2.0,
    l_z::Float64 = 2.0
)::Tuple{Matrix{Float64}, Vector{Vector{Int}}}

    l_x_h = l_x / 2   # half-domain x-extent (x-symmetry)
    l_y_h = l_y / 2   # half y-extent (centred: y ∈ [-l_y/2, l_y/2])
    l_z_h = l_z / 2   # half-domain z-extent (z-symmetry)

    # Patch 1 (lower): y ∈ [-l_y/2, arc_y(x)]; top row (j=n_y+1) is on the arc
    n1_1 = n_x_lower + 1;  n2_1 = n_y + 1;  n3_1 = n_z + 1
    B1 = zeros(n1_1 * n2_1 * n3_1, 4)
    for k in 1:n3_1, j in 1:n2_1, i in 1:n1_1
        A  = (k-1)*n1_1*n2_1 + (j-1)*n1_1 + i
        xi = (i - 1) / n_x_lower * l_x_h
        ya = arc_y_beam(xi; l_x=l_x, l_y=l_y)
        B1[A, 1] = xi
        B1[A, 2] = -l_y_h + (j - 1) / n_y * (ya + l_y_h)  # linear blend: -l_y/2 at j=1, arc_y at j=n_y+1
        B1[A, 3] = (k - 1) / n_z * l_z_h
        B1[A, 4] = 1.0
    end

    # Patch 2 (upper): y ∈ [arc_y(x), l_y/2]; bottom row (j=1) is on the arc
    n1_2 = n_x_upper + 1;  n2_2 = n_y + 1;  n3_2 = n_z + 1
    B2 = zeros(n1_2 * n2_2 * n3_2, 4)
    for k in 1:n3_2, j in 1:n2_2, i in 1:n1_2
        A  = (k-1)*n1_2*n2_2 + (j-1)*n1_2 + i
        xi = (i - 1) / n_x_upper * l_x_h
        ya = arc_y_beam(xi; l_x=l_x, l_y=l_y)
        B2[A, 1] = xi
        B2[A, 2] = ya + (j - 1) / n_y * (l_y_h - ya)  # arc_y at j=1, l_y/2 at j=n_y+1
        B2[A, 3] = (k - 1) / n_z * l_z_h
        B2[A, 4] = 1.0
    end

    ncp1 = size(B1, 1);  ncp2 = size(B2, 1)
    return vcat(B1, B2), [collect(1:ncp1), collect(ncp1+1:ncp1+ncp2)]
end

# ─── L2 displacement error ─────────────────────────────────────────────────────

"""
    l2_disp_error_beam(U, ID, npc, nsd, npd, p, n, KV, P, B,
                       nen, nel, IEN, INC, NQUAD, disp_fn)
        -> (err_abs, err_ref)

Compute L2 displacement error against the exact solution `disp_fn(x,y,z)`.
  err_abs² = ∫_Ω ‖u_h − u_ex‖² dΩ
  err_ref² = ∫_Ω ‖u_ex‖² dΩ
"""
function l2_disp_error_beam(
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
    disp_fn::Function   # (x, y, z) -> (ux, uy, uz)
)::Tuple{Float64, Float64}

    ncp = size(B, 1)
    Ub  = zeros(ncp, nsd)
    for A in 1:ncp, i in 1:nsd
        eq = ID[i, A];  eq != 0 && (Ub[A, i] = U[eq])
    end

    err2 = 0.0;  ref2 = 0.0
    GPW  = gauss_product(NQUAD, npd)

    for pc in 1:npc
        ien = IEN[pc];  inc = INC[pc]
        for el in 1:nel[pc]
            anchor = ien[el, 1]
            n0     = inc[anchor]
            for (gp, gw) in GPW
                R_s, _, _, detJ, _ = shape_function(
                    p[pc,:], n[pc,:], KV[pc], B, P[pc], gp,
                    nen[pc], nsd, npd, el, n0, ien, inc
                )
                detJ <= 0 && continue
                gwJ = gw * detJ

                Ue_mat = Ub[P[pc][ien[el,:]], 1:nsd]
                u_h    = Ue_mat' * R_s              # (nsd,)

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

# ─── VTK postprocessing ───────────────────────────────────────────────────────

# Sample a knot vector at element boundaries + n_per_span interior points per span.
# The resulting grid lines coincide with element boundaries, making mesh non-conformity visible.
function _kv_sample(kv_vec::AbstractVector{Float64}, n_per_span::Int)::Vector{Float64}
    breaks = unique(kv_vec)   # break-points (unique knot values)
    pts = Float64[]
    for i in 1:length(breaks)-1
        append!(pts, range(breaks[i], breaks[i+1]; length = n_per_span + 1)[1:end-1])
    end
    push!(pts, breaks[end])
    return pts
end

"""
    write_vtk_beam(prefix, U, ID, npc, nsd, npd, p_mat, n_mat, KV, P, B,
                   nen_vec, IEN, INC, E, nu; n_vis=4)

Write one VTK STRUCTURED_GRID file per patch:  `prefix_1.vtk`, `prefix_2.vtk`, ...

Sampling is aligned with element boundaries: `n_vis` points per knot span per direction.
Grid lines in the VTK therefore coincide with the actual IGA element edges, making the
non-conforming mesh ratio between patches clearly visible.

Point data written:
  VECTORS displacement  — (ux, uy, uz)
  SCALARS stress_xx     — σ_xx (bending stress)
  SCALARS stress_yy     — σ_yy
  SCALARS stress_zz     — σ_zz
  SCALARS von_mises     — σ_vm
"""
function write_vtk_beam(
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
    n_vis::Int = 4   # points per knot span per direction (not total)
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

        # Per-direction samples aligned with knot span boundaries.
        # Each patch gets its own sampling grid → non-conforming mesh ratio is visible.
        s1 = _kv_sample(collect(kv[1]), n_vis)
        s2 = _kv_sample(collect(kv[2]), n_vis)
        s3 = _kv_sample(collect(kv[3]), n_vis)
        nd1, nd2, nd3 = length(s1), length(s2), length(s3)
        n_pts = nd1 * nd2 * nd3

        pts  = zeros(3, n_pts)
        disp = zeros(3, n_pts)
        sxx  = zeros(n_pts); syy = zeros(n_pts); szz = zeros(n_pts)
        svm  = zeros(n_pts)

        # Number of elements per parametric direction (needed for element index)
        n_elem = [nv[d] - pv[d] for d in 1:npd]

        idx = 0
        # Loop: i1 (ξ) fastest, i3 (ζ) slowest → matches VTK STRUCTURED_GRID ordering
        for xi3 in s3, xi2 in s2, xi1 in s1
            idx += 1
            Xi = [xi1, xi2, xi3]

            # Knot span per direction
            n0 = [find_span(nv[d]-1, pv[d], Float64(Xi[d]), collect(kv[d])) for d in 1:npd]

            # Element index (ξ fastest, ζ slowest — matches build_ien ordering)
            e = [n0[d] - pv[d] for d in 1:npd]
            el = (e[3]-1)*n_elem[1]*n_elem[2] + (e[2]-1)*n_elem[1] + e[1]

            # Parent coords in [-1,1]
            xi_tilde = zeros(npd)
            for d in 1:npd
                kv_d = collect(kv[d])
                a, b = kv_d[n0[d]], kv_d[n0[d]+1]
                xi_tilde[d] = (b > a) ? clamp((2*Xi[d] - a - b) / (b - a), -1.0, 1.0) : 0.0
            end

            R, dR_dx, _, detJ, _ = shape_function(
                pv, nv, kv, B, Ppc, xi_tilde,
                nen, nsd, npd, el, n0, ien, inc
            )

            el_nodes = Ppc[ien[el, :]]

            # Physical coordinates
            Xe = B[el_nodes, 1:nsd]
            X  = Xe' * R
            pts[1:nsd, idx] = X

            # Displacement
            Ue = Ub[el_nodes, :]
            u  = Ue' * R
            disp[1:nsd, idx] = u

            # Stress (skip at degenerate points)
            if detJ > 0.0
                dN_dX = Matrix(dR_dx')           # nsd × nen
                B_mat = strain_displacement_matrix(nsd, nen, dN_dX)
                # u_vec: [u1_x, u1_y, u1_z, u2_x, u2_y, u2_z, ...]
                u_vec = Vector{Float64}(undef, nsd * nen)
                for a in 1:nen, d in 1:nsd; u_vec[(a-1)*nsd + d] = Ue[a, d]; end
                sig = D * (B_mat * u_vec)  # [σxx, σyy, σzz, τxy, τyz, τzx]
                sxx[idx] = sig[1]; syy[idx] = sig[2]; szz[idx] = sig[3]
                svm[idx] = sqrt(max(0.0,
                    0.5*((sig[1]-sig[2])^2 + (sig[2]-sig[3])^2 + (sig[3]-sig[1])^2)
                    + 3*(sig[4]^2 + sig[5]^2 + sig[6]^2)))
            end
        end

        fname = "$(prefix)_$(pc).vtk"
        open(fname, "w") do f
            println(f, "# vtk DataFile Version 2.0")
            println(f, "Bending beam patch $pc")
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
            for (name, arr) in [("stress_xx", sxx), ("stress_yy", syy),
                                 ("stress_zz", szz), ("von_mises",  svm)]
                println(f, "SCALARS $name float 1")
                println(f, "LOOKUP_TABLE default")
                for i in 1:n_pts; @printf f "%e\n" arr[i]; end
            end
        end
        @printf "  Wrote %s  (%d×%d×%d grid)\n" fname nd1 nd2 nd3
    end
end

# ─── Single-level solve (p ≥ 2) ───────────────────────────────────────────────

"""
    solve_beam(p_ord, exp_level; conforming, epss, ...) -> (err_rel, err_abs)

Run one refinement level of the 3D beam benchmark using NURBS of degree p_ord ≥ 2.
Non-conforming ratio set by n_x_lower_base:n_x_upper_base (default 2:1).

BCs:
  Homogeneous Dirichlet (via bc_per_dof, ID=0):
    ux = 0  on x = 0 face  (facet 4, both patches)  — x-symmetry
    uz = 0  on z = 0 face  (facet 1, both patches)  — z-symmetry
  Non-homogeneous Dirichlet (via enforce_dirichlet):
    uy = 0  at (0, -l_y/2, 0) — bottom face of Patch 1, prevents rigid-body translation in y
  Neumann (via segment_load, assembled into F):
    t_x = 2*p_load*y/l_y  on x = l_x/2 face  (facet 2, both patches)
  Natural (automatic, traction-free):
    all remaining faces: y = -l_y/2, y = l_y/2, z = l_z/2, and the mortar interface
"""
function solve_beam(
    p_ord::Int,
    exp_level::Int;
    conforming::Bool              = false,
    l_x::Float64                  = 8.0,
    l_y::Float64                  = 2.0,
    l_z::Float64                  = 2.0,
    E::Float64                    = 1000.0,
    nu::Float64                   = 0.0,
    p_load::Float64               = 10.0,
    epss::Float64                 = 0.0,
    NQUAD::Int                    = p_ord + 1,
    NQUAD_mortar::Int             = p_ord + 2,
    strategy::IntegrationStrategy    = ElementBasedIntegration(),
    formulation::FormulationStrategy = TwinMortarFormulation(),
    normal_strategy::NormalStrategy  = SlaveNormal(),
    n_x_lower_base::Int           = 2,   # slave  (Patch 1) base x-elements
    n_x_upper_base::Int           = 1,   # master (Patch 2) base x-elements
    vtk_prefix::String            = "",  # write VTK if non-empty (e.g. "beam")
    n_vis::Int                    = 4,   # VTK sampling points per knot span per direction
)::Tuple{Float64, Float64}

    nsd = 3;  npd = 3;  ned = 3;  npc = 2

    # ── Initial coarse geometry (single Bezier span per patch) ───────────────
    B0, P  = beam_geometry(p_ord; l_x=l_x, l_y=l_y, l_z=l_z)
    p_mat  = fill(p_ord, npc, npd)
    n_mat  = fill(p_ord + 1, npc, npd)
    KV     = generate_knot_vectors(npc, npd, p_mat, n_mat)

    # ── y-offset hack: prevent CP merging at interface during k-refinement ────
    B0_hack = copy(B0)
    B0_hack[P[1], 2] .+= 1000.0

    # ── h-refinement ──────────────────────────────────────────────────────────
    n_x  = n_x_upper_base * 2^exp_level
    n_xl = conforming ? n_x : n_x_lower_base * 2^exp_level
    n_y  = 2^exp_level
    n_z  = max(1, 2^exp_level)

    u_x_l = [i / n_xl for i in 1:n_xl - 1]
    u_x_u = [i / n_x  for i in 1:n_x  - 1]
    u_y   = [i / n_y  for i in 1:n_y  - 1]
    u_z   = [i / n_z  for i in 1:n_z  - 1]

    kref_data = Vector{Float64}[
        vcat([1.0, 1.0], u_x_l),
        vcat([1.0, 2.0], u_y),
        vcat([1.0, 3.0], u_z),
        vcat([2.0, 1.0], u_x_u),
        vcat([2.0, 2.0], u_y),
        vcat([2.0, 3.0], u_z),
    ]

    n_mat_ref, _, KV_ref, B_ref_hack, P_ref = krefinement(
        nsd, npd, npc, n_mat, p_mat, KV, B0_hack, P, kref_data
    )

    B_ref = copy(B_ref_hack)
    for i in axes(B_ref, 1)
        B_ref[i, 2] > 100.0 && (B_ref[i, 2] -= 1000.0)
    end
    ncp = size(B_ref, 1)

    epss_use = epss > 0.0 ? epss : 1.0e6

    # ── Connectivity ──────────────────────────────────────────────────────────
    nel, nnp, nen = patch_metrics(npc, npd, p_mat, n_mat_ref)
    IEN = build_ien(nsd, npd, npc, p_mat, n_mat_ref, nel, nnp, nen)
    INC = [build_inc(n_mat_ref[pc, :]) for pc in 1:npc]

    # ── Homogeneous Dirichlet BCs (bc_per_dof, ID = 0) ────────────────────────
    # ux = 0  on x = 0 face (facet 4, ξ=1)  — x-symmetry plane
    # uz = 0  on z = 0 face (facet 1, ζ=1)  — z-symmetry plane
    dBC = [1 4 2 1 2;    # ux = 0, facet 4 (ξ=1, x=0)
           3 1 2 1 2]    # uz = 0, facet 1 (ζ=1, z=0)

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

    # ── Non-homogeneous Dirichlet BC: uy = 0 at origin ────────────────────────
    F   = zeros(neq)
    IND = Tuple{Int, Float64}[]
    tol = 1e-10

    for pc in 1:npc
        for loc_A in 1:nnp[pc]
            cp = P_ref[pc][loc_A]
            x  = B_ref[cp, 1];  y = B_ref[cp, 2];  z = B_ref[cp, 3]
            if abs(x) < tol && abs(y + l_y/2) < tol && abs(z) < tol
                eq = ID[2, cp]
                eq != 0 && push!(IND, (eq, 0.0))
            end
        end
    end
    unique!(IND)

    # ── Neumann BC: traction t_x = 2*p_load*y/l_y at x = l_x/2 face ─────────
    traction_fn = (x, y, z) -> begin
        σ = zeros(3, 3)
        σ[1, 1] = 2*p_load * y / l_y
        σ
    end
    for pc in 1:npc
        F = segment_load(n_mat_ref[pc, :], p_mat[pc, :], KV_ref[pc], P_ref[pc], B_ref,
                          nnp[pc], nen[pc], nsd, npd, ned,
                          Int[], 2, ID, F, traction_fn, 1.0, NQUAD)
    end

    K_bc, F_bc = enforce_dirichlet(IND, K, F)

    # ── Mortar coupling at the curved arc interface ───────────────────────────
    pairs_full = [InterfacePair(1, 3, 2, 5), InterfacePair(2, 5, 1, 3)]
    pairs_sp   = [InterfacePair(1, 3, 2, 5)]
    pairs = formulation isa SinglePassFormulation ? pairs_sp : pairs_full

    Pc = build_interface_cps(pairs, p_mat, n_mat_ref, KV_ref, P_ref, npd, nnp, formulation)
    C, Z = build_mortar_coupling(Pc, pairs, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
                                  ID, nnp, ned, nsd, npd, neq, NQUAD_mortar, epss_use,
                                  strategy, formulation, normal_strategy)

    # Zero C rows for non-homogeneous Dirichlet DOFs (from enforce_dirichlet).
    # Homogeneous DOFs (ID=0) are already excluded from C by the mortar assembly.
    fixed_eqs = Set(A for (A, _) in IND)
    rows_C, cols_C, vals_C = findnz(C)
    keep  = [i for i in eachindex(rows_C) if !(rows_C[i] in fixed_eqs)]
    C_bc  = sparse(rows_C[keep], cols_C[keep], vals_C[keep], size(C, 1), size(C, 2))

    # Remove inactive Lagrange multipliers: zero columns of C with no Z diagonal.
    # Occurs when slave interface DOFs are Dirichlet-constrained (SP only; no-op for TM/DP).
    _, cols_C_nz, _ = findnz(C_bc)
    active_lm = sort(unique(cols_C_nz))
    if length(active_lm) < size(C_bc, 2)
        C_bc = C_bc[:, active_lm]
        Z    = Z[active_lm, active_lm]
    end

    U, _ = solve_mortar(K_bc, C_bc, Z, F_bc)

    disp_fn = (x, y, z) -> beam_exact_disp(x, y, z; p_load=p_load, E=E, l_x=l_x, l_y=l_y)
    err_abs, err_ref = l2_disp_error_beam(
        U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
        nen, nel, IEN, INC, NQUAD, disp_fn
    )

    if !isempty(vtk_prefix)
        write_vtk_beam(vtk_prefix, U, ID, npc, nsd, npd,
                       p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
                       nen, IEN, INC, E, nu; n_vis=n_vis)
    end

    return err_abs / err_ref, err_abs
end

# ─── Single-level solve p=1 ───────────────────────────────────────────────────

"""
    solve_beam_p1(exp_level; conforming, epss, ...) -> (err_rel, err_abs)

Beam benchmark with **p=1 trilinear elements** and direct mesh generation.
CPs are placed directly on the arc at each level so geometry error = O(h²),
faster than the O(h) discretisation error.

BCs:
  Homogeneous Dirichlet (bc_per_dof): ux=0 on x=0, uz=0 on z=0.
  Non-homogeneous Dirichlet: uy=0 at (0, l_y/2, 0).
  Neumann: t_x = 2*p_load*y/l_y  at x = l_x/2 face (y is centred).

Non-conforming mesh at level exp_level:
  n_x_upper = n_x_upper_base · 2^exp_level   (upper/master patch, coarser)
  n_x_lower = n_x_lower_base · 2^exp_level   (lower/slave  patch, finer)
  n_y = n_z = max(1, 2^exp_level)
"""
function solve_beam_p1(
    exp_level::Int;
    conforming::Bool              = false,
    l_x::Float64                  = 8.0,
    l_y::Float64                  = 2.0,
    l_z::Float64                  = 2.0,
    E::Float64                    = 1000.0,
    nu::Float64                   = 0.0,
    p_load::Float64               = 10.0,
    epss::Float64                 = 0.0,
    NQUAD::Int                    = 2,
    NQUAD_mortar::Int             = 3,
    strategy::IntegrationStrategy    = ElementBasedIntegration(),
    formulation::FormulationStrategy = TwinMortarFormulation(),
    normal_strategy::NormalStrategy  = SlaveNormal(),
    n_x_lower_base::Int           = 4,   # slave  (Patch 1) base x-elements
    n_x_upper_base::Int           = 3,   # master (Patch 2) base x-elements
    curved::Bool                  = true,  # true → curved arc interface (variational crime demo)
    vtk_prefix::String            = "",  # write VTK if non-empty (e.g. "beam_p1")
    n_vis::Int                    = 4,   # VTK sampling points per knot span per direction
)::Tuple{Float64, Float64}

    nsd = 3;  npd = 3;  ned = 3;  npc = 2;  p_ord = 1

    n_x_u = n_x_upper_base * 2^exp_level
    n_x_l = conforming ? n_x_u : n_x_lower_base * 2^exp_level
    n_y   = max(1, 2^exp_level)
    n_z   = max(1, 2^exp_level)

    # ── Geometry ──────────────────────────────────────────────────────────────
    geom_fn = curved ? beam_geometry_p1_curved : beam_geometry_p1
    B_ref, P_ref = geom_fn(n_x_l, n_x_u, n_y, n_z;
                            l_x=l_x, l_y=l_y, l_z=l_z)
    ncp = size(B_ref, 1)

    p_mat     = fill(p_ord, npc, npd)
    n_mat_ref = [n_x_l+1  n_y+1  n_z+1;
                 n_x_u+1  n_y+1  n_z+1]

    kv_l   = open_uniform_kv(n_x_l, 1)
    kv_u   = open_uniform_kv(n_x_u, 1)
    kv_y   = open_uniform_kv(n_y,   1)
    kv_z   = open_uniform_kv(n_z,   1)
    KV_ref = Vector{Vector{Vector{Float64}}}([
        [kv_l, kv_y, kv_z],
        [kv_u, kv_y, kv_z],
    ])

    epss_use = epss > 0.0 ? epss : 1.0e6

    # ── Connectivity ──────────────────────────────────────────────────────────
    nel, nnp, nen = patch_metrics(npc, npd, p_mat, n_mat_ref)
    IEN = build_ien(nsd, npd, npc, p_mat, n_mat_ref, nel, nnp, nen)
    INC = [build_inc(n_mat_ref[pc, :]) for pc in 1:npc]

    # ── Homogeneous Dirichlet BCs (bc_per_dof, ID = 0) ────────────────────────
    # ux = 0 on x=0 face (facet 4, both patches) — x-symmetry plane
    # uz = 0 on z=0 face (facet 1, both patches) — z-symmetry plane
    dBC = [1 4 2 1 2;    # ux = 0, facet 4 (ξ=1, x=0)
           3 1 2 1 2]    # uz = 0, facet 1 (ζ=1, z=0)

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

    # ── Non-homogeneous Dirichlet: uy = 0 at origin ───────────────────────────
    F   = zeros(neq)
    IND = Tuple{Int, Float64}[]
    tol = 1e-10

    for pc in 1:npc
        for loc_A in 1:nnp[pc]
            cp = P_ref[pc][loc_A]
            x  = B_ref[cp, 1];  y = B_ref[cp, 2];  z = B_ref[cp, 3]
            if abs(x) < tol && abs(y + l_y/2) < tol && abs(z) < tol
                eq = ID[2, cp]
                eq != 0 && push!(IND, (eq, 0.0))
            end
        end
    end
    unique!(IND)

    # ── Neumann BC: tx = 2*p_load*y/l_y at x = l_x/2 (facet 2) ─────────────
    traction_fn = (x, y, z) -> begin
        σ = zeros(3, 3); σ[1, 1] = 2*p_load * y / l_y; σ
    end
    for pc in 1:npc
        F = segment_load(n_mat_ref[pc, :], p_mat[pc, :], KV_ref[pc], P_ref[pc], B_ref,
                          nnp[pc], nen[pc], nsd, npd, ned, Int[], 2, ID, F, traction_fn, 1.0, NQUAD)
    end

    K_bc, F_bc = enforce_dirichlet(IND, K, F)

    # ── Mortar coupling ───────────────────────────────────────────────────────
    pairs_full = [InterfacePair(1, 3, 2, 5), InterfacePair(2, 5, 1, 3)]
    pairs_sp   = [InterfacePair(1, 3, 2, 5)]
    pairs = formulation isa SinglePassFormulation ? pairs_sp : pairs_full

    Pc = build_interface_cps(pairs, p_mat, n_mat_ref, KV_ref, P_ref, npd, nnp, formulation)
    C, Z = build_mortar_coupling(Pc, pairs, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
                                  ID, nnp, ned, nsd, npd, neq, NQUAD_mortar, epss_use,
                                  strategy, formulation, normal_strategy)

    fixed_eqs = Set(A for (A, _) in IND)
    rows_C, cols_C, vals_C = findnz(C)
    keep  = [i for i in eachindex(rows_C) if !(rows_C[i] in fixed_eqs)]
    C_bc  = sparse(rows_C[keep], cols_C[keep], vals_C[keep], size(C, 1), size(C, 2))

    # Remove inactive Lagrange multipliers (zero C columns, no-op for TM/DP).
    _, cols_C_nz, _ = findnz(C_bc)
    active_lm = sort(unique(cols_C_nz))
    if length(active_lm) < size(C_bc, 2)
        C_bc = C_bc[:, active_lm]
        Z    = Z[active_lm, active_lm]
    end

    # ── Solve ─────────────────────────────────────────────────────────────────
    U, _ = solve_mortar(K_bc, C_bc, Z, F_bc)

    # ── L2 displacement error ─────────────────────────────────────────────────
    disp_fn = (x, y, z) -> beam_exact_disp(x, y, z; p_load=p_load, E=E, l_x=l_x, l_y=l_y)
    err_abs, err_ref = l2_disp_error_beam(
        U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
        nen, nel, IEN, INC, NQUAD, disp_fn
    )

    if !isempty(vtk_prefix)
        write_vtk_beam(vtk_prefix, U, ID, npc, nsd, npd,
                       p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
                       nen, IEN, INC, E, nu; n_vis=n_vis)
    end

    return err_abs / err_ref, err_abs
end

# ─── Convergence study ─────────────────────────────────────────────────────────

"""
    run_convergence_beam(; p_ord, exp_range, formulation, strategy, kwargs...)

Run a convergence study for the 3D beam benchmark.
For p_ord = 1 uses `solve_beam_p1`; for p_ord ≥ 2 uses `solve_beam`.
Prints a table of (exp, err_abs, rate).
"""
function run_convergence_beam(;
    p_ord::Int                       = 1,
    exp_range                        = 0:4,
    formulation::FormulationStrategy = TwinMortarFormulation(),
    strategy::IntegrationStrategy    = ElementBasedIntegration(),
    kwargs...
)
    solver = (p_ord == 1) ?
        (exp) -> solve_beam_p1(exp; formulation=formulation, strategy=strategy, kwargs...) :
        (exp) -> solve_beam(p_ord, exp; formulation=formulation, strategy=strategy, kwargs...)

    fmt_s  = replace(split(string(typeof(formulation)), '.')[end], "Formulation" => "")
    strat_s = replace(split(string(typeof(strategy)),  '.')[end], "Integration"  => "")

    @printf "\n=== Bending beam: p=%d, %s, %s ===\n" p_ord fmt_s strat_s
    @printf "  exp |   err_abs   |  rate\n"
    @printf "  ----+------------+-------\n"

    err_prev = NaN
    for exp in exp_range
        err_rel, err_abs = solver(exp)
        rate = isnan(err_prev) || err_prev == 0.0 ? NaN : log(err_prev / err_abs) / log(2.0)
        @printf "   %2d |  %.4e  |  %5.2f\n" exp err_abs (isnan(rate) ? 0.0 : rate)
        err_prev = err_abs
    end
end

"""
    _beam_setup_p1(exp_level; ...) -> NamedTuple

Same as `_beam_setup` but for p=1 (direct-mesh geometry, no krefinement).
"""
function _beam_setup_p1(
    exp_level::Int;
    l_x::Float64  = 8.0, l_y::Float64 = 2.0, l_z::Float64 = 2.0,
    E::Float64    = 1000.0, nu::Float64 = 0.0, p_load::Float64 = 10.0,
    epss::Float64 = 0.0, NQUAD::Int = 2, NQUAD_mortar::Int = 3,
    n_x_lower_base::Int = 4, n_x_upper_base::Int = 3,
)
    nsd = 3;  npd = 3;  ned = 3;  npc = 2;  p_ord = 1

    n_x_u = n_x_upper_base * 2^exp_level
    n_x_l = n_x_lower_base  * 2^exp_level
    n_y   = max(1, 2^exp_level)
    n_z   = max(1, 2^exp_level)

    B_ref, P_ref = beam_geometry_p1(n_x_l, n_x_u, n_y, n_z; l_x=l_x, l_y=l_y, l_z=l_z)
    ncp = size(B_ref, 1)

    p_mat     = fill(p_ord, npc, npd)
    n_mat_ref = [n_x_l+1  n_y+1  n_z+1;
                 n_x_u+1  n_y+1  n_z+1]

    kv_l = open_uniform_kv(n_x_l, 1); kv_u = open_uniform_kv(n_x_u, 1)
    kv_y = open_uniform_kv(n_y,   1); kv_z = open_uniform_kv(n_z,   1)
    KV_ref = Vector{Vector{Vector{Float64}}}([
        [kv_l, kv_y, kv_z], [kv_u, kv_y, kv_z]])

    epss_use = epss > 0.0 ? epss : 1.0e6

    nel, nnp, nen = patch_metrics(npc, npd, p_mat, n_mat_ref)
    IEN = build_ien(nsd, npd, npc, p_mat, n_mat_ref, nel, nnp, nen)
    INC = [build_inc(n_mat_ref[pc, :]) for pc in 1:npc]

    dBC = [1 4 2 1 2; 3 1 2 1 2]
    bc_per_dof = dirichlet_bc_control_points(p_mat, n_mat_ref, KV_ref, P_ref,
                                              npd, nnp, ned, dBC)
    neq, ID = build_id(bc_per_dof, ned, ncp)
    LM      = build_lm(nen, ned, npc, nel, ID, IEN, P_ref)

    mats = [LinearElastic(E, nu, :three_d), LinearElastic(E, nu, :three_d)]
    Ub0  = zeros(ncp, nsd)
    K, _ = build_stiffness_matrix(npc, nsd, npd, ned, neq,
                                   p_mat, n_mat_ref, KV_ref, P_ref, B_ref, Ub0,
                                   nen, nel, IEN, INC, LM, mats, NQUAD, 1.0)

    F = zeros(neq); IND = Tuple{Int,Float64}[]
    tol = 1e-10
    for pc in 1:npc
        for loc_A in 1:nnp[pc]
            cp = P_ref[pc][loc_A]
            x = B_ref[cp,1]; y = B_ref[cp,2]; z = B_ref[cp,3]
            if abs(x) < tol && abs(y + l_y/2) < tol && abs(z) < tol
                eq = ID[2, cp]; eq != 0 && push!(IND, (eq, 0.0))
            end
        end
    end
    unique!(IND)

    traction_fn = (x, y, z) -> begin σ = zeros(3,3); σ[1,1] = 2*p_load*y/l_y; σ end
    for pc in 1:npc
        F = segment_load(n_mat_ref[pc,:], p_mat[pc,:], KV_ref[pc], P_ref[pc], B_ref,
                          nnp[pc], nen[pc], nsd, npd, ned,
                          Int[], 2, ID, F, traction_fn, 1.0, NQUAD)
    end
    K_bc, F_bc = enforce_dirichlet(IND, K, F)

    pairs_tm = [InterfacePair(1, 3, 2, 5), InterfacePair(2, 5, 1, 3)]
    pairs_sp = [InterfacePair(1, 3, 2, 5)]

    n_iface_slave  = n_x_l * n_z
    n_iface_master = n_x_u * n_z

    return (
        p_mat=p_mat, n_mat_ref=n_mat_ref, KV_ref=KV_ref, P_ref=P_ref, B_ref=B_ref,
        ID=ID, nnp=nnp, ned=ned, nsd=nsd, npd=npd, neq=neq, ncp=ncp,
        K_bc=K_bc, F_bc=F_bc, NQUAD=NQUAD, NQUAD_mortar=NQUAD_mortar,
        epss=epss_use, nen=nen, nel=nel, IEN=IEN, INC=INC, mats=mats,
        pairs_tm=pairs_tm, pairs_sp=pairs_sp,
        n_iface_slave=n_iface_slave, n_iface_master=n_iface_master,
    )
end

"""
    _beam_setup(p_ord, exp_level; ...) -> NamedTuple

Extract all arguments needed for `build_mortar_coupling` from the bending beam
benchmark at polynomial degree `p_ord` and refinement level `exp_level`, without
performing the linear solve.  Used by `run_cost_study_beam`.
"""
function _beam_setup(
    p_ord::Int, exp_level::Int;
    l_x::Float64  = 8.0, l_y::Float64 = 2.0, l_z::Float64 = 2.0,
    E::Float64    = 1000.0, nu::Float64 = 0.0, p_load::Float64 = 10.0,
    epss::Float64 = 0.0, NQUAD::Int = p_ord + 1, NQUAD_mortar::Int = p_ord + 2,
    n_x_lower_base::Int = 2, n_x_upper_base::Int = 1,
)
    nsd = 3;  npd = 3;  ned = 3;  npc = 2

    B0, P  = beam_geometry(p_ord; l_x=l_x, l_y=l_y, l_z=l_z)
    p_mat  = fill(p_ord, npc, npd)
    n_mat  = fill(p_ord + 1, npc, npd)
    KV     = generate_knot_vectors(npc, npd, p_mat, n_mat)

    B0_hack = copy(B0)
    B0_hack[P[1], 2] .+= 1000.0

    n_x  = n_x_upper_base * 2^exp_level
    n_xl = n_x_lower_base  * 2^exp_level
    n_y  = 2^exp_level
    n_z  = max(1, 2^exp_level)

    u_x_l = [i / n_xl for i in 1:n_xl - 1]
    u_x_u = [i / n_x  for i in 1:n_x  - 1]
    u_y   = [i / n_y  for i in 1:n_y  - 1]
    u_z   = [i / n_z  for i in 1:n_z  - 1]

    kref_data = Vector{Float64}[
        vcat([1.0, 1.0], u_x_l), vcat([1.0, 2.0], u_y), vcat([1.0, 3.0], u_z),
        vcat([2.0, 1.0], u_x_u), vcat([2.0, 2.0], u_y), vcat([2.0, 3.0], u_z),
    ]
    n_mat_ref, _, KV_ref, B_ref_hack, P_ref = krefinement(
        nsd, npd, npc, n_mat, p_mat, KV, B0_hack, P, kref_data)
    B_ref = copy(B_ref_hack)
    for i in axes(B_ref, 1)
        B_ref[i, 2] > 100.0 && (B_ref[i, 2] -= 1000.0)
    end
    ncp = size(B_ref, 1)

    epss_use = epss > 0.0 ? epss : 1.0e6

    nel, nnp, nen = patch_metrics(npc, npd, p_mat, n_mat_ref)
    IEN = build_ien(nsd, npd, npc, p_mat, n_mat_ref, nel, nnp, nen)
    INC = [build_inc(n_mat_ref[pc, :]) for pc in 1:npc]

    dBC = [1 4 2 1 2; 3 1 2 1 2]
    bc_per_dof = dirichlet_bc_control_points(p_mat, n_mat_ref, KV_ref, P_ref,
                                              npd, nnp, ned, dBC)
    neq, ID = build_id(bc_per_dof, ned, ncp)
    LM      = build_lm(nen, ned, npc, nel, ID, IEN, P_ref)

    mats = [LinearElastic(E, nu, :three_d), LinearElastic(E, nu, :three_d)]
    Ub0  = zeros(ncp, nsd)
    K, _ = build_stiffness_matrix(npc, nsd, npd, ned, neq,
                                   p_mat, n_mat_ref, KV_ref, P_ref, B_ref, Ub0,
                                   nen, nel, IEN, INC, LM, mats, NQUAD, 1.0)

    F    = zeros(neq)
    IND  = Tuple{Int, Float64}[]
    tol  = 1e-10
    for pc in 1:npc
        for loc_A in 1:nnp[pc]
            cp = P_ref[pc][loc_A]
            x  = B_ref[cp, 1];  y = B_ref[cp, 2];  z = B_ref[cp, 3]
            if abs(x) < tol && abs(y + l_y/2) < tol && abs(z) < tol
                eq = ID[2, cp]; eq != 0 && push!(IND, (eq, 0.0))
            end
        end
    end
    unique!(IND)

    traction_fn = (x, y, z) -> begin σ = zeros(3,3); σ[1,1] = 2*p_load*y/l_y; σ end
    for pc in 1:npc
        F = segment_load(n_mat_ref[pc, :], p_mat[pc, :], KV_ref[pc], P_ref[pc], B_ref,
                          nnp[pc], nen[pc], nsd, npd, ned,
                          Int[], 2, ID, F, traction_fn, 1.0, NQUAD)
    end
    K_bc, F_bc = enforce_dirichlet(IND, K, F)

    pairs_tm = [InterfacePair(1, 3, 2, 5), InterfacePair(2, 5, 1, 3)]
    pairs_sp = [InterfacePair(1, 3, 2, 5)]

    # Count interface elements on each side
    # Interface: facet 3 of Patch 1 (η=n₂) → surface element count = n_elem_ξ × n_elem_ζ
    n_elem_ξ1 = n_mat_ref[1, 1] - p_ord   # slave  (Patch 1) interface elements in ξ
    n_elem_ξ2 = n_mat_ref[2, 1] - p_ord   # master (Patch 2) interface elements in ξ
    n_elem_ζ  = n_mat_ref[1, 3] - p_ord   # elements in ζ (same for both patches)
    n_iface_slave  = n_elem_ξ1 * n_elem_ζ
    n_iface_master = n_elem_ξ2 * n_elem_ζ

    return (
        p_mat=p_mat, n_mat_ref=n_mat_ref, KV_ref=KV_ref, P_ref=P_ref, B_ref=B_ref,
        ID=ID, nnp=nnp, ned=ned, nsd=nsd, npd=npd, neq=neq, ncp=ncp,
        K_bc=K_bc, F_bc=F_bc, NQUAD=NQUAD, NQUAD_mortar=NQUAD_mortar,
        epss=epss_use, nen=nen, nel=nel, IEN=IEN, INC=INC, mats=mats,
        pairs_tm=pairs_tm, pairs_sp=pairs_sp,
        n_iface_slave=n_iface_slave, n_iface_master=n_iface_master,
    )
end

"""
    run_cost_study_beam(; degrees, exp_levels, n_repeats, epss)

Time `build_mortar_coupling` for TM-Elm, TM-Seg, SP-Elm, SP-Seg on the 3D bending
beam benchmark.  Prints a table suitable for §6.6 of the paper showing the wall-clock
overhead of segment-based integration relative to element-based.
"""
function run_cost_study_beam(;
    degrees::Vector{Int}    = [1, 2],
    exp_levels::Vector{Int} = [1, 2],
    n_repeats::Int          = 5,
    epss::Float64           = 1e6,
    kwargs...
)
    configs = [
        ("TM-Elm", TwinMortarFormulation(), ElementBasedIntegration()),
        ("TM-Seg", TwinMortarFormulation(), SegmentBasedIntegration()),
        ("SP-Elm", SinglePassFormulation(), ElementBasedIntegration()),
        ("SP-Seg", SinglePassFormulation(), SegmentBasedIntegration()),
    ]

    hdr = @sprintf("%-4s  %-3s  %-10s  %-9s  %-9s  %-7s  %-10s",
                   "p", "exp", "n_iface_s", "method", "n_cells", "t(ms)", "t/t_TM-Elm")
    println("\n=== Interface assembly cost study (3D bending beam) ===")
    println(hdr)
    println("-"^length(hdr))

    for p_ord in degrees
        NQUAD_mortar = p_ord + 2
        for exp in exp_levels
            d = if p_ord == 1
                _beam_setup_p1(exp; epss=epss, NQUAD_mortar=NQUAD_mortar, kwargs...)
            else
                _beam_setup(p_ord, exp; epss=epss, NQUAD_mortar=NQUAD_mortar, kwargs...)
            end

            # Reference TM-Elm time (JIT + timing)
            Pc_ref = build_interface_cps(d.pairs_tm, d.p_mat, d.n_mat_ref,
                                         d.KV_ref, d.P_ref, d.npd, d.nnp,
                                         TwinMortarFormulation())
            # warm-up
            build_mortar_coupling(Pc_ref, d.pairs_tm, d.p_mat, d.n_mat_ref,
                d.KV_ref, d.P_ref, d.B_ref, d.ID, d.nnp, d.ned, d.nsd, d.npd, d.neq,
                NQUAD_mortar, d.epss, ElementBasedIntegration(), TwinMortarFormulation())
            t_elm = minimum(
                @elapsed(build_mortar_coupling(Pc_ref, d.pairs_tm, d.p_mat, d.n_mat_ref,
                    d.KV_ref, d.P_ref, d.B_ref, d.ID, d.nnp, d.ned, d.nsd, d.npd, d.neq,
                    NQUAD_mortar, d.epss, ElementBasedIntegration(), TwinMortarFormulation()))
                for _ in 1:n_repeats)

            for (label, form, strat) in configs
                pairs = form isa TwinMortarFormulation ? d.pairs_tm : d.pairs_sp
                Pc = build_interface_cps(pairs, d.p_mat, d.n_mat_ref,
                                          d.KV_ref, d.P_ref, d.npd, d.nnp, form)

                # Count integration cells / QPs
                n_cells = if strat isa ElementBasedIntegration
                    n_passes = form isa TwinMortarFormulation ? 2 : 1
                    n_passes * d.n_iface_slave * NQUAD_mortar^2
                else
                    # Segment-based: extract boundary surface data for each facet
                    # then count triangle cells produced by Sutherland-Hodgman clipping
                    ps_s, ns_s, KVs_s, Ps_s, _, _, _, _, _, _, _ = get_segment_patch(
                        d.p_mat[1,:], d.n_mat_ref[1,:], d.KV_ref[1], d.P_ref[1],
                        d.npd, d.nnp[1], 3)   # slave = Patch 1 facet 3
                    ps_m, ns_m, KVm_m, Pm_m, _, _, _, _, _, _, _ = get_segment_patch(
                        d.p_mat[2,:], d.n_mat_ref[2,:], d.KV_ref[2], d.P_ref[2],
                        d.npd, d.nnp[2], 5)   # master = Patch 2 facet 5
                    cells1 = find_interface_segments_2d(
                        ps_s, ns_s, KVs_s, ps_m, ns_m, KVm_m,
                        d.B_ref, Ps_s, Pm_m, d.nsd)
                    n_tri_pts = NQUAD_mortar <= 1 ? 1 : NQUAD_mortar <= 3 ? 3 : 7
                    n_passes  = form isa TwinMortarFormulation ? 2 : 1
                    n_passes * length(cells1) * n_tri_pts
                end

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

                ratio = label == "TM-Elm" ? 1.0 : t_min / t_elm
                @printf("%-4d  %-3d  %-10d  %-9s  %-9d  %-7.2f  %-10.2f\n",
                        p_ord, exp, d.n_iface_slave, label, n_cells,
                        t_min*1000, ratio)
            end
            println()
        end
    end
end

function run_all_formulations_beam(;
    p_ord::Int = 1,
    exp_range  = 0:4,
    kwargs...
)
    configs = [
        (TwinMortarFormulation(), ElementBasedIntegration(), "TM-Elm"),
        (TwinMortarFormulation(), SegmentBasedIntegration(), "TM-Seg"),
        (DualPassFormulation(),   ElementBasedIntegration(), "DP-Elm"),
        (DualPassFormulation(),   SegmentBasedIntegration(), "DP-Seg"),
        (SinglePassFormulation(), ElementBasedIntegration(), "SP-Elm"),
        (SinglePassFormulation(), SegmentBasedIntegration(), "SP-Seg"),
    ]

    labels = [c[3] for c in configs]
    @printf "\n=== Bending beam p=%d: L2 displacement error ===\n" p_ord
    print("  exp |")
    for lb in labels; @printf " %-11s|" lb; end; println()
    print("  ----|")
    for _ in labels; print("------------|"); end; println()

    results = [Float64[] for _ in configs]
    for exp in exp_range
        row = Float64[]
        for (k, (form, strat, _)) in enumerate(configs)
            if p_ord == 1
                _, err_abs = solve_beam_p1(exp; formulation=form, strategy=strat, kwargs...)
            else
                _, err_abs = solve_beam(p_ord, exp; formulation=form, strategy=strat, kwargs...)
            end
            push!(row, err_abs)
            push!(results[k], err_abs)
        end
        @printf "   %2d |" exp
        for e in row; @printf " %.4e  |" e; end
        println()
    end

    # Print convergence rates (last two levels)
    print("  rate|")
    for k in 1:length(configs)
        errs = results[k]
        if length(errs) >= 2 && errs[end] > 0 && errs[end-1] > 0
            rate = log(errs[end-1] / errs[end]) / log(2.0)
            @printf "   %+5.2f     |" rate
        else
            @printf "    ---      |"
        end
    end
    println()
end
