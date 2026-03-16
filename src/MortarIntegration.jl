# MortarIntegration.jl
# Integration strategy types and segment-finding utilities for Twin Mortar coupling.
#
# Provides the Strategy design pattern for mortar integration:
#   ElementBasedIntegration  — Gauss points per slave element, projected onto master
#   SegmentBasedIntegration  — integration over slave/master knot-span intersections

"""
    IntegrationStrategy

Abstract type for the mortar numerical integration method used in
`build_mortar_coupling`.  Concrete subtypes select between the classical
element-based approach and the more accurate segment-based approach.
"""
abstract type IntegrationStrategy end

"""
    ElementBasedIntegration()

Gauss points are placed at standard quadrature points within each slave
boundary element.  Each Gauss point is projected onto the master boundary
via Newton closest-point iteration (`closest_point_1d`).

This is the classical *Gauss-point-to-segment* approach.  It is the
default strategy and is sufficient for most non-conforming problems.
"""
struct ElementBasedIntegration <: IntegrationStrategy end

"""
    SegmentBasedIntegration()

Integration segments are formed by intersecting slave and master
knot-span boundaries in slave parametric space (Puso & Laursen 2004,
adapted for NURBS).  Gauss points are distributed within each
sub-segment, guaranteeing that both slave and master NURBS shape
functions are smooth (single Bezier span) within every integration
cell.  This enables exact polynomial quadrature and eliminates
integration errors caused by knot-span mismatches across the interface.
"""
struct SegmentBasedIntegration <: IntegrationStrategy end

# ─── Formulation strategy ─────────────────────────────────────────────────────

"""
    FormulationStrategy

Abstract type selecting the mortar constraint formulation used in
`build_mortar_coupling`.  Controls both the multiplier space and the
algebraic structure of the assembled (C, Z) matrices.
"""
abstract type FormulationStrategy end

"""
    TwinMortarFormulation()

Default (symmetric dual-pass) formulation.  Lagrange multipliers reside on
**both** slave and master surfaces; the stabilization matrix Z ≠ 0.
Requires `pairs` to contain two `InterfacePair`s with slave/master roles
swapped.
"""
struct TwinMortarFormulation <: FormulationStrategy end

"""
    SinglePassFormulation()

Standard single-pass mortar method.  Lagrange multipliers reside on the
**slave surface only**; Z = 0 (hard Lagrange constraint, no stabilization).
Supply **one** `InterfacePair` (slave→master).  The cross-coupling matrix
M^(sm) is assembled so that the constraint C^⊤u = 0 enforces
D^(s)·u^s ≈ M^(sm)·u^m across the interface.
"""
struct SinglePassFormulation <: FormulationStrategy end

"""
    DualPassFormulation()

Dual-pass mortar formulation following Puso & Solberg (2020).  Like
`TwinMortarFormulation`, multipliers reside on **both** surfaces and Z ≠ 0;
requires two `InterfacePair`s with swapped roles.

The difference from Twin Mortar is in the Z assembly: each half-pass
contributes **only its slave-side rows** to Z (with full factor ε instead of
ε/2), leaving the off-diagonal Z blocks unsymmetrised for non-conforming
meshes assembled with element-based integration.  For segment-based
integration (symmetric intersection segments) the result is identical to
`TwinMortarFormulation`.
"""
struct DualPassFormulation <: FormulationStrategy end

# ─── Normal (constraint direction) strategy ──────────────────────────────────
#
# Note: The Lagrange multiplier λ is now resolved in global Cartesian directions
# (e1, e2, e3), so the NormalStrategy does NOT affect dir_vecs (always I).
# The types are retained for segment-based integration where the projection
# plane may depend on the normal choice.

"""
    NormalStrategy

Abstract type selecting how the interface normal direction is computed.

Currently retained for segment-based integration where the projection plane
may depend on the normal choice.  The constraint direction matrix `dir_vecs`
is always the identity (global Cartesian).
"""
abstract type NormalStrategy end

"""
    SlaveNormal()

Use the outward unit normal of the **slave** surface.
This is the default and corresponds to the classical Gauss-point-to-segment
(element-based) integration strategy.
"""
struct SlaveNormal  <: NormalStrategy end

"""
    MasterNormal()

Use the outward unit normal of the **master** surface, negated so that it points
in the same direction as the slave outward normal (toward the master side).
"""
struct MasterNormal <: NormalStrategy end

"""
    AverageNormal()

Use the bisector of slave and master outward normals:
  n̄ = normalize(n_s − n_m_outward)
This is the projection-plane strategy used by Puso & Solberg (2020) for the
symmetric dual-pass mortar method.
"""
struct AverageNormal <: NormalStrategy end

"""
    find_interface_segments_1d(ps_s, ns_s, kv_s, ps_m, ns_m, kv_m,
                                B, Ps, Pm, nsd) -> Vector{Float64}

Find all integration-segment breakpoints in **slave** parametric space
for a non-conforming 1D mortar interface.

The result is the sorted union of:
- slave knot-span boundaries (already in ξ_s ∈ [0, 1])
- master knot-span boundaries projected onto the slave curve via
  Newton closest-point iteration

Integrating between consecutive breakpoints guarantees that both slave
and master NURBS shape functions are smooth (single Bezier span) within
each integration cell.

Arguments
- `ps_s, ns_s, kv_s`: slave boundary degree, #CPs, and 1D knot vector
- `ps_m, ns_m, kv_m`: master boundary degree, #CPs, and 1D knot vector
- `B`: global control point array (ncp × (nsd+1)); last column is weight
- `Ps, Pm`: slave/master boundary local → global CP index maps
- `nsd`: number of spatial dimensions
"""
# ─── 3D segment-based integration utilities ──────────────────────────────────

"""
    SegmentCell2D

One triangular integration cell produced by intersecting a slave NURBS surface
element with a master NURBS surface element (Sutherland-Hodgman polygon clipping
in physical space).  Used in `find_interface_segments_2d`.

Fields
------
- `verts`  : (nsd × 3) physical coordinates of the three triangle vertices (columns)
- `ξ0_s, η0_s` : slave element centroid in parametric space (initial guess for
                  `closest_point_2d` at each quadrature point)
- `ξ0_m, η0_m` : master element centroid in parametric space (initial guess)
"""
struct SegmentCell2D
    verts  :: Matrix{Float64}   # nsd × 3
    ξ0_s   :: Float64
    η0_s   :: Float64
    ξ0_m   :: Float64
    η0_m   :: Float64
end

"""
    sutherland_hodgman_clip(subject_verts, clip_verts, normal; tol=1e-12)
        -> Vector{Vector{Float64}}

Clip a convex polygon (`subject_verts`, list of nsd-vectors) against a convex
polygon (`clip_verts`, list of nsd-vectors) using the Sutherland-Hodgman
algorithm.  Both polygons are assumed to lie in the plane defined by `normal`.

Returns the vertices of the intersection polygon (possibly empty).

Reference: Puso & Laursen (2004) Appendix A.
"""
function sutherland_hodgman_clip(
    subject_verts :: Vector{<:AbstractVector},
    clip_verts    :: Vector{<:AbstractVector},
    normal        :: AbstractVector;
    tol           :: Float64 = 1e-12
)
    output = [copy(v) for v in subject_verts]
    nclip  = length(clip_verts)

    for i in 1:nclip
        isempty(output) && break
        input  = output
        output = Vector{Vector{Float64}}()

        A = clip_verts[i]
        B = clip_verts[mod1(i + 1, nclip)]

        edge_dir   = B .- A
        edge_n     = cross(normal, edge_dir)
        edge_n_len = norm(edge_n)
        edge_n_len < tol && continue
        edge_n = edge_n ./ edge_n_len

        prev         = input[end]
        prev_inside  = dot(prev .- A, edge_n) >= -tol

        for curr in input
            curr_inside = dot(curr .- A, edge_n) >= -tol
            if curr_inside
                if !prev_inside
                    push!(output, _sh_intersect(prev, curr, A, edge_n, tol))
                end
                push!(output, copy(curr))
            elseif prev_inside
                push!(output, _sh_intersect(prev, curr, A, edge_n, tol))
            end
            prev        = curr
            prev_inside = curr_inside
        end
    end
    return output
end

function _sh_intersect(P, Q, A, edge_n, tol)
    d_P   = dot(P .- A, edge_n)
    d_Q   = dot(Q .- A, edge_n)
    denom = d_P - d_Q
    abs(denom) < tol && return copy(P)
    t = d_P / denom
    return P .+ t .* (Q .- P)
end

"""
    triangulate_polygon(verts) -> Vector{NTuple{3,Vector{Float64}}}

Fan-triangulate a convex polygon from its centroid.
Returns a list of (v1, v2, v3) vertex triples.
"""
function triangulate_polygon(verts::Vector{<:AbstractVector})
    n = length(verts)
    n < 3 && return Tuple{Vector{Float64},Vector{Float64},Vector{Float64}}[]
    xc   = sum(verts) ./ n
    tris = Tuple{Vector{Float64},Vector{Float64},Vector{Float64}}[]
    for i in 1:n
        j = mod1(i + 1, n)
        push!(tris, (copy(verts[i]), copy(verts[j]), copy(xc)))
    end
    return tris
end

"""
    tri_gauss_rule(nquad) -> (points::Matrix{Float64}, weights::Vector{Float64})

Return a Gauss quadrature rule for the standard reference triangle with vertices
(0,0), (1,0), (0,1).  Points are returned as a (2 × npts) matrix; weights sum to
0.5 (the area of the reference triangle).

The `nquad` argument selects the number of 1-D points used in analogous tensor-
product rules and maps to triangle rules as follows:
  - nquad = 1       → 1-point  rule (degree 1)
  - nquad = 2 or 3  → 3-point  rule (degree 2)
  - nquad ≥ 4       → 7-point  rule (degree 5, Cowper 1973)

Integration over a physical triangle with area `J`:

    ∫_phys f dA  ≈  Σ_q  weights[q] * 2J * f(x_q)
"""
function tri_gauss_rule(nquad::Int)
    if nquad <= 1
        pts = [1/3  1/3]'
        w   = [0.5]
    elseif nquad <= 3
        a   = 1/6;  b = 2/3
        pts = [a a; b a; a b]'
        w   = [1/6, 1/6, 1/6]
    else
        # Cowper (1973) degree-5 rule, 7 points
        a1  = 0.1012865073235;  b1 = 0.7974269853531
        a2  = 0.4701420641051;  b2 = 0.0597158717898
        pts = [1/3   1/3;
               a1    a1;
               b1    a1;
               a1    b1;
               a2    b2;
               b2    a2;
               a2    a2]'
        w   = [0.1125,
               0.0629695902724, 0.0629695902724, 0.0629695902724,
               0.0661970763943, 0.0661970763943, 0.0661970763943]
    end
    return pts, w
end

"""
    find_interface_segments_2d(ps, ns, KVs, pm, nm, KVm, B, Ps, Pm, nsd;
                                min_area_tol) -> Vector{SegmentCell2D}

Find all triangular integration cells covering the intersection of slave and master
NURBS surface elements on a non-conforming 3D mortar interface.

For every (slave element, master element) pair, the algorithm:
1. Evaluates the 4 physical corner points of each element via `eval_surface_point`.
2. Projects slave corners onto the master element plane.
3. Clips the projected slave quad against the master quad (Sutherland-Hodgman).
4. Fan-triangulates the intersection polygon from its centroid.
5. Discards triangles with physical area < `min_area_tol`.

The returned `SegmentCell2D` objects carry the triangle vertices and slave/master
element parametric centroids as initial guesses for `closest_point_2d`.

Arguments
---------
- `ps, ns, KVs, Ps` : slave surface NURBS data (degrees, #CPs, knot vectors, CP map)
- `pm, nm, KVm, Pm` : master surface NURBS data
- `B`   : global control-point array (ncp × (nsd+1)); last column is weight
- `nsd` : spatial dimension (must be 3)
- `min_area_tol` : triangles with area below this threshold are discarded
"""
function find_interface_segments_2d(
    ps :: AbstractVector{Int}, ns :: AbstractVector{Int},
    KVs :: AbstractVector{<:AbstractVector{Float64}},
    pm :: AbstractVector{Int}, nm :: AbstractVector{Int},
    KVm :: AbstractVector{<:AbstractVector{Float64}},
    B :: AbstractMatrix{Float64},
    Ps :: AbstractVector{Int}, Pm :: AbstractVector{Int},
    nsd :: Int;
    min_area_tol :: Float64 = 1e-20
) :: Vector{SegmentCell2D}

    ξ_breaks_s = unique(sort(KVs[1]))
    η_breaks_s = unique(sort(KVs[2]))
    ξ_breaks_m = unique(sort(KVm[1]))
    η_breaks_m = unique(sort(KVm[2]))

    n_eξ_s = length(ξ_breaks_s) - 1
    n_eη_s = length(η_breaks_s) - 1
    n_eξ_m = length(ξ_breaks_m) - 1
    n_eη_m = length(η_breaks_m) - 1

    # Pre-evaluate all slave and master element corner physical positions
    # slave_corners[ie, je] = Vector of 4 physical points (CCW: SW, SE, NE, NW)
    slave_corners = Matrix{Vector{Vector{Float64}}}(undef, n_eξ_s, n_eη_s)
    slave_ξ0      = Vector{Float64}(undef, n_eξ_s)
    slave_η0      = Vector{Float64}(undef, n_eη_s)
    for ie in 1:n_eξ_s
        ξ_a = ξ_breaks_s[ie];  ξ_b = ξ_breaks_s[ie+1]
        slave_ξ0[ie] = 0.5 * (ξ_a + ξ_b)
        for je in 1:n_eη_s
            η_a = η_breaks_s[je];  η_b = η_breaks_s[je+1]
            if ie == 1
                slave_η0[je] = 0.5 * (η_a + η_b)
            end
            xa, _, _, _, _ = eval_surface_point(ξ_a, η_a, ps, ns, KVs, B, Ps, nsd)
            xb, _, _, _, _ = eval_surface_point(ξ_b, η_a, ps, ns, KVs, B, Ps, nsd)
            xc, _, _, _, _ = eval_surface_point(ξ_b, η_b, ps, ns, KVs, B, Ps, nsd)
            xd, _, _, _, _ = eval_surface_point(ξ_a, η_b, ps, ns, KVs, B, Ps, nsd)
            slave_corners[ie, je] = [xa, xb, xc, xd]
        end
    end

    master_corners = Matrix{Vector{Vector{Float64}}}(undef, n_eξ_m, n_eη_m)
    master_normals = Matrix{Vector{Float64}}(undef, n_eξ_m, n_eη_m)
    master_ξ0      = Vector{Float64}(undef, n_eξ_m)
    master_η0      = Vector{Float64}(undef, n_eη_m)
    for ie in 1:n_eξ_m
        ξ_a = ξ_breaks_m[ie];  ξ_b = ξ_breaks_m[ie+1]
        master_ξ0[ie] = 0.5 * (ξ_a + ξ_b)
        for je in 1:n_eη_m
            η_a = η_breaks_m[je];  η_b = η_breaks_m[je+1]
            if ie == 1
                master_η0[je] = 0.5 * (η_a + η_b)
            end
            xa, _, _, _, _ = eval_surface_point(ξ_a, η_a, pm, nm, KVm, B, Pm, nsd)
            xb, _, _, _, _ = eval_surface_point(ξ_b, η_a, pm, nm, KVm, B, Pm, nsd)
            xc, _, _, _, _ = eval_surface_point(ξ_b, η_b, pm, nm, KVm, B, Pm, nsd)
            xd, _, _, _, _ = eval_surface_point(ξ_a, η_b, pm, nm, KVm, B, Pm, nsd)
            master_corners[ie, je] = [xa, xb, xc, xd]
            # Plane normal from two diagonals (more robust than single edge cross)
            t1   = xc .- xa
            t2   = xd .- xb
            n_m  = cross(t1, t2)
            nlen = norm(n_m)
            master_normals[ie, je] = nlen > 1e-15 ? n_m ./ nlen : zeros(nsd)
        end
    end

    cells = SegmentCell2D[]

    for ie_s in 1:n_eξ_s, je_s in 1:n_eη_s
        sc  = slave_corners[ie_s, je_s]
        ξ0s = slave_ξ0[ie_s]
        η0s = slave_η0[je_s]

        for ie_m in 1:n_eξ_m, je_m in 1:n_eη_m
            mc   = master_corners[ie_m, je_m]
            n_m  = master_normals[ie_m, je_m]
            norm(n_m) < 1e-15 && continue

            ξ0m = master_ξ0[ie_m]
            η0m = master_η0[je_m]

            # Project slave corners onto master element plane
            x0_m    = mc[1]
            slave_proj = [v .- dot(v .- x0_m, n_m) .* n_m for v in sc]

            # Sutherland-Hodgman: clip projected slave quad against master quad
            poly = sutherland_hodgman_clip(slave_proj, mc, n_m)
            length(poly) < 3 && continue

            # Fan-triangulate and store cells
            for (v1, v2, v3) in triangulate_polygon(poly)
                area = norm(cross(v2 .- v1, v3 .- v1)) / 2.0
                area < min_area_tol && continue
                push!(cells, SegmentCell2D(hcat(v1, v2, v3), ξ0s, η0s, ξ0m, η0m))
            end
        end
    end

    return cells
end

function find_interface_segments_1d(
    ps_s::Int, ns_s::Int, kv_s::AbstractVector{Float64},
    ps_m::Int, ns_m::Int, kv_m::AbstractVector{Float64},
    B::AbstractMatrix{Float64},
    Ps::AbstractVector{Int}, Pm::AbstractVector{Int},
    nsd::Int
)::Vector{Float64}

    # 1. Slave breakpoints: all unique knot values (includes 0 and 1)
    ξ_breaks = unique(sort(kv_s))

    # 2. Project each master knot-span boundary onto the slave curve
    for ξ_m_break in unique(sort(kv_m))
        # Physical point on master boundary at this breakpoint
        x_m, _, _, _ = eval_boundary_point(ξ_m_break, ps_m, ns_m, kv_m, B, Pm, nsd)

        # Closest-point projection onto slave curve
        ξ_s0 = clamp(ξ_m_break, 0.0, 1.0)   # initial guess: same parameter
        ξ_s, _, _, _, _ = closest_point_1d(ξ_s0, x_m, ps_s, ns_s, kv_s, B, Ps, nsd)

        # Only add if projection lands inside the slave domain
        ξ_s < -1e-10 && continue
        ξ_s > 1.0 + 1e-10 && continue
        push!(ξ_breaks, clamp(ξ_s, 0.0, 1.0))
    end

    # 3. Sort and remove near-duplicates (tolerance 1e-12)
    sort!(ξ_breaks)
    result = [ξ_breaks[1]]
    for ξ in ξ_breaks[2:end]
        ξ - result[end] > 1e-12 && push!(result, ξ)
    end
    return result
end
