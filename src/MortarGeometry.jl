# MortarGeometry.jl
# Boundary geometry evaluation and closest-point projection for mortar coupling.

"""
    eval_boundary_point(ξ, ps, ns, kv, B, Ps, nsd) -> (x, dxdξ, R, span)

Evaluate physical position, tangent vector, and NURBS shape functions on a
1D NURBS boundary at global parametric coordinate ξ ∈ [0, 1].

- `ps`, `ns`, `kv`: degree, #CPs, and knot vector of the 1D boundary
- `B`: global control point array (ncp × (nsd+1)), last column is weight
- `Ps`: boundary local CP index → global B row index (length ns)

Returns:
- `x`: physical position (nsd,)
- `dxdξ`: tangent vector dx/dξ in physical space (nsd,)
- `R`: NURBS shape functions (ps+1,) at the non-zero span
- `span`: knot span index (for recovering which CPs are active)
"""
function eval_boundary_point(
    ξ::Float64,
    ps::Int, ns::Int, kv::AbstractVector{Float64},
    B::AbstractMatrix{Float64}, Ps::AbstractVector{Int},
    nsd::Int
)
    ξ = clamp(ξ, kv[ps+1], kv[ns+1])
    span = find_span(ns - 1, ps, ξ, kv)
    dN   = bspline_basis_and_deriv(span, ξ, ps, 1, kv)   # (2, ps+1)

    x_num    = zeros(nsd)
    dxdξ_num = zeros(nsd)
    W  = 0.0
    dW = 0.0
    R  = zeros(ps + 1)

    for a in 1:(ps + 1)
        local_idx  = span - ps + a - 1   # 1-based index into Ps
        global_cp  = Ps[local_idx]
        w          = B[global_cp, end]
        Xcp        = B[global_cp, 1:nsd]
        N          = dN[1, a]
        N1         = dN[2, a]
        x_num    .+= N  * w .* Xcp
        dxdξ_num .+= N1 * w .* Xcp
        W        += N  * w
        dW       += N1 * w
        R[a]      = N  * w
    end

    x_phys    = x_num / W
    dxdξ_phys = (dxdξ_num .- (dW / W) .* x_num) / W
    R        ./= W   # normalize to NURBS basis functions

    return x_phys, dxdξ_phys, R, span
end

"""
    eval_surface_point(ξ, η, ps, ns, kv, B, Ps, nsd) -> (x, dxdξ, dxdη, R, spans)

Evaluate physical position, partial tangents, and NURBS shape functions on a
2D NURBS boundary surface at parametric coordinates (ξ, η) ∈ [0,1]².

- `ps[2]`, `ns[2]`, `kv[2]`: degrees, #CPs, and knot vectors per parametric direction
- `B`: global control point array (ncp × (nsd+1)), last column is weight
- `Ps`: surface local CP index → global B row (length ns[1]*ns[2]);
  ordering ξ-inner (fast), η-outer (slow): Ps[(η-1)*ns[1] + ξ]

Returns:
- `x`: physical position (nsd,)
- `dxdξ`, `dxdη`: partial tangents (nsd,)
- `R`: NURBS shape functions ((ps[1]+1)*(ps[2]+1),), ξ-inner ordering
- `spans`: knot span indices [span_ξ, span_η]
"""
function eval_surface_point(
    ξ::Float64, η::Float64,
    ps::AbstractVector{Int}, ns::AbstractVector{Int},
    kv::AbstractVector{<:AbstractVector{Float64}},
    B::AbstractMatrix{Float64}, Ps::AbstractVector{Int},
    nsd::Int
)
    ξ = clamp(ξ, kv[1][ps[1]+1], kv[1][ns[1]+1])
    η = clamp(η, kv[2][ps[2]+1], kv[2][ns[2]+1])

    span_ξ = find_span(ns[1]-1, ps[1], ξ, kv[1])
    span_η = find_span(ns[2]-1, ps[2], η, kv[2])
    dNξ    = bspline_basis_and_deriv(span_ξ, ξ, ps[1], 1, kv[1])  # 2×(ps[1]+1)
    dNη    = bspline_basis_and_deriv(span_η, η, ps[2], 1, kv[2])  # 2×(ps[2]+1)

    x_num    = zeros(nsd)
    dxdξ_num = zeros(nsd)
    dxdη_num = zeros(nsd)
    W = 0.0; dWdξ = 0.0; dWdη = 0.0

    nsen = (ps[1]+1) * (ps[2]+1)
    R    = zeros(nsen)
    idx  = 0

    for b in 1:(ps[2]+1)
        η_loc  = span_η - ps[2] + b - 1      # 1-based index into ns[2]
        Nη     = dNη[1, b]
        dNη_b  = dNη[2, b]
        for a in 1:(ps[1]+1)
            ξ_loc    = span_ξ - ps[1] + a - 1   # 1-based index into ns[1]
            global_cp = Ps[(η_loc - 1)*ns[1] + ξ_loc]
            w        = B[global_cp, end]
            Xcp      = B[global_cp, 1:nsd]
            Nξ       = dNξ[1, a]
            dNξ_a    = dNξ[2, a]
            N_w      = Nξ * Nη * w
            idx += 1
            R[idx]    = N_w
            x_num    .+= N_w .* Xcp
            dxdξ_num .+= dNξ_a * Nη * w .* Xcp
            dxdη_num .+= Nξ * dNη_b * w .* Xcp
            W    += N_w
            dWdξ += dNξ_a * Nη * w
            dWdη += Nξ * dNη_b * w
        end
    end

    x    = x_num ./ W
    dxdξ = (dxdξ_num .- dWdξ .* x) ./ W
    dxdη = (dxdη_num .- dWdη .* x) ./ W
    R  ./= W

    return x, dxdξ, dxdη, R, [span_ξ, span_η]
end

"""
    closest_point_2d(ξ0, η0, x_s, ps_m, ns_m, kvm, B, Pm, nsd;
                     tol=1e-13, max_iter=100)
        -> (ξ_m, η_m, x_m, dxdξ, dxdη, R_m, spans_m)

Newton-Raphson closest-point projection of point `x_s` onto a 2D NURBS surface
in nsd-dimensional space.

Uses the full 2×2 Hessian H_ij = (∂x_m/∂ξ_i)·(∂x_m/∂ξ_j) (ignores curvature).
Clamps (ξ, η) to [0,1]² after each step.

Returns converged parametric coordinates, physical position, tangents,
NURBS shape functions, and knot span indices.
"""
function closest_point_2d(
    ξ0::Float64, η0::Float64, x_s::AbstractVector{Float64},
    ps_m::AbstractVector{Int}, ns_m::AbstractVector{Int},
    kvm::AbstractVector{<:AbstractVector{Float64}},
    B::AbstractMatrix{Float64}, Pm::AbstractVector{Int},
    nsd::Int;
    tol::Float64  = 1e-13,
    max_iter::Int = 100
)
    ξ = clamp(ξ0, 0.0, 1.0)
    η = clamp(η0, 0.0, 1.0)

    for _ in 1:max_iter
        x_m, dxdξ, dxdη, _, _ = eval_surface_point(ξ, η, ps_m, ns_m, kvm, B, Pm, nsd)
        gap = x_m .- x_s
        g1  = dot(dxdξ, gap)
        g2  = dot(dxdη, gap)
        (abs(g1) < tol && abs(g2) < tol) && break

        H11 = dot(dxdξ, dxdξ)
        H22 = dot(dxdη, dxdη)
        H12 = dot(dxdξ, dxdη)
        det_H = H11*H22 - H12^2
        det_H < 1e-30 && break

        Δξ = (-H22*g1 + H12*g2) / det_H
        Δη = ( H12*g1 - H11*g2) / det_H
        ξ  = clamp(ξ + Δξ, 0.0, 1.0)
        η  = clamp(η + Δη, 0.0, 1.0)
    end

    x_m, dxdξ, dxdη, R_m, spans_m = eval_surface_point(ξ, η, ps_m, ns_m, kvm, B, Pm, nsd)
    return ξ, η, x_m, dxdξ, dxdη, R_m, spans_m
end

"""
    closest_point_1d(ξ0, x_s, ps_m, ns_m, kvm, B, Pm, nsd;
                     tol=1e-13, max_iter=100) -> (ξ_m, x_m, dxdξ, R_m, span_m)

Newton-Raphson closest-point projection of point `x_s` onto a 1D NURBS boundary
in nsd-dimensional space.

Uses gradient g = (dx_m/dξ) · (x_m − x_s) and Hessian H ≈ |dx_m/dξ|²
(exact for p=1; ignores curvature for higher degrees). Clamps ξ to [0, 1].

Returns converged (ξ_m, x_m, tangent, shape functions R_m, span_m).
"""
function closest_point_1d(
    ξ0::Float64, x_s::AbstractVector{Float64},
    ps_m::Int, ns_m::Int, kvm::AbstractVector{Float64},
    B::AbstractMatrix{Float64}, Pm::AbstractVector{Int},
    nsd::Int;
    tol::Float64     = 1e-13,
    max_iter::Int    = 100
)
    ξ = clamp(ξ0, 0.0, 1.0)

    for _ in 1:max_iter
        x_m, dxdξ, _, _ = eval_boundary_point(ξ, ps_m, ns_m, kvm, B, Pm, nsd)
        gap = x_m .- x_s
        g   = dot(dxdξ, gap)
        H   = dot(dxdξ, dxdξ)
        abs(g) < tol && break
        H < 1e-30   && break
        ξ = clamp(ξ - g / H, 0.0, 1.0)
    end

    x_m, dxdξ, R_m, span_m = eval_boundary_point(ξ, ps_m, ns_m, kvm, B, Pm, nsd)
    return ξ, x_m, dxdξ, R_m, span_m
end
