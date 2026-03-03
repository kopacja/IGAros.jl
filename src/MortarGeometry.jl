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
