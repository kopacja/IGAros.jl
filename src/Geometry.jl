# Geometry.jl
# NURBS shape functions: basis evaluation, Jacobians, physical gradients.
# Ported from MATLAB: Shape_function.m, @NURBSpatch/getSecondDerivatives.m

"""
    shape_function(p, n, KV, B, P, xi_tilde, nen, nsd, npd, el, n0, IEN, INC)
        -> (R, dR_dx, dx_dXi, detJ, n_vec)

Evaluate NURBS shape functions and their physical-space gradients at parent
coordinates `xi_tilde ∈ [-1,1]^npd`.

Inputs:
- `p[i]`: polynomial degree per direction (length npd)
- `n[i]`: number of control points per direction (length npd)
- `KV[i]`: knot vector per direction (length npd, each a Vector{Float64})
- `B`: global control point array (ncp × (nsd+1)); last column is weight
- `P`: patch control point indices into B (length nnp)
- `xi_tilde`: parent-element coordinates (length npd)
- `nen`: number of local basis functions per element
- `nsd`: number of spatial dimensions
- `npd`: number of parametric dimensions
- `el`: element index (1-based)
- `n0[i]`: knot span index per direction (from `find_span`)
- `IEN`: element-to-local-node connectivity (nel × nen)
- `INC`: local-to-NURBS-coord map

Outputs:
- `R`: NURBS basis functions (nen,)
- `dR_dx`: physical gradients (nen × nsd) [only meaningful when nsd == npd]
- `dx_dXi`: physical-to-parametric Jacobian (nsd × npd)
- `detJ`: Jacobian determinant (including parent-to-parametric scaling)
- `n_vec`: outward unit normal (nsd,) [for boundary elements where nsd > npd]

Ported from Shape_function.m + getSecondDerivatives.m.
"""
function shape_function(
    p::AbstractVector{Int},
    n::AbstractVector{Int},
    KV::AbstractVector{<:AbstractVector{Float64}},
    B::AbstractMatrix{Float64},
    P::AbstractVector{Int},
    xi_tilde::AbstractVector{Float64},
    nen::Int, nsd::Int, npd::Int,
    el::Int, n0::AbstractVector{Int},
    IEN::AbstractMatrix{Int},
    INC::AbstractVector{<:AbstractVector{Int}}
)

    # ── 1. Parent-to-parametric mapping ──────────────────────────────────────
    Xi = zeros(npd)
    dXi_dtildeXi = zeros(npd, npd)   # diagonal: Jacobian of parent→param mapping

    NN      = Vector{Vector{Float64}}(undef, npd)
    dNN_dXi = Vector{Vector{Float64}}(undef, npd)

    for i in 1:npd
        kv  = KV[i]
        a   = kv[n0[i]]
        b   = kv[n0[i] + 1]
        Xi[i] = 0.5 * ((b - a) * xi_tilde[i] + (b + a))
        dXi_dtildeXi[i, i] = 0.5 * (b - a)

        ders = bspline_basis_and_deriv(n0[i], Xi[i], p[i], 1, kv)
        NN[i]      = ders[1, :]
        dNN_dXi[i] = ders[2, :]
    end

    # ── 2. Build tensor-product NURBS functions ───────────────────────────────
    A_nodes = IEN[el, 1:nen]   # local node indices within patch

    # Build local (direction-offset) indices nc[a, i] for tensor product
    # nc[a, i] ∈ {0, …, p[i]}  (zero-based offset from the anchor node)
    nc = _local_nc(p, npd, nen)  # nen × npd, 0-based offsets

    # Weighted numerators and their parametric derivatives
    W  = B[P[A_nodes], end]      # weights at element nodes (nen,) — last column

    # For each direction, precompute the 1D basis values indexed by local offset
    # idx[a, i] = (p[i]+1) - nc[a, i]  (1-based lookup into NN[i])
    idx = zeros(Int, nen, npd)
    for i in 1:npd
        idx[:, i] = (p[i] + 1) .- nc[:, i]
    end

    # NURBS numerator: N_a = ∏_i NN[i][idx[a,i]]
    N_num = ones(nen)
    for i in 1:npd
        N_num .*= NN[i][idx[:, i]]
    end
    N_num .*= W  # weighted

    sum_W = sum(N_num)           # denominator W(Xi)

    # Parametric derivatives: ∂N_a/∂ξ_i = (∂NN_i/∂ξ_i) * ∏_{k≠i} NN[k]
    # Compute the "unweighted cross-product" directly to avoid division by zero.
    dN_dXi_num = zeros(nen, npd)
    for i in 1:npd
        col = dNN_dXi[i][idx[:, i]]   # derivative in direction i
        for k in 1:npd
            k == i && continue
            col = col .* NN[k][idx[:, k]]
        end
        dN_dXi_num[:, i] = col .* W   # apply weights
    end

    # Derivative of denominator W(Xi)
    dsum_W = vec(sum(dN_dXi_num; dims=1))  # length npd

    # NURBS basis and derivatives (rational formula)
    R      = N_num ./ sum_W
    # dR/dXi_i = (dN_i*W - N*W*dsum_i) / W^2  = (dN_i - R*dsum_i) / sum_W
    dR_dXi = (dN_dXi_num .- R * dsum_W') ./ sum_W  # nen × npd

    # ── 3. Physical Jacobian: dx/dXi ─────────────────────────────────────────
    Xcp = B[P[A_nodes], 1:nsd]     # physical coords of element nodes (nen × nsd)
    dx_dXi = Xcp' * dR_dXi          # nsd × npd

    # ── 4. Jacobian determinant and physical gradients ────────────────────────
    dR_dx  = zeros(nen, nsd)
    n_vec  = zeros(nsd)
    detJ   = 0.0

    if nsd == npd
        J_mat = dx_dXi * dXi_dtildeXi   # nsd × nsd
        detJ  = det(J_mat)
        dXi_dx = inv(dx_dXi)
        dR_dx  = dR_dXi * dXi_dx         # nen × nsd
    elseif nsd == 3 && npd == 2
        # Surface element in 3D
        gx = dx_dXi[2, 1] * dx_dXi[3, 2] - dx_dXi[3, 1] * dx_dXi[2, 2]
        gy = dx_dXi[3, 1] * dx_dXi[1, 2] - dx_dXi[1, 1] * dx_dXi[3, 2]
        gz = dx_dXi[1, 1] * dx_dXi[2, 2] - dx_dXi[2, 1] * dx_dXi[1, 2]
        g_norm = sqrt(gx^2 + gy^2 + gz^2)
        n_vec  = [gx; gy; gz] ./ g_norm
        detJ   = dXi_dtildeXi[1, 1] * dXi_dtildeXi[2, 2] * g_norm
    elseif nsd == 2 && npd == 1
        # Curve element in 2D
        gx = dx_dXi[1, 1]; gy = dx_dXi[2, 1]
        g_norm = sqrt(gx^2 + gy^2)
        t_vec  = [gx; gy; 0.0] ./ g_norm
        n_full = cross(t_vec, [0.0; 0.0; 1.0])
        n_vec  = n_full[1:2]
        detJ   = dXi_dtildeXi[1, 1] * g_norm
    end

    return R, dR_dx, dx_dXi, detJ, n_vec
end

"""
    find_element_span(p, n, KV, xi_tilde, INC) -> (n0, el)

Given parametric coordinates (parent space ξ̃ ∈ [-1,1]^npd), compute the
knot span indices n0[i] and the 1-based element number el.
"""
function find_element_span(
    p::AbstractVector{Int},
    n::AbstractVector{Int},
    KV::AbstractVector{<:AbstractVector{Float64}},
    xi_tilde::AbstractVector{Float64}
)::Tuple{Vector{Int}, Int}
    npd = length(p)
    n0 = zeros(Int, npd)

    # Map parent coords to actual parametric coords first via representative
    # midpoint: just evaluate at xi_tilde and find span
    # (For find_span we need the actual parametric coordinate.)
    # Here xi_tilde is in [-1,1]; we use the element's knot span to map.
    # In the assembly loop, n0 is provided externally from the INC table.
    # This utility just wraps find_span per direction using midpoints.
    for i in 1:npd
        kv = KV[i]
        # Map to full parameter range [kv[p+1], kv[end-p]]
        u  = 0.5 * (kv[end] + kv[1]) + 0.5 * xi_tilde[i] * (kv[end] - kv[1])
        n0[i] = find_span(n[i] - 1, p[i], clamp(u, kv[1], kv[end] - 1e-14), kv)
    end

    el = n0[1] - p[1]
    mult = n[1] - p[1]
    for i in 2:npd
        el += (n0[i] - p[i] - 1) * mult
        mult *= (n[i] - p[i])
    end

    return n0, el
end

# ─── Internal helpers ─────────────────────────────────────────────────────────

"""
    _local_nc(p, npd, nen) -> Matrix{Int}

Build local multi-index offset table: `nc[a, i]` is the 0-based offset in
direction i for local node a. Order matches MATLAB BuildIEN (loop over directions
fastest in direction 1).

Size: (nen × npd).
"""
function _local_nc(p::AbstractVector{Int}, npd::Int, nen::Int)::Matrix{Int}
    nc = zeros(Int, nen, npd)
    denom = 1
    for i in 1:npd
        for a in 1:nen
            nc[a, i] = floor(Int, mod((a - 1) / denom, p[i] + 1))
        end
        denom *= (p[i] + 1)
    end
    return nc
end
