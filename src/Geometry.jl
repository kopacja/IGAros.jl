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
    shape_function!(ws, pc, p, n, KV, B, P, xi_tilde, nen, nsd, npd, el, n0, IEN, INC)
    shape_function!(ws, pc, p, n, KV, B, P, igp,      nen, nsd, npd, el, n0, IEN, INC)

In-place variant of `shape_function`.  All output is written into `ws`:
  `ws.R`, `ws.dR_dx`, `ws.dx_dXi`, `ws.n_vec`.
Returns the Jacobian determinant `detJ`.  No heap allocation occurs.

The `igp::Int` overload reads Gauss-point coordinates directly from
`pc.gp_coords[igp, :]`, avoiding a `@view` allocation.
"""
function shape_function!(
    ws::AssemblyWorkspace,
    pc::PatchConstants,
    p::AbstractVector{Int},
    n::AbstractVector{Int},
    KV::AbstractVector{<:AbstractVector{Float64}},
    B::AbstractMatrix{Float64},
    P::AbstractVector{Int},
    igp::Int,
    nen::Int, nsd::Int, npd::Int,
    el::Int, n0::AbstractVector{Int},
    IEN::AbstractMatrix{Int},
    INC::AbstractVector{<:AbstractVector{Int}}
)::Float64

    # ── 1. Parent-to-parametric mapping ──────────────────────────────────────
    @inbounds for i in 1:npd
        kv = KV[i]
        a  = kv[n0[i]]
        b  = kv[n0[i] + 1]
        ws.Xi[i] = 0.5 * ((b - a) * pc.gp_coords[igp, i] + (b + a))
        ws.dXi_dtildeXi[i, i] = 0.5 * (b - a)
        for j in 1:npd
            j != i && (ws.dXi_dtildeXi[i, j] = 0.0)
        end
    end

    return _shape_function_core!(ws, pc, p, n, KV, B, P, nen, nsd, npd, el, n0, IEN, INC)
end

function shape_function!(
    ws::AssemblyWorkspace,
    pc::PatchConstants,
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
)::Float64

    # ── 1. Parent-to-parametric mapping ──────────────────────────────────────
    @inbounds for i in 1:npd
        kv = KV[i]
        a  = kv[n0[i]]
        b  = kv[n0[i] + 1]
        ws.Xi[i] = 0.5 * ((b - a) * xi_tilde[i] + (b + a))
        ws.dXi_dtildeXi[i, i] = 0.5 * (b - a)
        for j in 1:npd
            j != i && (ws.dXi_dtildeXi[i, j] = 0.0)
        end
    end

    return _shape_function_core!(ws, pc, p, n, KV, B, P, nen, nsd, npd, el, n0, IEN, INC)
end

"""Internal: shared core of shape_function! after parent-to-parametric mapping."""
function _shape_function_core!(
    ws::AssemblyWorkspace,
    pc::PatchConstants,
    p::AbstractVector{Int},
    n::AbstractVector{Int},
    KV::AbstractVector{<:AbstractVector{Float64}},
    B::AbstractMatrix{Float64},
    P::AbstractVector{Int},
    nen::Int, nsd::Int, npd::Int,
    el::Int, n0::AbstractVector{Int},
    IEN::AbstractMatrix{Int},
    INC::AbstractVector{<:AbstractVector{Int}}
)::Float64

    # ── 1D basis functions via in-place bspline ──────────────────────────────
    @inbounds for i in 1:npd
        bspline_basis_and_deriv!(
            ws.ders, ws.ndu, ws.bsp_left, ws.bsp_right, ws.a_bsp,
            n0[i], ws.Xi[i], p[i], 1, KV[i]
        )
        # Copy results into per-direction buffers
        for j in 1:p[i]+1
            ws.NN[i][j]      = ws.ders[1, j]
            ws.dNN_dXi[i][j] = ws.ders[2, j]
        end
    end

    # ── 2. Build tensor-product NURBS functions ───────────────────────────────
    # Extract weights (no allocation — write into ws.W_buf)
    @inbounds for a in 1:nen
        ws.W_buf[a] = B[P[IEN[el, a]], end]
    end

    # NURBS numerator: N_a = W_a * ∏_i NN[i][idx[a,i]]
    @inbounds for a in 1:nen
        v = ws.W_buf[a]
        for i in 1:npd
            v *= ws.NN[i][pc.idx[a, i]]
        end
        ws.N_num[a] = v
    end

    sum_W = 0.0
    @inbounds for a in 1:nen
        sum_W += ws.N_num[a]
    end

    # Parametric derivatives
    @inbounds for i in 1:npd
        for a in 1:nen
            v = ws.dNN_dXi[i][pc.idx[a, i]]
            for k in 1:npd
                k == i && continue
                v *= ws.NN[k][pc.idx[a, k]]
            end
            ws.dN_dXi_num[a, i] = v * ws.W_buf[a]
        end
    end

    # dsum_W
    @inbounds for i in 1:npd
        s = 0.0
        for a in 1:nen
            s += ws.dN_dXi_num[a, i]
        end
        ws.dsum_W[i] = s
    end

    # NURBS basis R and derivatives dR_dXi
    inv_sum_W = 1.0 / sum_W
    @inbounds for a in 1:nen
        ws.R[a] = ws.N_num[a] * inv_sum_W
    end
    @inbounds for i in 1:npd
        for a in 1:nen
            ws.dR_dXi[a, i] = (ws.dN_dXi_num[a, i] - ws.R[a] * ws.dsum_W[i]) * inv_sum_W
        end
    end

    # ── 3. Physical Jacobian dx/dXi ──────────────────────────────────────────
    # Copy Xcp in-place (avoid fancy indexing allocation)
    @inbounds for a in 1:nen
        gidx = P[IEN[el, a]]
        for d in 1:nsd
            ws.Xcp[a, d] = B[gidx, d]
        end
    end

    # dx_dXi = Xcp' * dR_dXi  →  (nsd × npd)
    @inbounds for j in 1:npd
        for d in 1:nsd
            s = 0.0
            for a in 1:nen
                s += ws.Xcp[a, d] * ws.dR_dXi[a, j]
            end
            ws.dx_dXi[d, j] = s
        end
    end

    # ── 4. Jacobian determinant and physical gradients ────────────────────────
    detJ = 0.0

    if nsd == npd
        # J_mat = dx_dXi * dXi_dtildeXi  (nsd × nsd)
        @inbounds for j in 1:nsd
            for i in 1:nsd
                s = 0.0
                for k in 1:npd
                    s += ws.dx_dXi[i, k] * ws.dXi_dtildeXi[k, j]
                end
                ws.J_mat[i, j] = s
            end
        end

        if nsd == 2
            detJ = ws.J_mat[1,1] * ws.J_mat[2,2] - ws.J_mat[1,2] * ws.J_mat[2,1]
            # Inline 2×2 inverse of dx_dXi
            det_dxdXi = ws.dx_dXi[1,1] * ws.dx_dXi[2,2] - ws.dx_dXi[1,2] * ws.dx_dXi[2,1]
            inv_det = 1.0 / det_dxdXi
            ws.dXi_dx[1,1] =  ws.dx_dXi[2,2] * inv_det
            ws.dXi_dx[1,2] = -ws.dx_dXi[1,2] * inv_det
            ws.dXi_dx[2,1] = -ws.dx_dXi[2,1] * inv_det
            ws.dXi_dx[2,2] =  ws.dx_dXi[1,1] * inv_det
        elseif nsd == 3
            # Inline 3×3 determinant
            a11 = ws.J_mat[1,1]; a12 = ws.J_mat[1,2]; a13 = ws.J_mat[1,3]
            a21 = ws.J_mat[2,1]; a22 = ws.J_mat[2,2]; a23 = ws.J_mat[2,3]
            a31 = ws.J_mat[3,1]; a32 = ws.J_mat[3,2]; a33 = ws.J_mat[3,3]
            detJ = a11*(a22*a33 - a23*a32) - a12*(a21*a33 - a23*a31) + a13*(a21*a32 - a22*a31)
            # Inline 3×3 inverse of dx_dXi
            b11 = ws.dx_dXi[1,1]; b12 = ws.dx_dXi[1,2]; b13 = ws.dx_dXi[1,3]
            b21 = ws.dx_dXi[2,1]; b22 = ws.dx_dXi[2,2]; b23 = ws.dx_dXi[2,3]
            b31 = ws.dx_dXi[3,1]; b32 = ws.dx_dXi[3,2]; b33 = ws.dx_dXi[3,3]
            det_b = b11*(b22*b33 - b23*b32) - b12*(b21*b33 - b23*b31) + b13*(b21*b32 - b22*b31)
            inv_det_b = 1.0 / det_b
            ws.dXi_dx[1,1] = (b22*b33 - b23*b32) * inv_det_b
            ws.dXi_dx[1,2] = (b13*b32 - b12*b33) * inv_det_b
            ws.dXi_dx[1,3] = (b12*b23 - b13*b22) * inv_det_b
            ws.dXi_dx[2,1] = (b23*b31 - b21*b33) * inv_det_b
            ws.dXi_dx[2,2] = (b11*b33 - b13*b31) * inv_det_b
            ws.dXi_dx[2,3] = (b13*b21 - b11*b23) * inv_det_b
            ws.dXi_dx[3,1] = (b21*b32 - b22*b31) * inv_det_b
            ws.dXi_dx[3,2] = (b12*b31 - b11*b32) * inv_det_b
            ws.dXi_dx[3,3] = (b11*b22 - b12*b21) * inv_det_b
        end

        # dR_dx = dR_dXi * dXi_dx  (nen × nsd)
        @inbounds for j in 1:nsd
            for a in 1:nen
                s = 0.0
                for k in 1:npd
                    s += ws.dR_dXi[a, k] * ws.dXi_dx[k, j]
                end
                ws.dR_dx[a, j] = s
            end
        end

    elseif nsd == 3 && npd == 2
        gx = ws.dx_dXi[2,1]*ws.dx_dXi[3,2] - ws.dx_dXi[3,1]*ws.dx_dXi[2,2]
        gy = ws.dx_dXi[3,1]*ws.dx_dXi[1,2] - ws.dx_dXi[1,1]*ws.dx_dXi[3,2]
        gz = ws.dx_dXi[1,1]*ws.dx_dXi[2,2] - ws.dx_dXi[2,1]*ws.dx_dXi[1,2]
        g_norm = sqrt(gx^2 + gy^2 + gz^2)
        ws.n_vec[1] = gx / g_norm
        ws.n_vec[2] = gy / g_norm
        ws.n_vec[3] = gz / g_norm
        detJ = ws.dXi_dtildeXi[1,1] * ws.dXi_dtildeXi[2,2] * g_norm

    elseif nsd == 2 && npd == 1
        gx = ws.dx_dXi[1,1]; gy = ws.dx_dXi[2,1]
        g_norm = sqrt(gx^2 + gy^2)
        # n = R90 * tangent (outward normal for 2D curve)
        ws.n_vec[1] =  gy / g_norm
        ws.n_vec[2] = -gx / g_norm
        detJ = ws.dXi_dtildeXi[1,1] * g_norm
    end

    return detJ
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
