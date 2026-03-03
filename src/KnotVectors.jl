# KnotVectors.jl
# Knot vector generation and h-refinement (knot insertion / k-refinement).
# Ported from MATLAB: knot_vectors_generator.m, krefinement.m,
#   multipleKnotInsertion.m, knotInsertion.m,
#   buildSegmentNumbering.m, buildPatchNumbering.m

# ─── Knot vector generation ───────────────────────────────────────────────────

"""
    generate_knot_vector(n, p) -> Vector{Float64}

Generate an open uniform B-spline knot vector for n control points of degree p.

The knot vector has p+1 zeros, n-p-1 interior knots uniformly spaced, and p+1 ones.
"""
function generate_knot_vector(n::Int, p::Int)::Vector{Float64}
    kv = zeros(n + p + 1)
    # Interior knots
    for j in 1:(n - p - 1)
        kv[p + 1 + j] = j / (n - p)
    end
    # Right clamping
    kv[n + 1:end] .= 1.0
    return kv
end

"""
    generate_knot_vectors(npc, npd, p, n) -> Vector{Vector{Vector{Float64}}}

Generate knot vectors for all patches.

- `npc`: number of patches
- `npd`: number of parametric dimensions
- `p[pc, i]`: degree for patch pc, direction i
- `n[pc, i]`: number of control points for patch pc, direction i

Returns `KV[pc][i]` = knot vector for patch `pc`, direction `i`.
"""
function generate_knot_vectors(
    npc::Int, npd::Int,
    p::Matrix{Int}, n::Matrix{Int}
)::Vector{Vector{Vector{Float64}}}
    KV = Vector{Vector{Vector{Float64}}}(undef, npc)
    for pc in 1:npc
        KVp = Vector{Vector{Float64}}(undef, npd)
        for i in 1:npd
            KVp[i] = generate_knot_vector(n[pc, i], p[pc, i])
        end
        KV[pc] = KVp
    end
    return KV
end

# ─── Knot insertion ───────────────────────────────────────────────────────────

"""
    knot_insertion(n, p, u, kv, Qw) -> (n_new, kv_new, Qw_new)

Insert a single knot `u` into knot vector `kv` for a B-spline curve of degree `p`
with n+1 control points. Control points are weighted: Qw[:, 1:end-1] are the spatial
coordinates times weight, Qw[:, end] is the weight.

Algorithm A5.1 (simplified single-knot version) from The NURBS Book.
"""
function knot_insertion(
    n::Int, p::Int, u::Float64,
    kv::Vector{Float64}, Qw::Matrix{Float64}
)::Tuple{Int, Vector{Float64}, Matrix{Float64}}
    # Multiplicity of u in kv
    s = count(==(u), kv)

    # Find knot span k (1-based); n_basis = n (0-based max CP index = #CPs - 1)
    k = find_span(n, p, u, kv)

    nrow = size(Qw, 2)
    Qw_new = zeros(n + 2, nrow)   # one new CP after insertion

    for i in 1:n+2
        if i < k - p + 1
            Qw_new[i, :] = Qw[i, :]
        end
        if (k - p + 1) <= i <= (k - s)
            alpha = (u - kv[i]) / (kv[i + p] - kv[i])
            Qw_new[i, :] = Qw[i-1, :] + alpha * (Qw[i, :] - Qw[i-1, :])
        end
        if i > k - s
            Qw_new[i, :] = Qw[i-1, :]
        end
    end

    kv_new = [kv[1:k]; u; kv[k+1:end]]
    return n + 1, kv_new, Qw_new
end

# ─── Helpers for k-refinement ────────────────────────────────────────────────

"""
    build_patch_numbering(pd, npd, n) -> Vector{Int}

Build local numbering of patch control points reordered so that direction `pd` is last.
Ported from buildPatchNumbering.m.
"""
function build_patch_numbering(pd::Int, npd::Int, n::Vector{Int})::Vector{Int}
    ipb = [i for i in 1:npd if i != pd]
    ns  = n[ipb]
    nscp = prod(ns; init=1)

    total = prod(n; init=1)
    P = zeros(Int, total)
    locNum = 1

    for As in 1:nscp
        ncs = nurbs_coords(As, npd - 1, ns)
        for i in 1:n[pd]
            nc = zeros(Int, npd)
            nc[pd] = i
            for (ki, j) in enumerate(ipb)
                nc[j] = ncs[ki]
            end
            # Linear index from NURBS coords
            A = nc[1]
            for j in 2:npd
                nn = prod(n[1:j-1]; init=1)
                A += nn * (nc[j] - 1)
            end
            P[A] = locNum
            locNum += 1
        end
    end
    return P
end

"""
    build_segment_numbering(pd, npd, n, P) -> Matrix{Int}

Build segment numbering matrix: `Ps[As, i]` = global CP index for ray `As`, position `i`
along direction `pd`. Ported from buildSegmentNumbering.m.
"""
function build_segment_numbering(
    pd::Int, npd::Int, n::Vector{Int}, P::Vector{Int}
)::Matrix{Int}
    ipb = [i for i in 1:npd if i != pd]
    ns  = n[ipb]
    nscp = prod(ns; init=1)

    Ps = zeros(Int, nscp, n[pd])
    for As in 1:nscp
        ncs = nurbs_coords(As, npd - 1, ns)
        for i in 1:n[pd]
            nc = zeros(Int, npd)
            nc[pd] = i
            for (ki, j) in enumerate(ipb)
                nc[j] = ncs[ki]
            end
            A = nc[1]
            for j in 2:npd
                nn = prod(n[1:j-1]; init=1)
                A += nn * (nc[j] - 1)
            end
            Ps[As, i] = P[A]
        end
    end
    return Ps
end

"""
    multiple_knot_insertion(pd, npd, nsd, n, p, kv_pd, Bw, P, u_new) ->
        (n_new, kv_new, Bw_new, P_new)

Insert multiple knots `u_new` into direction `pd` of the patch.
`Bw` is the global weighted control point array (rows = CPs, cols = [x*w, y*w, w]).
`P` is the patch-to-global CP index array.

Ported from multipleKnotInsertion.m.
"""
function multiple_knot_insertion(
    pd::Int, npd::Int, nsd::Int,
    n::Vector{Int}, p::Vector{Int},
    kv_pd::Vector{Float64},
    Bw::Matrix{Float64}, P::Vector{Int},
    u_new::Vector{Float64}
)::Tuple{Vector{Int}, Vector{Float64}, Matrix{Float64}, Vector{Int}}

    Ps = build_segment_numbering(pd, npd, n, P)
    nscp = size(Ps, 1)
    nrow = size(Bw, 2)

    # Insert knots along each ray/layer
    Qwp = zeros(0, nrow)
    nr = n[pd] - 1   # knot_insertion uses 0-based max index (NURBS Book convention)
    kv_new = kv_pd

    for As in 1:nscp
        # Extract ray
        Qw = Bw[Ps[As, :], :]
        # Insert each knot
        kv_tmp = kv_pd
        nr_tmp = n[pd] - 1   # 0-based max index = count - 1
        for u in u_new
            nr_tmp, kv_tmp, Qw = knot_insertion(nr_tmp, p[pd], u, kv_tmp, Qw)
        end
        Qwp = vcat(Qwp, Qw)
        kv_new = kv_tmp
        nr = nr_tmp
    end

    n_new = copy(n)
    n_new[pd] = nr + 1   # convert 0-based max back to count

    Pq = build_patch_numbering(pd, npd, n_new)
    npcp = prod(n_new; init=1)

    tol = 1e-5
    P_new = copy(P)
    # Resize P_new to hold npcp entries
    P_new = zeros(Int, npcp)
    Bw_new = copy(Bw)

    for i in 1:npcp
        x = Qwp[Pq[i], 1:nsd] ./ Qwp[Pq[i], end]   # physical coords
        found = false
        for j in 1:size(Bw_new, 1)
            y = Bw_new[j, 1:nsd] ./ Bw_new[j, end]  # physical coords
            if norm(x .- y) <= tol
                P_new[i] = j
                found = true
                break
            end
        end
        if !found
            Bw_new = vcat(Bw_new, Qwp[[Pq[i]], :])
            P_new[i] = size(Bw_new, 1)
        end
    end

    return n_new, kv_new, Bw_new, P_new
end

# ─── k-refinement (outer loop) ───────────────────────────────────────────────

"""
    krefinement(nsd, npd, npc, n, p, KV, B, P, kref_data) ->
        (n_new, ncp_new, KV_new, B_new, P_new)

Perform k-refinement (knot insertion) on all patches.

- `kref_data`: vector of vectors; each entry is `[pc, pd, u1, u2, ...]`
  where `pc` = patch index, `pd` = parametric direction, `u_i` = knots to insert.
- `B`: global control points (ncp × (nsd+1)), last column is weight.
- `P`: patch-to-global CP index, `P[pc]` = Vector{Int}.
- `KV`: knot vectors, `KV[pc][i]` = Vector{Float64}.

Returns updated (n, ncp, KV, B, P) after all insertions.
"""
function krefinement(
    nsd::Int, npd::Int, npc::Int,
    n::Matrix{Int}, p::Matrix{Int},
    KV::Vector{Vector{Vector{Float64}}},
    B::Matrix{Float64},
    P::Vector{Vector{Int}},
    kref_data::Vector{Vector{Float64}}
)
    # Build weighted control points Bw  (weight is last column)
    ncp_total = size(B, 1)
    Bw = copy(B)
    for i in 1:ncp_total
        w = B[i, end]
        Bw[i, 1:nsd] = B[i, 1:nsd] .* w
    end

    n_mut = [n[pc, :] for pc in 1:npc]
    KV_mut = deepcopy(KV)
    P_mut  = deepcopy(P)

    for krf in kref_data
        pc  = Int(krf[1])
        pd  = Int(krf[2])
        u   = krf[3:end]
        isempty(u) && continue

        n_pc = n_mut[pc]
        p_pc = p[pc, :]
        kv_pd = KV_mut[pc][pd]
        P_pc  = P_mut[pc]

        n_pc_new, kv_pd_new, Bw, P_pc_new = multiple_knot_insertion(
            pd, npd, nsd, n_pc, p_pc, kv_pd, Bw, P_pc, Float64.(u)
        )
        n_mut[pc] = n_pc_new
        KV_mut[pc][pd] = kv_pd_new
        P_mut[pc] = P_pc_new
    end

    # Divide back by weights  (weight is last column)
    B_new = copy(Bw)
    for i in 1:size(Bw, 1)
        w = Bw[i, end]
        B_new[i, 1:nsd] = Bw[i, 1:nsd] ./ w
    end

    # Compact: keep only active CPs (those referenced by at least one patch)
    n_total = size(B_new, 1)
    active = zeros(Bool, n_total)
    for pc in 1:npc
        for idx in P_mut[pc]
            active[idx] = true
        end
    end
    ncp_new = sum(active)

    # Build renumbering map old→new
    old_to_new = zeros(Int, n_total)
    A = 1
    for i in 1:n_total
        if active[i]
            old_to_new[i] = A
            A += 1
        end
    end

    B_compact = B_new[active, :]
    P_new = [old_to_new[P_mut[pc]] for pc in 1:npc]
    n_new = reduce(vcat, [reshape(n_mut[pc], 1, :) for pc in 1:npc]; init=zeros(Int,0,npd))

    return n_new, ncp_new, KV_mut, B_compact, P_new
end
