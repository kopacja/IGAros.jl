# MortarAssembly.jl
# Twin Mortar coupling matrix assembly for non-conforming mesh tying.
# Ported from MATLAB: buildTwoHalfPassLagrangeTying.m
#
# Integration strategy pattern:
#   build_mortar_coupling(..., ElementBasedIntegration())  — default, GP per slave element
#   build_mortar_coupling(..., SegmentBasedIntegration())  — exact segment-based quadrature


"""
    InterfacePair(slave_patch, slave_facet, master_patch, master_facet)

Describes one half-pass of the Twin Mortar method.
For full twin mortar tying supply two pairs with roles swapped:
  [(1, facet_s, 2, facet_m), (2, facet_m, 1, facet_s)]
"""
struct InterfacePair
    slave_patch::Int
    slave_facet::Int
    master_patch::Int
    master_facet::Int
end

"""
    build_interface_cps(pairs, p, n, KV, P, npd, nnp,
                        formulation = TwinMortarFormulation()) -> Pc

Collect boundary control point indices that form the multiplier space.

- `TwinMortarFormulation` (default): union of slave **and** master facet CPs
  from all pairs (both surfaces carry multipliers).
- `SinglePassFormulation`: slave facet CPs only from all pairs (multipliers
  on slave surface only).
"""
function build_interface_cps(
    pairs::Vector{InterfacePair},
    p::Matrix{Int}, n::Matrix{Int},
    KV::Vector{<:AbstractVector{<:AbstractVector{Float64}}},
    P::Vector{Vector{Int}},
    npd::Int, nnp::Vector{Int},
    formulation::FormulationStrategy = TwinMortarFormulation()
)::Vector{Int}
    cps  = Int[]
    seen = Set{Tuple{Int,Int}}()
    for pair in pairs, (pc, facet) in _cp_facets(pair, formulation)
        (pc, facet) in seen && continue
        push!(seen, (pc, facet))
        _, _, _, Ps, _, _, _, _, _, _, _ = get_segment_patch(
            p[pc, :], n[pc, :], KV[pc], P[pc], npd, nnp[pc], facet
        )
        append!(cps, Ps)
    end
    return unique(sort(cps))
end

# Which (patch, facet) pairs contribute to Pc for each formulation
_cp_facets(pair, ::TwinMortarFormulation)  =
    [(pair.slave_patch, pair.slave_facet),
     (pair.master_patch, pair.master_facet)]
_cp_facets(pair, ::DualPassFormulation)    =
    [(pair.slave_patch, pair.slave_facet),
     (pair.master_patch, pair.master_facet)]
_cp_facets(pair, ::SinglePassFormulation)  =
    [(pair.slave_patch, pair.slave_facet)]

# ─── Global Cartesian constraint directions ─────────────────────────────────
#
# The Lagrange multiplier λ is resolved in global Cartesian directions (e1,e2,e3)
# rather than the local normal/tangent frame (n, t1, t2).
# This simplifies the formulation and avoids issues with varying normals
# on curved interfaces.  dir_vecs = I (identity matrix, nsd × nsd).

# ─── Shared accumulation kernel ──────────────────────────────────────────────

"""
    _accumulate_mortar!(C, Z, R_s, R_m, slave_cps, master_cps,
                        dir_vecs, Pc, nlm, ID, ned, gwJ, epss)

Accumulate one Gauss-point contribution to the Twin Mortar coupling matrix C
and stabilization matrix Z.  Called identically by both element-based and
segment-based integration loops.

Twin Mortar formulation (per half-pass s on Γ_s):
  C assembly — two-half-pass (each pass fills its own surface row):
  - slave disp, slave mult: C[eq_s(b), As_c] += δ_{id} R_s[b] R_s[c] gwJ   (+D^(s))
  - slave disp, master mult: C[eq_s(b), Am_c] -= δ_{id} R_s[b] R_m[c] gwJ  (-M^(sm))
  Z assembly — full averaged: -(ε/2)(R_s + R_m)(R_s + R_m)^T
  - Z[s,s]: -(ε/2) R_s ⊗ R_s,  Z[m,m]: -(ε/2) R_m ⊗ R_m
  - Z[s,m]: -(ε/2) R_s ⊗ R_m,  Z[m,s]: -(ε/2) R_m ⊗ R_s
  Summing both passes: Z = ε[D̄^(1), M̄; M̄^T, D̄^(2)]  (symmetric)

Arguments
- `R_s, R_m`   : slave/master NURBS shape functions at the Gauss point
- `slave_cps`  : active slave CP global indices
- `master_cps` : active master CP global indices
- `dir_vecs`   : (ned × n_dirs) matrix of interface directions;
                 column 1 = outward normal, columns 2..n_dirs = tangents.
                 For 2D: 2×2 [n_vec | t_vec]; for 3D: 3×3 [n | t1 | t2].
- `Pc`         : sorted vector of interface multiplier CP indices
- `nlm`        : length(Pc) (block size of the multiplier)
- `ID`         : equation-number array (ned × ncp)
- `ned`        : number of displacement DOFs per node
- `gwJ`        : Gauss weight × Jacobian (integration measure)
- `epss`       : stabilization parameter ε
"""
# Dispatch wrapper — TwinMortarFormulation delegates to the core kernel
@inline function _accumulate_mortar!(C, Z, ::TwinMortarFormulation, args...)
    _accumulate_mortar!(C, Z, args...)
end

# Dispatch wrapper — DualPassFormulation has its own kernel
@inline function _accumulate_mortar!(C, Z, ::DualPassFormulation, args...)
    _accumulate_mortar_dp!(C, Z, args...)
end


# ─── Single-pass accumulation kernel ─────────────────────────────────────────
"""
Single-pass mortar kernel.  Assembles:
  C[slave_disp,  slave_mult] += -n_d · R_s[b] · R_s[c] · gwJ   (D^(s) block)
  C[master_disp, slave_mult] += +n_d · R_m[b] · R_s[c] · gwJ   (M^(sm) block)
  Z: no contribution (hard constraint, Z = 0).
"""
function _accumulate_mortar!(
    C, Z, ::SinglePassFormulation,
    R_s::AbstractVector{Float64}, R_m::AbstractVector{Float64},
    slave_cps::AbstractVector{Int}, master_cps::AbstractVector{Int},
    dir_vecs::AbstractMatrix{Float64},
    Pc::AbstractVector{Int}, nlm::Int,
    ID::AbstractMatrix{Int}, ned::Int,
    gwJ::Float64, ::Float64   # epss ignored for single-pass
)
    nsen_s = length(slave_cps)
    nsen_m = length(master_cps)
    n_dirs = size(dir_vecs, 2)

    # ── D^(s): force on slave DOFs due to slave multipliers ──────────────────
    for b in 1:nsen_s
        cp_s_b = slave_cps[b]
        for i in 1:ned
            eq_i = ID[i, cp_s_b]
            eq_i == 0 && continue
            for c in 1:nsen_s
                cp_s_c = slave_cps[c]
                As_c   = findfirst(==(cp_s_c), Pc)
                As_c === nothing && continue
                for d in 1:n_dirs
                    C[eq_i, As_c + (d-1)*nlm] -= dir_vecs[i, d] * R_s[b] * R_s[c] * gwJ
                end
            end
        end
    end

    # ── M^(sm): force on master DOFs due to slave multipliers (+ sign) ───────
    for b in 1:nsen_m
        cp_m_b = master_cps[b]
        for i in 1:ned
            eq_i = ID[i, cp_m_b]
            eq_i == 0 && continue
            for c in 1:nsen_s
                cp_s_c = slave_cps[c]
                As_c   = findfirst(==(cp_s_c), Pc)
                As_c === nothing && continue
                for d in 1:n_dirs
                    C[eq_i, As_c + (d-1)*nlm] += dir_vecs[i, d] * R_m[b] * R_s[c] * gwJ
                end
            end
        end
    end
    # Z: no contribution
end

# ─── Core Twin Mortar kernel (paper formulation: halfC + full-averaged Z) ─────
#
# C: two-half-pass assembly — each pass s fills the ENTIRE displacement row
#    for surface s.  Pass s on Γ_s contributes:
#      C[disp_s, λ_s] += R_s ⊗ R_s · gwJ           (+D^(s))
#      C[disp_s, λ_m] -= R_s ⊗ R_m · gwJ           (��M^(sm))
#
# Z: full two-pass averaged — -(ε/2)(R_s + R_m)(R_s + R_m)^T
#    accumulates all 4 sub-blocks per pass, giving Z = ε[D̄, M̄; M̄^T, D̄]
#    which is symmetric by construction.
#
function _accumulate_mortar!(
    C, Z,
    R_s::AbstractVector{Float64}, R_m::AbstractVector{Float64},
    slave_cps::AbstractVector{Int}, master_cps::AbstractVector{Int},
    dir_vecs::AbstractMatrix{Float64},
    Pc::AbstractVector{Int}, nlm::Int,
    ID::AbstractMatrix{Int}, ned::Int,
    gwJ::Float64, epss::Float64
)
    nsen_s = length(slave_cps)
    nsen_m = length(master_cps)
    n_dirs = size(dir_vecs, 2)   # number of multiplier directions (= nsd)

    # ── C: slave disp rows, slave multiplier cols  +D^(s) ────────────────────
    for b in 1:nsen_s
        cp_s_b = slave_cps[b]
        for i in 1:ned
            eq_i = ID[i, cp_s_b]
            eq_i == 0 && continue
            for c in 1:nsen_s
                cp_s_c = slave_cps[c]
                As_c   = findfirst(==(cp_s_c), Pc)
                As_c === nothing && continue
                for d in 1:n_dirs
                    C[eq_i, As_c + (d-1)*nlm] += dir_vecs[i, d] * R_s[b] * R_s[c] * gwJ
                end
            end
        end
    end

    # ── C: slave disp rows, master multiplier cols  −M^(sm) ──────────────────
    for b in 1:nsen_s
        cp_s_b = slave_cps[b]
        for i in 1:ned
            eq_i = ID[i, cp_s_b]
            eq_i == 0 && continue
            for c in 1:nsen_m
                cp_m_c = master_cps[c]
                Am_c   = findfirst(==(cp_m_c), Pc)
                Am_c === nothing && continue
                for d in 1:n_dirs
                    C[eq_i, Am_c + (d-1)*nlm] -= dir_vecs[i, d] * R_s[b] * R_m[c] * gwJ
                end
            end
        end
    end

    # ── Z: -(ε/2)(R_s + R_m)(R_s + R_m)^T  (penalizes flux sum λ^s + λ̄^m) ─
    for b in 1:nsen_s
        cp_s_b = slave_cps[b]
        As_b   = findfirst(==(cp_s_b), Pc)
        As_b === nothing && continue

        # Z: slave-slave  [-ε/2] per direction
        for c in 1:nsen_s
            cp_s_c = slave_cps[c]
            As_c   = findfirst(==(cp_s_c), Pc)
            As_c === nothing && continue
            for d in 1:n_dirs
                Z[As_b + (d-1)*nlm, As_c + (d-1)*nlm] -= 0.5 * epss * R_s[b] * R_s[c] * gwJ
            end
        end

        # Z: slave-master  [-ε/2] per direction
        for c in 1:nsen_m
            cp_m_c = master_cps[c]
            Am_c   = findfirst(==(cp_m_c), Pc)
            Am_c === nothing && continue
            for d in 1:n_dirs
                Z[As_b + (d-1)*nlm, Am_c + (d-1)*nlm] -= 0.5 * epss * R_s[b] * R_m[c] * gwJ
            end
        end
    end

    # ── Z: master rows ──────────────────────────────────────────────────────
    for bm in 1:nsen_m
        cp_m_b = master_cps[bm]
        Am_b   = findfirst(==(cp_m_b), Pc)
        Am_b === nothing && continue

        # Z: master-slave  [-ε/2] per direction
        for c in 1:nsen_s
            cp_s_c = slave_cps[c]
            As_c   = findfirst(==(cp_s_c), Pc)
            As_c === nothing && continue
            for d in 1:n_dirs
                Z[Am_b + (d-1)*nlm, As_c + (d-1)*nlm] -= 0.5 * epss * R_m[bm] * R_s[c] * gwJ
            end
        end

        # Z: master-master  [-ε/2] per direction
        for cm in 1:nsen_m
            cp_m_c = master_cps[cm]
            Am_c   = findfirst(==(cp_m_c), Pc)
            Am_c === nothing && continue
            for d in 1:n_dirs
                Z[Am_b + (d-1)*nlm, Am_c + (d-1)*nlm] -= 0.5 * epss * R_m[bm] * R_m[cm] * gwJ
            end
        end
    end
end

# ─── Dual-Pass (Puso-Solberg) kernel ─────────────────────────────────────────
"""
Dual-pass mortar kernel (Puso & Solberg 2020, Eqs. 8 & 13).

KKT system (paper convention): [K, Cᵀ; C, Z]

C = [D⁽¹⁾, −M⁽²¹⁾; −M⁽¹²⁾, D⁽²⁾]   (multiplier rows × displacement cols)
  Pass s assembles row s: +D⁽ˢ⁾ (slave×slave) and −M⁽ᵐˢ⁾ (slave×master).
  In C_code (= Cᵀ): D block → slave disp rows × slave mult cols,
                     M block → master disp rows × slave mult cols.

Z = ε·[D⁽¹⁾, M⁽²¹⁾; M⁽¹²⁾, D⁽²⁾]   (positive definite)
  Z_code = −Z (solver uses [K,C;C',−Z]).
  Pass s assembles slave mult rows: −ε·D⁽ˢ⁾ and −ε·M⁽ˢᵐ⁾.

ε mapping to P&S parameter: ε = α·E·h / 2  (§4.3: γ = α·E).
"""
function _accumulate_mortar_dp!(
    C, Z,
    R_s::AbstractVector{Float64}, R_m::AbstractVector{Float64},
    slave_cps::AbstractVector{Int}, master_cps::AbstractVector{Int},
    dir_vecs::AbstractMatrix{Float64},
    Pc::AbstractVector{Int}, nlm::Int,
    ID::AbstractMatrix{Int}, ned::Int,
    gwJ::Float64, epss::Float64
)
    nsen_s = length(slave_cps)
    nsen_m = length(master_cps)
    n_dirs = size(dir_vecs, 2)

    # ── C: two blocks per pass ──────────────────────────────────────────────
    #  Pass s (slave=s, master=m, integration on Γ^s) assembles C_paper row s:
    #    C_paper[λ^s, U^s] = +D^(s)    → C_code[U^s, λ^s] += D   (slave disp, slave mult)
    #    C_paper[λ^s, U^m] = −M^(ms)   → C_code[U^m, λ^s] -= M^T (master disp, slave mult)
    #
    # D^(s) block: slave disp rows ← slave multiplier cols
    for b in 1:nsen_s
        cp_s_b = slave_cps[b]
        for i in 1:ned
            eq_i = ID[i, cp_s_b]
            eq_i == 0 && continue
            for c in 1:nsen_s
                cp_s_c = slave_cps[c]
                As_c   = findfirst(==(cp_s_c), Pc)
                As_c === nothing && continue
                for d in 1:n_dirs
                    C[eq_i, As_c + (d-1)*nlm] += dir_vecs[i, d] * R_s[b] * R_s[c] * gwJ
                end
            end
        end
    end
    # −M^T block: master disp rows ← slave multiplier cols
    for b in 1:nsen_m
        cp_m_b = master_cps[b]
        for i in 1:ned
            eq_i = ID[i, cp_m_b]
            eq_i == 0 && continue
            for c in 1:nsen_s
                cp_s_c = slave_cps[c]
                As_c   = findfirst(==(cp_s_c), Pc)
                As_c === nothing && continue
                for d in 1:n_dirs
                    C[eq_i, As_c + (d-1)*nlm] -= dir_vecs[i, d] * R_m[b] * R_s[c] * gwJ
                end
            end
        end
    end

    # ── Z: stabilization, slave rows only ───────────────────────────────────
    #  Z_paper = ε·[D^(s), M^(sm); M^(ms), D^(m)]  (positive definite)
    #  Z_code  = −Z_paper  (solver uses [K,C; C',−Z])
    #  Pass s assembles row s: Z_code[λ^s, λ^s] = −ε·D^s, Z_code[λ^s, λ^m] = −ε·M^sm
    for b in 1:nsen_s
        cp_s_b = slave_cps[b]
        As_b   = findfirst(==(cp_s_b), Pc)
        As_b === nothing && continue

        # Z_ss: −ε·D^s
        for c in 1:nsen_s
            cp_s_c = slave_cps[c]
            As_c   = findfirst(==(cp_s_c), Pc)
            As_c === nothing && continue
            for d in 1:n_dirs
                Z[As_b + (d-1)*nlm, As_c + (d-1)*nlm] -= epss * R_s[b] * R_s[c] * gwJ
            end
        end

        # Z_sm: −ε·M^sm
        for c in 1:nsen_m
            cp_m_c = master_cps[c]
            Am_c   = findfirst(==(cp_m_c), Pc)
            Am_c === nothing && continue
            for d in 1:n_dirs
                Z[As_b + (d-1)*nlm, Am_c + (d-1)*nlm] -= epss * R_s[b] * R_m[c] * gwJ
            end
        end
    end
    # Note: master rows (Z_ms, Z_mm) contributed by the second pass.
end

# ─── Element-based integration (default) ─────────────────────────────────────

function _assemble_pair!(
    C, Z, ::ElementBasedIntegration, formulation::FormulationStrategy,
    Pc, nlm, ps, ns, KVs, Ps, nsen_s, nsel_s, norm_sign_s, IEN_s, INC_s,
    pm, nm, KVm, Pm, norm_sign_m,
    B, ID, ned, nsd, npd, NQUAD, epss
)
    GPW = gauss_product(NQUAD, npd - 1)
    # Global Cartesian directions.
    # TwinMortar / DualPass: plain identity (two-pass C provides coupling;
    #   both D and M must keep their relative signs per pass).
    # SinglePass: scaled by norm_sign_s for consistent half-pass signs.
    dir_vecs = formulation isa SinglePassFormulation ?
        norm_sign_s * Matrix{Float64}(I, nsd, nsd) :
        Matrix{Float64}(I, nsd, nsd)

    for sel in 1:nsel_s
        anchor = IEN_s[sel, 1]
        n0_s   = INC_s[anchor]

        for (gp, gw) in GPW
            R_s, _, dx_dXi, detJ_s, _ = shape_function(
                ps, ns, KVs, B, Ps, gp, nsen_s, nsd, npd - 1,
                sel, n0_s, IEN_s, INC_s
            )

            gwJ = gw * detJ_s

            # Physical point on slave surface
            x_s = zeros(nsd)
            for a in 1:nsen_s
                x_s .+= R_s[a] .* B[Ps[IEN_s[sel, a]], 1:nsd]
            end

            if npd == 2
                # ── 2D: 1D boundary curve ─────────────────────────────────
                kv1  = KVs[1]
                ξ_s  = 0.5*(kv1[n0_s[1]+1] + kv1[n0_s[1]]) +
                        0.5*(kv1[n0_s[1]+1] - kv1[n0_s[1]]) * gp[1]
                ξ_m, _, _, R_m, span_m = closest_point_1d(
                    clamp(ξ_s, 0.0, 1.0), x_s, pm[1], nm[1], KVm[1], B, Pm, nsd
                )
                (ξ_m > 1.0+1e-10 || ξ_m < -1e-10) && continue

                slave_cps  = [Ps[IEN_s[sel, b]]          for b in 1:nsen_s]
                master_cps = [Pm[span_m - pm[1] + c - 1] for c in 1:(pm[1]+1)]

            else
                # ── 3D: 2D boundary surface ───────────────────────────────
                kv1, kv2 = KVs[1], KVs[2]
                ξ_s = 0.5*(kv1[n0_s[1]+1]+kv1[n0_s[1]]) +
                       0.5*(kv1[n0_s[1]+1]-kv1[n0_s[1]]) * gp[1]
                η_s = 0.5*(kv2[n0_s[2]+1]+kv2[n0_s[2]]) +
                       0.5*(kv2[n0_s[2]+1]-kv2[n0_s[2]]) * gp[2]

                ξ_m, η_m, _, _, _, R_m, spans_m = closest_point_2d(
                    clamp(ξ_s, 0.0, 1.0), clamp(η_s, 0.0, 1.0),
                    x_s, pm, nm, KVm, B, Pm, nsd
                )
                (ξ_m > 1.0+1e-10 || ξ_m < -1e-10 ||
                 η_m > 1.0+1e-10 || η_m < -1e-10) && continue

                slave_cps = [Ps[IEN_s[sel, b]] for b in 1:nsen_s]
                span_ξm, span_ηm = spans_m[1], spans_m[2]
                master_cps = Int[]
                for b in 1:(pm[2]+1)
                    η_loc = span_ηm - pm[2] + b - 1
                    for a in 1:(pm[1]+1)
                        ξ_loc = span_ξm - pm[1] + a - 1
                        push!(master_cps, Pm[(η_loc-1)*nm[1] + ξ_loc])
                    end
                end
            end

            _accumulate_mortar!(C, Z, formulation, R_s, R_m, slave_cps, master_cps,
                                dir_vecs, Pc, nlm, ID, ned, gwJ, epss)
        end
    end
end

# ─── Segment-based integration ────────────────────────────────────────────────

function _assemble_pair!(
    C, Z, ::SegmentBasedIntegration, formulation::FormulationStrategy,
    Pc, nlm, ps, ns, KVs, Ps, nsen_s, nsel_s, norm_sign_s, IEN_s, INC_s,
    pm, nm, KVm, Pm, norm_sign_m,
    B, ID, ned, nsd, npd, NQUAD, epss
)
    dir_vecs = formulation isa SinglePassFormulation ?
        norm_sign_s * Matrix{Float64}(I, nsd, nsd) :
        Matrix{Float64}(I, nsd, nsd)

    if npd == 2
        # ── 2D problem: 1D segment-based integration (knot-span intersections) ──

        ξ_breaks = find_interface_segments_1d(
            ps[1], ns[1], KVs[1], pm[1], nm[1], KVm[1], B, Ps, Pm, nsd
        )

        pts1d, wts1d = gauss_rule(NQUAD)
        nsen_m_local = pm[1] + 1

        for k in 1:(length(ξ_breaks) - 1)
            ξ_a, ξ_b = ξ_breaks[k], ξ_breaks[k+1]
            ξ_b - ξ_a < 1e-14 && continue

            dξ = 0.5 * (ξ_b - ξ_a)

            for q in 1:NQUAD
                ξ_s = 0.5 * (ξ_a + ξ_b) + dξ * pts1d[q]

                x_s, dxdξ_s, R_s, span_s = eval_boundary_point(
                    ξ_s, ps[1], ns[1], KVs[1], B, Ps, nsd
                )
                detJ_s = norm(dxdξ_s)
                gwJ    = wts1d[q] * dξ * detJ_s

                ξ_m, _, _, R_m, span_m = closest_point_1d(
                    clamp(ξ_s, 0.0, 1.0), x_s, pm[1], nm[1], KVm[1], B, Pm, nsd
                )
                (ξ_m > 1.0 + 1e-10 || ξ_m < -1e-10) && continue

                slave_cps  = [Ps[span_s - ps[1] + b - 1] for b in 1:(ps[1]+1)]
                master_cps = [Pm[span_m - pm[1] + c - 1] for c in 1:nsen_m_local]

                _accumulate_mortar!(C, Z, formulation, R_s, R_m, slave_cps, master_cps,
                                    dir_vecs, Pc, nlm, ID, ned, gwJ, epss)
            end
        end

    else
        # ── 3D problem: 2D segment-based integration (Sutherland-Hodgman clipping) ──

        cells = find_interface_segments_2d(ps, ns, KVs, pm, nm, KVm, B, Ps, Pm, nsd)
        tri_pts, tri_wts = tri_gauss_rule(NQUAD)
        npts = length(tri_wts)

        for cell in cells
            v1 = cell.verts[:, 1]
            v2 = cell.verts[:, 2]
            v3 = cell.verts[:, 3]
            J_tri = norm(cross(v2 .- v1, v3 .- v1)) / 2.0
            J_tri < 1e-20 && continue

            for q in 1:npts
                ξ_ref = tri_pts[:, q]
                w_q   = tri_wts[q]

                N1 = 1.0 - ξ_ref[1] - ξ_ref[2]
                N2 = ξ_ref[1]
                N3 = ξ_ref[2]
                x_q = N1 .* v1 .+ N2 .* v2 .+ N3 .* v3

                ξ_s, η_s, _, _, _, R_s, spans_s = closest_point_2d(
                    cell.ξ0_s, cell.η0_s, x_q, ps, ns, KVs, B, Ps, nsd
                )
                (ξ_s > 1.0 + 1e-10 || ξ_s < -1e-10 ||
                 η_s > 1.0 + 1e-10 || η_s < -1e-10) && continue

                ξ_m, η_m, _, _, _, R_m, spans_m = closest_point_2d(
                    cell.ξ0_m, cell.η0_m, x_q, pm, nm, KVm, B, Pm, nsd
                )
                (ξ_m > 1.0 + 1e-10 || ξ_m < -1e-10 ||
                 η_m > 1.0 + 1e-10 || η_m < -1e-10) && continue

                # Active slave CPs from converged span
                span_ξs, span_ηs = spans_s[1], spans_s[2]
                slave_cps = Int[]
                for b in 1:(ps[2]+1)
                    η_loc = span_ηs - ps[2] + b - 1
                    for a in 1:(ps[1]+1)
                        ξ_loc = span_ξs - ps[1] + a - 1
                        push!(slave_cps, Ps[(η_loc - 1)*ns[1] + ξ_loc])
                    end
                end

                # Active master CPs from converged span
                span_ξm, span_ηm = spans_m[1], spans_m[2]
                master_cps = Int[]
                for b in 1:(pm[2]+1)
                    η_loc = span_ηm - pm[2] + b - 1
                    for a in 1:(pm[1]+1)
                        ξ_loc = span_ξm - pm[1] + a - 1
                        push!(master_cps, Pm[(η_loc - 1)*nm[1] + ξ_loc])
                    end
                end

                gwJ = w_q * 2.0 * J_tri

                _accumulate_mortar!(C, Z, formulation, R_s, R_m, slave_cps, master_cps,
                                    dir_vecs, Pc, nlm, ID, ned, gwJ, epss)
            end
        end
    end
end

# ─── Main entry point ─────────────────────────────────────────────────────────

"""
    build_mortar_coupling(Pc, pairs, p, n, KV, P, B, ID, nnp,
                          ned, nsd, npd, neq, NQUAD, epss,
                          strategy = ElementBasedIntegration()) -> (C, Z)

Assemble the Twin Mortar coupling matrix C (neq × 2·nlm) and stabilization
matrix Z (2·nlm × 2·nlm), where nlm = length(Pc).

`strategy` selects the mortar integration method:
  - `ElementBasedIntegration()` (default): Gauss points per slave element,
    projected onto master via closest-point Newton iteration.
  - `SegmentBasedIntegration()`: integration over slave/master knot-span
    intersection segments; both shape function families are smooth (single
    Bezier span) within every integration cell.

`pairs` contains one entry per half-pass. For full twin mortar tying, supply
both directions:
    pairs = [InterfacePair(1, s_facet, 2, m_facet),
             InterfacePair(2, m_facet, 1, s_facet)]

Twin Mortar formulation (per half-pass s, integrated on Γ^(s)):
    δΠ_C = ∫_Γ^(s) (δu^s − δū^m)·λ^s dΓ           (coupling)
    δΠ_λ = ∫_Γ^(s) δλ^s·(u^s − ū^m) dΓ + δΠ_ε     (constraint)
    δΠ_ε = Σ_s (ε/2)∫_Γ^(s) (δλ^s + δλ̄^m)·(λ^s + λ̄^m) dΓ  (stabilization)

→ C has slave AND master displacement rows per half-pass:
  slave rows: +D^(s) (self-coupling), master rows: −M^(sm)^T (cross-coupling)
→ Z = -(ε/2)(Rs + Rm)(Rs + Rm)^T  (penalizes flux sum λ^s + λ̄^m)

The system assembled is:
    [K    C ] [d]   [F_ext]
    [C^T  -Z] [λ] = [  0  ]
where λ = [λ_x; λ_y; λ_z] are Cartesian Lagrange multipliers at Pc CPs.
"""
function build_mortar_coupling(
    Pc::AbstractVector{Int},
    pairs::Vector{InterfacePair},
    p::Matrix{Int}, n::Matrix{Int},
    KV::Vector{<:AbstractVector{<:AbstractVector{Float64}}},
    P::Vector{Vector{Int}},
    B::AbstractMatrix{Float64},
    ID::AbstractMatrix{Int},
    nnp::Vector{Int},
    ned::Int, nsd::Int, npd::Int, neq::Int,
    NQUAD::Int, epss::Float64,
    strategy::IntegrationStrategy    = ElementBasedIntegration(),
    formulation::FormulationStrategy = TwinMortarFormulation(),
    normal_strategy::NormalStrategy  = SlaveNormal()   # retained for API compat
)
    nlm   = length(Pc)
    ndirs = nsd              # nsd global Cartesian directions
    C     = spzeros(Float64, neq,   ndirs * nlm)
    Z     = spzeros(Float64, ndirs * nlm, ndirs * nlm)

    for pair in pairs
        spc    = pair.slave_patch
        sfacet = pair.slave_facet
        mpc    = pair.master_patch
        mfacet = pair.master_facet

        # ── Slave boundary data ──────────────────────────────────────────────
        ps, ns, KVs, Ps, nsn_s, nsen_s, nsel_s, norm_sign_s, _, _, _ =
            get_segment_patch(p[spc, :], n[spc, :], KV[spc], P[spc], npd, nnp[spc], sfacet)

        IEN_s_vec = build_ien(nsd, npd - 1, 1,
                              reshape(ps, 1, :), reshape(ns, 1, :),
                              [nsel_s], [nsn_s], [nsen_s])
        IEN_s = IEN_s_vec[1]
        INC_s = build_inc(ns)

        # ── Master boundary data ─────────────────────────────────────────────
        pm, nm, KVm, Pm, _, nsen_m, _, norm_sign_m, _, _, _ =
            get_segment_patch(p[mpc, :], n[mpc, :], KV[mpc], P[mpc], npd, nnp[mpc], mfacet)

        # ── Dispatch to integration + formulation ─────────────────────────
        _assemble_pair!(C, Z, strategy, formulation,
                        Pc, nlm, ps, ns, KVs, Ps, nsen_s, nsel_s, norm_sign_s, IEN_s, INC_s,
                        pm, nm, KVm, Pm, norm_sign_m,
                        B, ID, ned, nsd, npd, NQUAD, epss)
    end

    return C, Z
end

# ─── Mortar mass matrices for mesh modification ──────────────────────────────

"""
    build_mortar_mass_matrices(pair, p, n, KV, P, B, nnp, nsd, npd, NQUAD,
                               strategy=ElementBasedIntegration()) -> (D, M, slave_ifc_cps, master_ifc_cps)

Assemble the scalar mortar mass matrices for a single interface pair:
  D_{ij} = ∫_Γ R_s^i R_s^j dΓ   (slave-slave, square)
  M_{ij} = ∫_Γ R_s^i R_m^j dΓ   (slave-master, rectangular)

Returns D, M as sparse matrices, plus the global CP indices for slave and
master interface CPs (in the order used by D and M).

Used for mesh modification (Puso 2004): X_new^(s) = D^{-1} M X^(m)
to eliminate the L² geometric gap before solving.
"""
function build_mortar_mass_matrices(
    pair::InterfacePair,
    p::Matrix{Int}, n::Matrix{Int},
    KV::Vector{<:AbstractVector{<:AbstractVector{Float64}}},
    P::Vector{Vector{Int}},
    B::AbstractMatrix{Float64},
    nnp::Vector{Int},
    nsd::Int, npd::Int, NQUAD::Int,
    strategy::IntegrationStrategy = ElementBasedIntegration()
)
    spc    = pair.slave_patch
    sfacet = pair.slave_facet
    mpc    = pair.master_patch
    mfacet = pair.master_facet

    # Slave boundary data
    ps, ns, KVs, Ps, nsn_s, nsen_s, nsel_s, norm_sign_s, _, _, _ =
        get_segment_patch(p[spc,:], n[spc,:], KV[spc], P[spc], npd, nnp[spc], sfacet)

    IEN_s_vec = build_ien(nsd, npd-1, 1,
                          reshape(ps,1,:), reshape(ns,1,:),
                          [nsel_s], [nsn_s], [nsen_s])
    IEN_s = IEN_s_vec[1]
    INC_s = build_inc(ns)

    # Master boundary data
    pm, nm, KVm, Pm, _, nsen_m, _, norm_sign_m, _, _, _ =
        get_segment_patch(p[mpc,:], n[mpc,:], KV[mpc], P[mpc], npd, nnp[mpc], mfacet)

    # Collect unique interface CPs
    slave_ifc_cps  = sort(unique(Ps[IEN_s[:]]))
    master_ifc_cps = sort(unique(Pm))

    n_s = length(slave_ifc_cps)
    n_m = length(master_ifc_cps)

    # Maps from global CP → local index in D/M
    s_map = Dict(cp => i for (i, cp) in enumerate(slave_ifc_cps))
    m_map = Dict(cp => i for (i, cp) in enumerate(master_ifc_cps))

    D = spzeros(Float64, n_s, n_s)
    M_mat = spzeros(Float64, n_s, n_m)

    _assemble_mass_pair!(D, M_mat, strategy,
                          ps, ns, KVs, Ps, nsen_s, nsel_s, norm_sign_s, IEN_s, INC_s,
                          pm, nm, KVm, Pm, norm_sign_m,
                          B, nsd, npd, NQUAD, s_map, m_map)

    return D, M_mat, slave_ifc_cps, master_ifc_cps
end

"""
Element-based assembly of scalar mortar mass matrices D and M.
"""
function _assemble_mass_pair!(
    D, M_mat, ::ElementBasedIntegration,
    ps, ns, KVs, Ps, nsen_s, nsel_s, norm_sign_s, IEN_s, INC_s,
    pm, nm, KVm, Pm, norm_sign_m,
    B, nsd, npd, NQUAD, s_map, m_map
)
    GPW = gauss_product(NQUAD, npd - 1)

    for sel in 1:nsel_s
        anchor = IEN_s[sel, 1]
        n0_s   = INC_s[anchor]

        for (gp, gw) in GPW
            R_s, _, dx_dXi, detJ_s, n_vec_s = shape_function(
                ps, ns, KVs, B, Ps, gp, nsen_s, nsd, npd - 1,
                sel, n0_s, IEN_s, INC_s)
            detJ_s <= 0.0 && continue
            gwJ = gw * detJ_s

            x_s = zeros(nsd)
            for a in 1:nsen_s
                x_s .+= R_s[a] .* B[Ps[IEN_s[sel, a]], 1:nsd]
            end

            if npd == 2
                kv1 = KVs[1]
                ξ_s = 0.5*(kv1[n0_s[1]+1]+kv1[n0_s[1]]) +
                      0.5*(kv1[n0_s[1]+1]-kv1[n0_s[1]]) * gp[1]
                ξ_m, _, _, R_m, span_m = closest_point_1d(
                    clamp(ξ_s, 0.0, 1.0), x_s, pm[1], nm[1], KVm[1], B, Pm, nsd)
                (ξ_m > 1.0+1e-10 || ξ_m < -1e-10) && continue

                slave_cps  = [Ps[IEN_s[sel, b]]          for b in 1:nsen_s]
                master_cps = [Pm[span_m - pm[1] + c - 1] for c in 1:(pm[1]+1)]
            else
                kv1, kv2 = KVs[1], KVs[2]
                ξ_s = 0.5*(kv1[n0_s[1]+1]+kv1[n0_s[1]]) +
                      0.5*(kv1[n0_s[1]+1]-kv1[n0_s[1]]) * gp[1]
                η_s = 0.5*(kv2[n0_s[2]+1]+kv2[n0_s[2]]) +
                      0.5*(kv2[n0_s[2]+1]-kv2[n0_s[2]]) * gp[2]

                ξ_m, η_m, _, dxdξ_m, dxdη_m, R_m, spans_m = closest_point_2d(
                    clamp(ξ_s, 0.0, 1.0), clamp(η_s, 0.0, 1.0),
                    x_s, pm, nm, KVm, B, Pm, nsd)
                (ξ_m > 1.0+1e-10 || ξ_m < -1e-10 ||
                 η_m > 1.0+1e-10 || η_m < -1e-10) && continue

                slave_cps = [Ps[IEN_s[sel, b]] for b in 1:nsen_s]
                span_ξm, span_ηm = spans_m[1], spans_m[2]
                master_cps = Int[]
                for b in 1:(pm[2]+1)
                    η_loc = span_ηm - pm[2] + b - 1
                    for a in 1:(pm[1]+1)
                        ξ_loc = span_ξm - pm[1] + a - 1
                        push!(master_cps, Pm[(η_loc-1)*nm[1] + ξ_loc])
                    end
                end
            end

            # Accumulate D = ∫ R_s R_s^T dΓ
            for b in 1:length(slave_cps)
                ib = get(s_map, slave_cps[b], 0)
                ib == 0 && continue
                for c in 1:length(slave_cps)
                    ic = get(s_map, slave_cps[c], 0)
                    ic == 0 && continue
                    D[ib, ic] += R_s[b] * R_s[c] * gwJ
                end
            end

            # Accumulate M = ∫ R_s R_m^T dΓ
            for b in 1:length(slave_cps)
                ib = get(s_map, slave_cps[b], 0)
                ib == 0 && continue
                for c in 1:length(master_cps)
                    jc = get(m_map, master_cps[c], 0)
                    jc == 0 && continue
                    M_mat[ib, jc] += R_s[b] * R_m[c] * gwJ
                end
            end
        end
    end
end

"""
Segment-based assembly of scalar mortar mass matrices D and M (1D interfaces only).
Uses knot-span intersection segments for exact integration.
"""
function _assemble_mass_pair!(
    D, M_mat, ::SegmentBasedIntegration,
    ps, ns, KVs, Ps, nsen_s, nsel_s, norm_sign_s, IEN_s, INC_s,
    pm, nm, KVm, Pm, norm_sign_m,
    B, nsd, npd, NQUAD, s_map, m_map
)
    npd == 2 || error("Segment-based _assemble_mass_pair! only implemented for 2D (1D interfaces)")

    ξ_breaks = find_interface_segments_1d(
        ps[1], ns[1], KVs[1], pm[1], nm[1], KVm[1], B, Ps, Pm, nsd
    )

    pts1d, wts1d = gauss_rule(NQUAD)
    nsen_m_local = pm[1] + 1

    for k in 1:(length(ξ_breaks) - 1)
        ξ_a, ξ_b = ξ_breaks[k], ξ_breaks[k+1]
        ξ_b - ξ_a < 1e-14 && continue

        dξ = 0.5 * (ξ_b - ξ_a)

        for q in 1:NQUAD
            ξ_s = 0.5 * (ξ_a + ξ_b) + dξ * pts1d[q]

            x_s, dxdξ_s, R_s, span_s = eval_boundary_point(
                ξ_s, ps[1], ns[1], KVs[1], B, Ps, nsd
            )
            detJ_s = norm(dxdξ_s)
            gwJ    = wts1d[q] * dξ * detJ_s

            ξ_m, _, _, R_m, span_m = closest_point_1d(
                clamp(ξ_s, 0.0, 1.0), x_s, pm[1], nm[1], KVm[1], B, Pm, nsd
            )
            (ξ_m > 1.0 + 1e-10 || ξ_m < -1e-10) && continue

            slave_cps  = [Ps[span_s - ps[1] + b - 1] for b in 1:(ps[1]+1)]
            master_cps = [Pm[span_m - pm[1] + c - 1] for c in 1:nsen_m_local]

            for b in 1:length(slave_cps)
                ib = get(s_map, slave_cps[b], 0)
                ib == 0 && continue
                for c in 1:length(slave_cps)
                    ic = get(s_map, slave_cps[c], 0)
                    ic == 0 && continue
                    D[ib, ic] += R_s[b] * R_s[c] * gwJ
                end
            end

            for b in 1:length(slave_cps)
                ib = get(s_map, slave_cps[b], 0)
                ib == 0 && continue
                for c in 1:length(master_cps)
                    jc = get(m_map, master_cps[c], 0)
                    jc == 0 && continue
                    M_mat[ib, jc] += R_s[b] * R_m[c] * gwJ
                end
            end
        end
    end
end
