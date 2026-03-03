# MortarAssembly.jl
# Twin Mortar coupling matrix assembly for non-conforming mesh tying.
# Ported from MATLAB: buildTwoHalfPassLagrangeTying.m

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
    build_interface_cps(pairs, p, n, KV, P, npd, nnp) -> Pc

Collect all boundary control point indices from all slave and master facets
of the given interface pairs. Returns the unique sorted union (global CP indices).
"""
function build_interface_cps(
    pairs::Vector{InterfacePair},
    p::Matrix{Int}, n::Matrix{Int},
    KV::Vector{<:AbstractVector{<:AbstractVector{Float64}}},
    P::Vector{Vector{Int}},
    npd::Int, nnp::Vector{Int}
)::Vector{Int}
    cps = Int[]
    seen = Set{Tuple{Int,Int}}()   # (patch, facet) pairs already added
    for pair in pairs
        for (pc, facet) in [(pair.slave_patch, pair.slave_facet),
                            (pair.master_patch, pair.master_facet)]
            (pc, facet) in seen && continue
            push!(seen, (pc, facet))
            _, _, _, Ps, _, _, _, _, _, _, _ = get_segment_patch(
                p[pc, :], n[pc, :], KV[pc], P[pc], npd, nnp[pc], facet
            )
            append!(cps, Ps)
        end
    end
    return unique(sort(cps))
end

"""
    build_mortar_coupling(Pc, pairs, p, n, KV, P, B, ID, nnp,
                          ned, nsd, npd, neq, NQUAD, epss) -> (C, Z)

Assemble the Twin Mortar coupling matrix C (neq × 2·nlm) and stabilization
matrix Z (2·nlm × 2·nlm), where nlm = length(Pc).

`pairs` contains one entry per half-pass. For full twin mortar tying, supply
both directions:
    pairs = [InterfacePair(1, s_facet, 2, m_facet),
             InterfacePair(2, m_facet, 1, s_facet)]

Symmetric stabilization (per half-pass s):
    Π_λ = ∫_Γ^(s) [ (δλ^s + δλ^m)·(ε/2)(λ^s + λ^m)
                    - δλ^s·(1/2)(u^m + u^s) ] dS

→ Z = (ε/2)(Rs+Rm)(Rs+Rm)^T  (symmetric, positive semidefinite)
→ C includes both slave and master displacement contributions (½ each)

The system assembled is:
    [K    C ] [d]   [F_ext]
    [C^T  -Z] [λ] = [  0  ]
where λ = [λ_n; λ_t] are normal + tangential Lagrange multipliers at Pc CPs.
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
    NQUAD::Int, epss::Float64
)
    nlm = length(Pc)
    C   = spzeros(Float64, neq, 2 * nlm)
    Z   = spzeros(Float64, 2 * nlm, 2 * nlm)

    GPW = gauss_product(NQUAD, npd - 1)

    for pair in pairs
        spc    = pair.slave_patch
        sfacet = pair.slave_facet
        mpc    = pair.master_patch
        mfacet = pair.master_facet

        # ── Slave boundary data ─────────────────────────────────────────────
        ps, ns, KVs, Ps, nsn_s, nsen_s, nsel_s, norm_sign_s, _, _, _ =
            get_segment_patch(p[spc, :], n[spc, :], KV[spc], P[spc], npd, nnp[spc], sfacet)

        IEN_s_vec = build_ien(nsd, npd - 1, 1,
                              reshape(ps, 1, :), reshape(ns, 1, :),
                              [nsel_s], [nsn_s], [nsen_s])
        IEN_s = IEN_s_vec[1]
        INC_s = build_inc(ns)

        # ── Master boundary data ────────────────────────────────────────────
        pm, nm, KVm, Pm, _, nsen_m, _, _, _, _, _ =
            get_segment_patch(p[mpc, :], n[mpc, :], KV[mpc], P[mpc], npd, nnp[mpc], mfacet)

        # ── Gauss loop over slave boundary elements ─────────────────────────
        for sel in 1:nsel_s
            anchor = IEN_s[sel, 1]
            n0_s   = INC_s[anchor]

            for (gp, gw) in GPW
                # Slave shape functions, Jacobian, and outward normal
                R_s, _, _, detJ_s, n_vec_s = shape_function(
                    ps, ns, KVs, B, Ps, gp, nsen_s, nsd, npd - 1,
                    sel, n0_s, IEN_s, INC_s
                )
                n_vec_s .*= norm_sign_s        # apply outward-normal sign
                t_vec_s   = [n_vec_s[2], -n_vec_s[1]]   # 90° CW tangent

                gwJ = gw * detJ_s   # thickness = 1.0

                # Physical point on slave surface
                x_s = zeros(nsd)
                for a in 1:nsen_s
                    x_s .+= R_s[a] .* B[Ps[IEN_s[sel, a]], 1:nsd]
                end

                # Initial guess for closest-point: same η-coordinate as slave GP
                kv1  = KVs[1]
                ξ_s  = 0.5 * (kv1[n0_s[1]+1] + kv1[n0_s[1]]) +
                        0.5 * (kv1[n0_s[1]+1] - kv1[n0_s[1]]) * gp[1]
                ξ_m0 = clamp(ξ_s, 0.0, 1.0)

                # Closest-point projection onto master boundary
                ξ_m, _, _, R_m, span_m = closest_point_1d(
                    ξ_m0, x_s, pm[1], nm[1], KVm[1], B, Pm, nsd
                )

                # Skip Gauss points whose projection falls outside the master
                (ξ_m > 1.0 + 1e-10 || ξ_m < -1e-10) && continue

                # ── Assemble C and Z ────────────────────────────────────────
                # Symmetric formulation:
                #   Z  = (ε/2)(Rs+Rm)(Rs+Rm)^T  [positive semidefinite, symmetric]

                # --- slave rows ---
                for b in 1:nsen_s
                    cp_s_b = Ps[IEN_s[sel, b]]
                    As_b   = findfirst(==(cp_s_b), Pc)   # row index for Z

                    # C: displacement DOF (b) ← multiplier at each slave CP (c)
                    # C[eq_i(b), As_c] -= n_s[i] * R_s[b] * R_s[c] * gwJ  (mortar mass matrix)
                    for i in 1:ned
                        eq_i = ID[i, cp_s_b]
                        eq_i == 0 && continue
                        for c in 1:nsen_s
                            cp_s_c = Ps[IEN_s[sel, c]]
                            As_c   = findfirst(==(cp_s_c), Pc)
                            As_c === nothing && continue
                            C[eq_i, As_c]       -= n_vec_s[i] * R_s[b] * R_s[c] * gwJ
                            C[eq_i, As_c + nlm] -= t_vec_s[i] * R_s[b] * R_s[c] * gwJ
                        end
                    end

                    As_b === nothing && continue   # Z rows require a valid As_b

                    # Z: slave-slave  [+ε/2, sign flipped from old formulation]
                    for c in 1:nsen_s
                        cp_s2 = Ps[IEN_s[sel, c]]
                        As_c  = findfirst(==(cp_s2), Pc)
                        As_c === nothing && continue
                        Z[As_b,       As_c]       -= 0.5 * epss * R_s[b] * R_s[c] * gwJ
                        Z[As_b + nlm, As_c + nlm] -= 0.5 * epss * R_s[b] * R_s[c] * gwJ
                    end

                    # Z: slave-master  [+ε/2]
                    for c in 1:(pm[1] + 1)
                        local_m = span_m - pm[1] + c - 1   # 1-based index into Pm
                        cp_m    = Pm[local_m]
                        Am_c    = findfirst(==(cp_m), Pc)
                        Am_c === nothing && continue
                        Z[As_b,       Am_c]       += 0.5 * epss * R_s[b] * R_m[c] * gwJ
                        Z[As_b + nlm, Am_c + nlm] += 0.5 * epss * R_s[b] * R_m[c] * gwJ
                    end
                end

                # --- master rows (symmetric counterpart) ---
                for bm in 1:(pm[1] + 1)
                    local_m_b = span_m - pm[1] + bm - 1
                    cp_m_b    = Pm[local_m_b]
                    Am_b      = findfirst(==(cp_m_b), Pc)

                    Am_b === nothing && continue   # Z rows require a valid Am_b

                    # Z: master-slave  [+ε/2]
                    for c in 1:nsen_s
                        cp_s_c = Ps[IEN_s[sel, c]]
                        As_c   = findfirst(==(cp_s_c), Pc)
                        As_c === nothing && continue
                        Z[Am_b,       As_c]       += 0.5 * epss * R_m[bm] * R_s[c] * gwJ
                        Z[Am_b + nlm, As_c + nlm] += 0.5 * epss * R_m[bm] * R_s[c] * gwJ
                    end

                    # Z: master-master  [+ε/2]
                    for cm in 1:(pm[1] + 1)
                        local_m_c = span_m - pm[1] + cm - 1
                        cp_m_c    = Pm[local_m_c]
                        Am_c      = findfirst(==(cp_m_c), Pc)
                        Am_c === nothing && continue
                        Z[Am_b,       Am_c]       -= 0.5 * epss * R_m[bm] * R_m[cm] * gwJ
                        Z[Am_b + nlm, Am_c + nlm] -= 0.5 * epss * R_m[bm] * R_m[cm] * gwJ
                    end
                end

            end   # Gauss points
        end   # slave elements
    end   # pairs

    return C, Z
end
