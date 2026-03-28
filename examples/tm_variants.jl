# tm_variants.jl — Compare 4 Twin Mortar assembly variants
#
# Variant 1 (full_two_pass):    Averaged D̄ and M̄ in both C and Z
# Variant 2 (two_half_pass):    Raw per-pass D and M, no averaging
# Variant 3 (collapsed_diag):   D on diagonal, M̄ for off-diagonal
# Variant 4 (halfC_fullZ):      Raw C (variant 2), symmetrized Z (variant 3)
#
# Raw mortar matrices from the two half-passes:
#   Pass 1 (slave=1, master=2, integrated on Γ₁):
#     D1  = ∫_{Γ1} R₁ R₁ᵀ dΓ      (n1×n1)
#     M12 = ∫_{Γ1} R₁ R̄₂ᵀ dΓ      (n1×n2)
#     X12 = ∫_{Γ1} R̄₂ R̄₂ᵀ dΓ     (n2×n2, projected master self-mass)
#   Pass 2 (slave=2, master=1, integrated on Γ₂):
#     D2  = ∫_{Γ2} R₂ R₂ᵀ dΓ      (n2×n2)
#     M21 = ∫_{Γ2} R₂ R̄₁ᵀ dΓ      (n2×n1)
#     X21 = ∫_{Γ2} R̄₁ R̄₁ᵀ dΓ     (n1×n1, projected slave self-mass)
#
# User's notation: D^(12)=X21 (n1×n1), D^(21)=X12 (n2×n2)

using Printf, SparseArrays, LinearAlgebra
include("curved_patch_test.jl")

# ═══════════════════════════════════════════════════════════════════════════════
# Raw matrix assembly for one half-pass
# ═══════════════════════════════════════════════════════════════════════════════

function _raw_mortar_pass(
    pair::InterfacePair, p_mat, n_mat_ref, KV_ref, P_ref, B_ref, nnp, nsd, npd, NQUAD
)
    spc, sfacet = pair.slave_patch, pair.slave_facet
    mpc, mfacet = pair.master_patch, pair.master_facet

    ps, ns, KVs, Ps, nsn_s, nsen_s, nsel_s, _, _, _, _ =
        get_segment_patch(p_mat[spc,:], n_mat_ref[spc,:], KV_ref[spc], P_ref[spc], npd, nnp[spc], sfacet)

    IEN_s = build_ien(nsd, npd-1, 1, reshape(ps,1,:), reshape(ns,1,:),
                      [nsel_s], [nsn_s], [nsen_s])[1]
    INC_s = build_inc(ns)

    pm, nm, KVm, Pm, _, _, _, _, _, _, _ =
        get_segment_patch(p_mat[mpc,:], n_mat_ref[mpc,:], KV_ref[mpc], P_ref[mpc], npd, nnp[mpc], mfacet)

    slave_cps_g  = sort(unique([Ps[IEN_s[e,a]] for e in 1:nsel_s for a in 1:nsen_s]))
    master_cps_g = sort(unique(Pm))

    n_s = length(slave_cps_g)
    n_m = length(master_cps_g)
    s_map = Dict(cp => i for (i,cp) in enumerate(slave_cps_g))
    m_map = Dict(cp => i for (i,cp) in enumerate(master_cps_g))

    D_ss = zeros(n_s, n_s)
    M_sm = zeros(n_s, n_m)
    X_mm = zeros(n_m, n_m)

    GPW = gauss_product(NQUAD, npd-1)

    for sel in 1:nsel_s
        anchor = IEN_s[sel, 1]
        n0_s   = INC_s[anchor]

        for (gp, gw) in GPW
            R_s, _, _, detJ_s, _ = shape_function(
                ps, ns, KVs, B_ref, Ps, gp, nsen_s, nsd, npd-1, sel, n0_s, IEN_s, INC_s)
            detJ_s <= 0.0 && continue
            gwJ = gw * detJ_s

            x_s = zeros(nsd)
            for a in 1:nsen_s; x_s .+= R_s[a] .* B_ref[Ps[IEN_s[sel,a]], 1:nsd]; end

            if npd == 2
                # 2D: 1D boundary
                kv1 = KVs[1]
                ξ_s = 0.5*(kv1[n0_s[1]+1]+kv1[n0_s[1]]) + 0.5*(kv1[n0_s[1]+1]-kv1[n0_s[1]])*gp[1]
                ξ_m, _, _, R_m, span_m = closest_point_1d(
                    clamp(ξ_s, 0.0, 1.0), x_s, pm[1], nm[1], KVm[1], B_ref, Pm, nsd)
                (ξ_m > 1.0+1e-10 || ξ_m < -1e-10) && continue
                s_cps = [Ps[IEN_s[sel,b]] for b in 1:nsen_s]
                m_cps = [Pm[span_m - pm[1] + c - 1] for c in 1:(pm[1]+1)]
            else
                # 3D: 2D boundary surface
                kv1, kv2 = KVs[1], KVs[2]
                ξ_s = 0.5*(kv1[n0_s[1]+1]+kv1[n0_s[1]]) + 0.5*(kv1[n0_s[1]+1]-kv1[n0_s[1]])*gp[1]
                η_s = 0.5*(kv2[n0_s[2]+1]+kv2[n0_s[2]]) + 0.5*(kv2[n0_s[2]+1]-kv2[n0_s[2]])*gp[2]
                ξ_m, η_m, _, _, _, R_m, spans_m = closest_point_2d(
                    clamp(ξ_s,0.0,1.0), clamp(η_s,0.0,1.0), x_s, pm, nm, KVm, B_ref, Pm, nsd)
                (ξ_m > 1.0+1e-10 || ξ_m < -1e-10 || η_m > 1.0+1e-10 || η_m < -1e-10) && continue
                s_cps = [Ps[IEN_s[sel,b]] for b in 1:nsen_s]
                span_ξm, span_ηm = spans_m
                m_cps = Int[]
                for b in 1:(pm[2]+1)
                    η_loc = span_ηm - pm[2] + b - 1
                    for a in 1:(pm[1]+1)
                        ξ_loc = span_ξm - pm[1] + a - 1
                        push!(m_cps, Pm[(η_loc-1)*nm[1] + ξ_loc])
                    end
                end
            end

            for b in eachindex(s_cps), c in eachindex(s_cps)
                ib = get(s_map, s_cps[b], 0); ic = get(s_map, s_cps[c], 0)
                (ib==0 || ic==0) && continue
                D_ss[ib, ic] += R_s[b] * R_s[c] * gwJ
            end
            for b in eachindex(s_cps), c in eachindex(m_cps)
                ib = get(s_map, s_cps[b], 0); jc = get(m_map, m_cps[c], 0)
                (ib==0 || jc==0) && continue
                M_sm[ib, jc] += R_s[b] * R_m[c] * gwJ
            end
            for b in eachindex(m_cps), c in eachindex(m_cps)
                ib = get(m_map, m_cps[b], 0); jc = get(m_map, m_cps[c], 0)
                (ib==0 || jc==0) && continue
                X_mm[ib, jc] += R_m[b] * R_m[c] * gwJ
            end
        end
    end

    return D_ss, M_sm, X_mm, slave_cps_g, master_cps_g
end


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers to expand scalar mortar matrices into global C and Z
# ═══════════════════════════════════════════════════════════════════════════════

function _add_C!(C, Mloc, disp_cps, lm_cps, sign, Pc, nlm, ID, ned, nsd)
    for (li, cp_d) in enumerate(disp_cps)
        for (lj, cp_l) in enumerate(lm_cps)
            val = Mloc[li, lj]; abs(val) < 1e-30 && continue
            gj = findfirst(==(cp_l), Pc); gj === nothing && continue
            for d in 1:nsd
                eq = ID[d, cp_d]; eq == 0 && continue
                C[eq, gj + (d-1)*nlm] += sign * val
            end
        end
    end
end

function _add_Z!(Z, Mloc, row_cps, col_cps, coeff, Pc, nlm, nsd)
    for (li, cp_r) in enumerate(row_cps)
        gi = findfirst(==(cp_r), Pc); gi === nothing && continue
        for (lj, cp_c) in enumerate(col_cps)
            val = Mloc[li, lj]; abs(val) < 1e-30 && continue
            gj = findfirst(==(cp_c), Pc); gj === nothing && continue
            for d in 1:nsd
                Z[gi + (d-1)*nlm, gj + (d-1)*nlm] += coeff * val
            end
        end
    end
end


# ═══════════════════════════════════════════════════════════════════════════════
# Build C and Z from raw matrices for each variant
# ═══════════════════════════════════════════════════════════════════════════════

function build_variant_CZ(
    variant::Symbol,
    D1, M12, X12,    # pass 1: slave=surface1
    D2, M21, X21,    # pass 2: slave=surface2
    cps1, cps2, Pc, nlm, ID, ned, nsd, neq, epss
)
    C = spzeros(Float64, neq, nsd * nlm)
    Z = spzeros(Float64, nsd * nlm, nsd * nlm)

    # User's D^(12) = X21 (n1×n1),  D^(21) = X12 (n2×n2)
    M_bar = 0.5 * (M12 .+ M21')   # symmetrized mortar (n1×n2)

    if variant == :full_two_pass
        D1_bar = 0.5 * (D1 .+ X21)
        D2_bar = 0.5 * (D2 .+ X12)
        _add_C!(C, D1_bar, cps1, cps1, +1.0, Pc, nlm, ID, ned, nsd)
        _add_C!(C, M_bar,  cps1, cps2, -1.0, Pc, nlm, ID, ned, nsd)
        _add_C!(C, M_bar', cps2, cps1, -1.0, Pc, nlm, ID, ned, nsd)
        _add_C!(C, D2_bar, cps2, cps2, +1.0, Pc, nlm, ID, ned, nsd)
        _add_Z!(Z, D1_bar, cps1, cps1, epss, Pc, nlm, nsd)
        _add_Z!(Z, M_bar,  cps1, cps2, epss, Pc, nlm, nsd)
        _add_Z!(Z, M_bar', cps2, cps1, epss, Pc, nlm, nsd)
        _add_Z!(Z, D2_bar, cps2, cps2, epss, Pc, nlm, nsd)

    elseif variant == :two_half_pass
        _add_C!(C, D1,  cps1, cps1, +1.0, Pc, nlm, ID, ned, nsd)
        _add_C!(C, M12, cps1, cps2, -1.0, Pc, nlm, ID, ned, nsd)
        _add_C!(C, M21, cps2, cps1, -1.0, Pc, nlm, ID, ned, nsd)
        _add_C!(C, D2,  cps2, cps2, +1.0, Pc, nlm, ID, ned, nsd)
        _add_Z!(Z, D1,  cps1, cps1, epss, Pc, nlm, nsd)
        _add_Z!(Z, M12, cps1, cps2, epss, Pc, nlm, nsd)
        _add_Z!(Z, M21, cps2, cps1, epss, Pc, nlm, nsd)
        _add_Z!(Z, D2,  cps2, cps2, epss, Pc, nlm, nsd)

    elseif variant == :collapsed_diag
        _add_C!(C, D1,     cps1, cps1, +1.0, Pc, nlm, ID, ned, nsd)
        _add_C!(C, M_bar,  cps1, cps2, -1.0, Pc, nlm, ID, ned, nsd)
        _add_C!(C, M_bar', cps2, cps1, -1.0, Pc, nlm, ID, ned, nsd)
        _add_C!(C, D2,     cps2, cps2, +1.0, Pc, nlm, ID, ned, nsd)
        _add_Z!(Z, D1,     cps1, cps1, epss, Pc, nlm, nsd)
        _add_Z!(Z, M_bar,  cps1, cps2, epss, Pc, nlm, nsd)
        _add_Z!(Z, M_bar', cps2, cps1, epss, Pc, nlm, nsd)
        _add_Z!(Z, D2,     cps2, cps2, epss, Pc, nlm, nsd)

    elseif variant == :halfC_fullZ
        _add_C!(C, D1,  cps1, cps1, +1.0, Pc, nlm, ID, ned, nsd)
        _add_C!(C, M12, cps1, cps2, -1.0, Pc, nlm, ID, ned, nsd)
        _add_C!(C, M21, cps2, cps1, -1.0, Pc, nlm, ID, ned, nsd)
        _add_C!(C, D2,  cps2, cps2, +1.0, Pc, nlm, ID, ned, nsd)
        _add_Z!(Z, D1,     cps1, cps1, epss, Pc, nlm, nsd)
        _add_Z!(Z, M_bar,  cps1, cps2, epss, Pc, nlm, nsd)
        _add_Z!(Z, M_bar', cps2, cps1, epss, Pc, nlm, nsd)
        _add_Z!(Z, D2,     cps2, cps2, epss, Pc, nlm, nsd)
    else
        error("Unknown variant: $variant")
    end

    return C, Z
end


# ═══════════════════════════════════════════════════════════════════════════════
# Driver for CPT (3D curved patch test)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    run_cpt_variants(; p_ord=2, exp_level=0, epss=1e6, ...)

Test 4 TM variants on the 3D curved patch test, reporting RMS stress error.
"""
function run_cpt_variants(;
    p_ord::Int     = 2,
    exp_level::Int = 0,
    epss::Float64  = 0.0,
    NQUAD::Int     = p_ord + 1,
    NQUAD_mortar::Int = p_ord + 4,
    kwargs...
)
    nsd=3; npd=3; ned=3; npc=2

    # ── Geometry setup (same as _cpt_solve) ──────────────────────────────────
    conforming = get(kwargs, :conforming, false)
    L_x = get(kwargs, :L_x, 1.0); L_y = get(kwargs, :L_y, 1.0); L_z = get(kwargs, :L_z, 1.0)
    arc_amp = get(kwargs, :arc_amp, 0.3); arc_amp_y = get(kwargs, :arc_amp_y, 0.3)
    E_val = get(kwargs, :E, 1e5); nu_val = get(kwargs, :nu, 0.3)
    n_x_lower_base = get(kwargs, :n_x_lower_base, 3)
    n_x_upper_base = get(kwargs, :n_x_upper_base, 2)
    n_y_lower_base = get(kwargs, :n_y_lower_base, 3)
    n_y_upper_base = get(kwargs, :n_y_upper_base, 2)

    B0, P  = cpt_geometry(p_ord; L_x=L_x, L_y=L_y, L_z=L_z, arc_amp=arc_amp, arc_amp_y=arc_amp_y)
    p_mat  = fill(p_ord, npc, npd)
    n_mat  = fill(p_ord+1, npc, npd)
    KV     = generate_knot_vectors(npc, npd, p_mat, n_mat)

    B0_hack = copy(B0); B0_hack[P[1], 3] .+= 1000.0

    n_x  = n_x_upper_base * 2^exp_level
    n_xl = conforming ? n_x : n_x_lower_base * 2^exp_level
    n_y  = n_y_upper_base * 2^exp_level
    n_yl = conforming ? n_y : n_y_lower_base * 2^exp_level
    n_z  = max(1, 2^exp_level)

    kref_data = Vector{Float64}[
        vcat([1.0,1.0], [i/n_xl for i in 1:n_xl-1]),
        vcat([1.0,2.0], [i/n_yl for i in 1:n_yl-1]),
        vcat([1.0,3.0], [i/n_z  for i in 1:n_z -1]),
        vcat([2.0,1.0], [i/n_x  for i in 1:n_x -1]),
        vcat([2.0,2.0], [i/n_y  for i in 1:n_y -1]),
        vcat([2.0,3.0], [i/n_z  for i in 1:n_z -1]),
    ]

    n_mat_ref, _, KV_ref, B_hack, P_ref = krefinement(nsd, npd, npc, n_mat, p_mat, KV, B0_hack, P, kref_data)
    B_ref = copy(B_hack)
    for i in axes(B_ref,1); B_ref[i,3] > 100.0 && (B_ref[i,3] -= 1000.0); end

    epss_use = epss > 0.0 ? epss : 1e6
    ncp = size(B_ref, 1)

    nel, nnp, nen = patch_metrics(npc, npd, p_mat, n_mat_ref)
    IEN = build_ien(nsd, npd, npc, p_mat, n_mat_ref, nel, nnp, nen)
    INC = [build_inc(n_mat_ref[pc,:]) for pc in 1:npc]

    dBC = [1 4 2 1 2; 2 5 2 1 2; 3 1 1 1 0]
    bc_per_dof = dirichlet_bc_control_points(p_mat, n_mat_ref, KV_ref, P_ref, npd, nnp, ned, dBC)
    neq, ID = build_id(bc_per_dof, ned, ncp)
    LM = build_lm(nen, ned, npc, nel, ID, IEN, P_ref)

    mats = [LinearElastic(E_val, nu_val, :three_d), LinearElastic(E_val, nu_val, :three_d)]
    Ub0 = zeros(ncp, nsd)
    K, _ = build_stiffness_matrix(npc, nsd, npd, ned, neq, p_mat, n_mat_ref, KV_ref, P_ref, B_ref, Ub0,
                                   nen, nel, IEN, INC, LM, mats, NQUAD, 1.0)

    # Neumann loading
    F = zeros(neq)
    traction_fn = (x,y,z) -> (σ=zeros(3,3); σ[3,3]=-1.0; σ)
    F = segment_load(n_mat_ref[2,:], p_mat[2,:], KV_ref[2], P_ref[2], B_ref,
                      nnp[2], nen[2], nsd, npd, ned, Int[], 6, ID, F, traction_fn, 1.0, NQUAD)

    K_bc, F_bc = enforce_dirichlet(Tuple{Int,Float64}[], K, F)

    # ── Interface data ───────────────────────────────────────────────────────
    pairs_tm = [InterfacePair(1,6,2,1), InterfacePair(2,1,1,6)]
    Pc = build_interface_cps(pairs_tm, p_mat, n_mat_ref, KV_ref, P_ref, npd, nnp, TwinMortarFormulation())
    nlm = length(Pc)

    # ── Raw mortar matrices ──────────────────────────────────────────────────
    D1, M12, X12, cps1, cps2a = _raw_mortar_pass(pairs_tm[1], p_mat, n_mat_ref, KV_ref, P_ref, B_ref, nnp, nsd, npd, NQUAD_mortar)
    D2, M21, X21, cps2, cps1a = _raw_mortar_pass(pairs_tm[2], p_mat, n_mat_ref, KV_ref, P_ref, B_ref, nnp, nsd, npd, NQUAD_mortar)
    @assert cps1 == cps1a "Surface 1 CPs mismatch"
    @assert cps2 == cps2a "Surface 2 CPs mismatch"

    # ── Test all 4 variants ──────────────────────────────────────────────────
    variants = [:full_two_pass, :two_half_pass, :collapsed_diag, :halfC_fullZ]
    labels   = ["Full 2-pass", "2-half-pass", "Collapsed diag", "HalfC+FullZ"]

    @printf "\n=== CPT variant comparison: p=%d, exp=%d, ε=%.0e ===\n\n" p_ord exp_level epss_use

    for (vi, variant) in enumerate(variants)
        C, Z = build_variant_CZ(variant, D1, M12, X12, D2, M21, X21,
                                cps1, cps2, Pc, nlm, ID, ned, nsd, neq, epss_use)
        C_sp = sparse(C)

        # Prune inactive multiplier columns (same as _cpt_solve)
        _, cols_nz, _ = findnz(C_sp)
        active_lm = sort(unique(cols_nz))
        if length(active_lm) < size(C_sp, 2)
            C_use = C_sp[:, active_lm]
            Z_use = sparse(Z)[active_lm, active_lm]
        else
            C_use = C_sp; Z_use = sparse(Z)
        end

        U, lam = solve_mortar(K_bc, C_use, Z_use, F_bc)

        rms_zz, max_zz, rms_all, max_all =
            stress_error_cpt(U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref,
                             B_ref, nen, nel, IEN, INC, E_val, nu_val, NQUAD)

        sym_Z = norm(Matrix(Z) - Matrix(Z)', Inf)
        sym_C_approx = norm(Matrix(C_sp) - Matrix(C_sp), Inf)  # C is not square, no symmetry check
        cond_Z = cond(Matrix(sparse(Z)[active_lm, active_lm]))

        @printf "  %-16s: RMS(σ_zz)=%.2e  MAX(σ_zz)=%.2e  ||Z-Z^T||=%.1e  cond(Z)=%.1e  ||λ||=%.2e\n" labels[vi] rms_zz max_zz sym_Z cond_Z norm(lam)
    end

    # Also run existing code's TM assembly for reference
    C_code, Z_code = build_mortar_coupling(
        Pc, pairs_tm, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
        ID, nnp, ned, nsd, npd, neq, NQUAD_mortar, epss_use,
        ElementBasedIntegration(), TwinMortarFormulation())
    _, cols_nz, _ = findnz(C_code)
    active_lm = sort(unique(cols_nz))
    if length(active_lm) < size(C_code,2)
        C_code = C_code[:, active_lm]; Z_code = Z_code[active_lm, active_lm]
    end
    U_code, lam_code = solve_mortar(K_bc, C_code, Z_code, F_bc)
    rms_code, max_code, _, _ =
        stress_error_cpt(U_code, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref,
                         B_ref, nen, nel, IEN, INC, E_val, nu_val, NQUAD)
    sym_Zc = norm(Matrix(Z_code) - Matrix(Z_code)', Inf)
    cond_Zc = cond(Matrix(Z_code))
    @printf "  %-16s: RMS(σ_zz)=%.2e  MAX(σ_zz)=%.2e  ||Z-Z^T||=%.1e  cond(Z)=%.1e  ||λ||=%.2e\n" "Code-TM (ref)" rms_code max_code sym_Zc cond_Zc norm(lam_code)
end
