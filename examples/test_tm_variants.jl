# test_tm_variants.jl
#
# Compare Twin Mortar formulation variants on the curved patch test.
#
# IMPORTANT NOTE ON SIGN CONVENTION:
# The code embeds a normal-direction flip into dir_vecs (norm_sign_s * I),
# so in the code's multiplier convention, equilibrium means λ_1 ≈ λ_2
# (not λ_1 = -λ_2 as in the paper's outward-normal convention).
# Consequently, the Z off-diagonal must be POSITIVE in Z_code (to penalize
# |λ_1 - λ_2|²), which is the OPPOSITE sign from the paper's Z definition.
#
# Variants tested (all with CORRECT code-convention Z signs):
#   Current:    D-only C,  Z has D + P + M̄   (baseline)
#   Full-C:     D+M in C,  Z has D + P + M̄   (same Z, full C)
#   No-P:       D-only C,  Z has D + M̄ (no P)
#   Full-C-noP: D+M in C,  Z has D + M̄ (no P)
#   DualPass:   D-only C,  Z slave-only, full ε, no P (Puso-Solberg style)
#   FullEps:    D-only C,  Z has D + M̄ (no P), full ε factor

import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using IGAros
using LinearAlgebra, SparseArrays, Printf

include(joinpath(@__DIR__, "curved_patch_test.jl"))

# ═══════════════════════════════════════════════════════════════════════════════
# New formulation types
# ═══════════════════════════════════════════════════════════════════════════════

struct FullC_TM <: FormulationStrategy end          # current Z + M^(sm) in C
struct NoP_TM <: FormulationStrategy end             # current C, no P in Z
struct FullC_NoP_TM <: FormulationStrategy end       # full C + no P
struct DualPassZ_TM <: FormulationStrategy end       # slave-only Z, full ε, no P
struct FullEps_TM <: FormulationStrategy end         # no P, full ε factor
struct CorrectedTM <: FormulationStrategy end        # Z_I = ε·[D, M̄; M̄ᵀ, D] (no P)
struct MbarCZ_TM <: FormulationStrategy end          # M̄ in BOTH C and Z (no P)

for T in [FullC_TM, NoP_TM, FullC_NoP_TM, DualPassZ_TM, FullEps_TM, CorrectedTM, MbarCZ_TM]
    @eval IGAros._cp_facets(pair, ::$T) =
        [(pair.slave_patch, pair.slave_facet), (pair.master_patch, pair.master_facet)]
end

# ═══════════════════════════════════════════════════════════════════════════════
# Helper: shared C assembly blocks
# ═══════════════════════════════════════════════════════════════════════════════

function _assemble_C_D!(C, R_s, slave_cps, dir_vecs, Pc, nlm, ID, ned, gwJ)
    nsen_s = length(slave_cps); n_dirs = size(dir_vecs, 2)
    for b in 1:nsen_s, i in 1:ned
        eq_i = ID[i, slave_cps[b]]; eq_i == 0 && continue
        for c in 1:nsen_s
            As_c = findfirst(==(slave_cps[c]), Pc); As_c === nothing && continue
            for d in 1:n_dirs
                C[eq_i, As_c + (d-1)*nlm] -= dir_vecs[i,d] * R_s[b] * R_s[c] * gwJ
            end
        end
    end
end

function _assemble_C_M!(C, R_m, R_s, slave_cps, master_cps, dir_vecs, Pc, nlm, ID, ned, gwJ; factor=1.0)
    nsen_m = length(master_cps); nsen_s = length(slave_cps); n_dirs = size(dir_vecs, 2)
    for bm in 1:nsen_m, i in 1:ned
        eq_i = ID[i, master_cps[bm]]; eq_i == 0 && continue
        for c in 1:nsen_s
            As_c = findfirst(==(slave_cps[c]), Pc); As_c === nothing && continue
            for d in 1:n_dirs
                C[eq_i, As_c + (d-1)*nlm] += factor * dir_vecs[i,d] * R_m[bm] * R_s[c] * gwJ
            end
        end
    end
end

"""Transpose of _assemble_C_M!: puts R_s[b]·R_m[c] into C[disp_s[b], mult_m[c]].
Per half-pass this gives (M^(sm))^T. After both passes, combining with _assemble_C_M!
at factor=0.5 each, the off-diagonal of C becomes M̄ = ½(M^(12)+(M^(21))^T)."""
function _assemble_C_Mt!(C, R_m, R_s, slave_cps, master_cps, dir_vecs, Pc, nlm, ID, ned, gwJ; factor=1.0)
    nsen_s = length(slave_cps); nsen_m = length(master_cps); n_dirs = size(dir_vecs, 2)
    for b in 1:nsen_s, i in 1:ned
        eq_i = ID[i, slave_cps[b]]; eq_i == 0 && continue
        for cm in 1:nsen_m
            Am_c = findfirst(==(master_cps[cm]), Pc); Am_c === nothing && continue
            for d in 1:n_dirs
                C[eq_i, Am_c + (d-1)*nlm] += factor * dir_vecs[i,d] * R_s[b] * R_m[cm] * gwJ
            end
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# FullC_TM: Current Z (with P) + M^(sm) in C
# ═══════════════════════════════════════════════════════════════════════════════

@inline function IGAros._accumulate_mortar!(C, Z, ::FullC_TM,
    R_s, R_m, slave_cps, master_cps, dir_vecs, Pc, nlm, ID, ned, gwJ, epss)
    nsen_s = length(slave_cps); nsen_m = length(master_cps); n_dirs = size(dir_vecs, 2)

    _assemble_C_D!(C, R_s, slave_cps, dir_vecs, Pc, nlm, ID, ned, gwJ)
    _assemble_C_M!(C, R_m, R_s, slave_cps, master_cps, dir_vecs, Pc, nlm, ID, ned, gwJ)

    # Z: same as TwinMortarFormulation (D+P diag, +M off-diag)
    for b in 1:nsen_s
        As_b = findfirst(==(slave_cps[b]), Pc); As_b === nothing && continue
        for c in 1:nsen_s
            As_c = findfirst(==(slave_cps[c]), Pc); As_c === nothing && continue
            for d in 1:n_dirs; Z[As_b+(d-1)*nlm, As_c+(d-1)*nlm] -= 0.5*epss*R_s[b]*R_s[c]*gwJ; end
        end
        for c in 1:nsen_m
            Am_c = findfirst(==(master_cps[c]), Pc); Am_c === nothing && continue
            for d in 1:n_dirs; Z[As_b+(d-1)*nlm, Am_c+(d-1)*nlm] += 0.5*epss*R_s[b]*R_m[c]*gwJ; end
        end
    end
    for bm in 1:nsen_m
        Am_b = findfirst(==(master_cps[bm]), Pc); Am_b === nothing && continue
        for c in 1:nsen_s
            As_c = findfirst(==(slave_cps[c]), Pc); As_c === nothing && continue
            for d in 1:n_dirs; Z[Am_b+(d-1)*nlm, As_c+(d-1)*nlm] += 0.5*epss*R_m[bm]*R_s[c]*gwJ; end
        end
        for cm in 1:nsen_m
            Am_c = findfirst(==(master_cps[cm]), Pc); Am_c === nothing && continue
            for d in 1:n_dirs; Z[Am_b+(d-1)*nlm, Am_c+(d-1)*nlm] -= 0.5*epss*R_m[bm]*R_m[cm]*gwJ; end
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# NoP_TM: Current C (D-only), Z without P (no Z_mm block)
# ═══════════════════════════════════════════════════════════════════════════════

@inline function IGAros._accumulate_mortar!(C, Z, ::NoP_TM,
    R_s, R_m, slave_cps, master_cps, dir_vecs, Pc, nlm, ID, ned, gwJ, epss)
    nsen_s = length(slave_cps); nsen_m = length(master_cps); n_dirs = size(dir_vecs, 2)

    _assemble_C_D!(C, R_s, slave_cps, dir_vecs, Pc, nlm, ID, ned, gwJ)

    # Z: D diag + M off-diag (POSITIVE off-diag), NO P
    for b in 1:nsen_s
        As_b = findfirst(==(slave_cps[b]), Pc); As_b === nothing && continue
        for c in 1:nsen_s
            As_c = findfirst(==(slave_cps[c]), Pc); As_c === nothing && continue
            for d in 1:n_dirs; Z[As_b+(d-1)*nlm, As_c+(d-1)*nlm] -= 0.5*epss*R_s[b]*R_s[c]*gwJ; end
        end
        for c in 1:nsen_m
            Am_c = findfirst(==(master_cps[c]), Pc); Am_c === nothing && continue
            for d in 1:n_dirs; Z[As_b+(d-1)*nlm, Am_c+(d-1)*nlm] += 0.5*epss*R_s[b]*R_m[c]*gwJ; end
        end
    end
    # Master Z rows: only M^T (for symmetry), NO P
    for bm in 1:nsen_m
        Am_b = findfirst(==(master_cps[bm]), Pc); Am_b === nothing && continue
        for c in 1:nsen_s
            As_c = findfirst(==(slave_cps[c]), Pc); As_c === nothing && continue
            for d in 1:n_dirs; Z[Am_b+(d-1)*nlm, As_c+(d-1)*nlm] += 0.5*epss*R_m[bm]*R_s[c]*gwJ; end
        end
        # NO Z_mm (P) block
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# FullC_NoP_TM: Full C (D+M) + Z without P
# ═══════════════════════════════════════════════════════════════════════════════

@inline function IGAros._accumulate_mortar!(C, Z, ::FullC_NoP_TM,
    R_s, R_m, slave_cps, master_cps, dir_vecs, Pc, nlm, ID, ned, gwJ, epss)
    nsen_s = length(slave_cps); nsen_m = length(master_cps); n_dirs = size(dir_vecs, 2)

    _assemble_C_D!(C, R_s, slave_cps, dir_vecs, Pc, nlm, ID, ned, gwJ)
    _assemble_C_M!(C, R_m, R_s, slave_cps, master_cps, dir_vecs, Pc, nlm, ID, ned, gwJ)

    # Z: D + M (no P)
    for b in 1:nsen_s
        As_b = findfirst(==(slave_cps[b]), Pc); As_b === nothing && continue
        for c in 1:nsen_s
            As_c = findfirst(==(slave_cps[c]), Pc); As_c === nothing && continue
            for d in 1:n_dirs; Z[As_b+(d-1)*nlm, As_c+(d-1)*nlm] -= 0.5*epss*R_s[b]*R_s[c]*gwJ; end
        end
        for c in 1:nsen_m
            Am_c = findfirst(==(master_cps[c]), Pc); Am_c === nothing && continue
            for d in 1:n_dirs; Z[As_b+(d-1)*nlm, Am_c+(d-1)*nlm] += 0.5*epss*R_s[b]*R_m[c]*gwJ; end
        end
    end
    for bm in 1:nsen_m
        Am_b = findfirst(==(master_cps[bm]), Pc); Am_b === nothing && continue
        for c in 1:nsen_s
            As_c = findfirst(==(slave_cps[c]), Pc); As_c === nothing && continue
            for d in 1:n_dirs; Z[Am_b+(d-1)*nlm, As_c+(d-1)*nlm] += 0.5*epss*R_m[bm]*R_s[c]*gwJ; end
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# DualPassZ_TM: Slave-only Z rows, full ε, no P (Puso-Solberg Z style)
# ═══════════════════════════════════════════════════════════════════════════════

@inline function IGAros._accumulate_mortar!(C, Z, ::DualPassZ_TM,
    R_s, R_m, slave_cps, master_cps, dir_vecs, Pc, nlm, ID, ned, gwJ, epss)
    nsen_s = length(slave_cps); nsen_m = length(master_cps); n_dirs = size(dir_vecs, 2)

    _assemble_C_D!(C, R_s, slave_cps, dir_vecs, Pc, nlm, ID, ned, gwJ)

    # Z: slave rows only, full ε (not ε/2), POSITIVE off-diag, no P
    for b in 1:nsen_s
        As_b = findfirst(==(slave_cps[b]), Pc); As_b === nothing && continue
        for c in 1:nsen_s
            As_c = findfirst(==(slave_cps[c]), Pc); As_c === nothing && continue
            for d in 1:n_dirs; Z[As_b+(d-1)*nlm, As_c+(d-1)*nlm] -= epss*R_s[b]*R_s[c]*gwJ; end
        end
        for c in 1:nsen_m
            Am_c = findfirst(==(master_cps[c]), Pc); Am_c === nothing && continue
            for d in 1:n_dirs; Z[As_b+(d-1)*nlm, Am_c+(d-1)*nlm] += epss*R_s[b]*R_m[c]*gwJ; end
        end
    end
    # NO master Z rows (master rows come from the other half-pass)
end

# ═══════════════════════════════════════════════════════════════════════════════
# FullEps_TM: D-only C, no P, full ε (vs ε/2), symmetric Z
# ═══════════════════════════════════════════════════════════════════════════════

@inline function IGAros._accumulate_mortar!(C, Z, ::FullEps_TM,
    R_s, R_m, slave_cps, master_cps, dir_vecs, Pc, nlm, ID, ned, gwJ, epss)
    nsen_s = length(slave_cps); nsen_m = length(master_cps); n_dirs = size(dir_vecs, 2)

    _assemble_C_D!(C, R_s, slave_cps, dir_vecs, Pc, nlm, ID, ned, gwJ)

    # Z: D + M (no P), FULL ε factor
    for b in 1:nsen_s
        As_b = findfirst(==(slave_cps[b]), Pc); As_b === nothing && continue
        for c in 1:nsen_s
            As_c = findfirst(==(slave_cps[c]), Pc); As_c === nothing && continue
            for d in 1:n_dirs; Z[As_b+(d-1)*nlm, As_c+(d-1)*nlm] -= epss*R_s[b]*R_s[c]*gwJ; end
        end
        for c in 1:nsen_m
            Am_c = findfirst(==(master_cps[c]), Pc); Am_c === nothing && continue
            for d in 1:n_dirs; Z[As_b+(d-1)*nlm, Am_c+(d-1)*nlm] += epss*R_s[b]*R_m[c]*gwJ; end
        end
    end
    for bm in 1:nsen_m
        Am_b = findfirst(==(master_cps[bm]), Pc); Am_b === nothing && continue
        for c in 1:nsen_s
            As_c = findfirst(==(slave_cps[c]), Pc); As_c === nothing && continue
            for d in 1:n_dirs; Z[Am_b+(d-1)*nlm, As_c+(d-1)*nlm] += epss*R_m[bm]*R_s[c]*gwJ; end
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# MbarCZ_TM: M̄ in BOTH C and Z
#
# C = [0, D^(1), -M̄; 0, -M̄^T, D^(2)]      (symmetrized mortar in C)
# Z = ε·[D^(1), M̄; M̄^T, D^(2)]             (Z_I, no P)
#
# Per half-pass (slave=s, master=m):
#   C: D^(s) + ½·M^(sm) [standard] + ½·(M^(sm))^T [reverse]
#   Z: same as CorrectedTM (full ε diagonal, ε/2 off-diagonal)
#
# After both passes:
#   C off-diagonal = ½·M^(12) + ½·(M^(21))^T = M̄  ✓
#   Z = ε·[D, M̄; M̄^T, D]                          ✓
# ═══════════════════════════════════════════════════════════════════════════════

@inline function IGAros._accumulate_mortar!(C, Z, ::MbarCZ_TM,
    R_s, R_m, slave_cps, master_cps, dir_vecs, Pc, nlm, ID, ned, gwJ, epss)
    nsen_s = length(slave_cps); nsen_m = length(master_cps); n_dirs = size(dir_vecs, 2)

    # C: D^(s) + ½·M^(sm) + ½·(M^(sm))^T  → after both passes gives D + M̄
    _assemble_C_D!(C, R_s, slave_cps, dir_vecs, Pc, nlm, ID, ned, gwJ)
    _assemble_C_M!(C, R_m, R_s, slave_cps, master_cps, dir_vecs, Pc, nlm, ID, ned, gwJ; factor=0.5)
    _assemble_C_Mt!(C, R_m, R_s, slave_cps, master_cps, dir_vecs, Pc, nlm, ID, ned, gwJ; factor=0.5)

    # Z: same as CorrectedTM = Z_I = ε·[D, M̄; M̄^T, D]
    # Diagonal: full ε
    for b in 1:nsen_s
        As_b = findfirst(==(slave_cps[b]), Pc); As_b === nothing && continue
        for c in 1:nsen_s
            As_c = findfirst(==(slave_cps[c]), Pc); As_c === nothing && continue
            for d in 1:n_dirs; Z[As_b+(d-1)*nlm, As_c+(d-1)*nlm] -= epss*R_s[b]*R_s[c]*gwJ; end
        end
        # Off-diagonal: ε/2 (M̄ has ½ inside)
        for c in 1:nsen_m
            Am_c = findfirst(==(master_cps[c]), Pc); Am_c === nothing && continue
            for d in 1:n_dirs; Z[As_b+(d-1)*nlm, Am_c+(d-1)*nlm] += 0.5*epss*R_s[b]*R_m[c]*gwJ; end
        end
    end
    for bm in 1:nsen_m
        Am_b = findfirst(==(master_cps[bm]), Pc); Am_b === nothing && continue
        for c in 1:nsen_s
            As_c = findfirst(==(slave_cps[c]), Pc); As_c === nothing && continue
            for d in 1:n_dirs; Z[Am_b+(d-1)*nlm, As_c+(d-1)*nlm] += 0.5*epss*R_m[bm]*R_s[c]*gwJ; end
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# CorrectedTM: Z = ε·[D, ½M̄; ½M̄ᵀ, D] — full ε on diagonal, ε/2 on off-diag, no P
#
# Per half-pass (slave=s, master=m):
#   Z_ss: -ε · R_s⊗R_s   (full ε diagonal)
#   Z_sm: +ε/2 · R_s⊗R_m (half ε off-diagonal)
#   Z_ms: +ε/2 · R_m⊗R_s (half ε off-diagonal)
#   Z_mm: none             (no P term)
#
# After both passes:
#   (1,1) = -ε·D^(1),  (1,2) = +ε/2·M̄,  (2,1) = +ε/2·M̄ᵀ,  (2,2) = -ε·D^(2)
#
# Eigenvalue: ε·(D - M) > 0  ⟹  positive-definite WITHOUT P.
# ═══════════════════════════════════════════════════════════════════════════════

@inline function IGAros._accumulate_mortar!(C, Z, ::CorrectedTM,
    R_s, R_m, slave_cps, master_cps, dir_vecs, Pc, nlm, ID, ned, gwJ, epss)
    nsen_s = length(slave_cps); nsen_m = length(master_cps); n_dirs = size(dir_vecs, 2)

    # C: D-only (same as current TwinMortarFormulation)
    _assemble_C_D!(C, R_s, slave_cps, dir_vecs, Pc, nlm, ID, ned, gwJ)

    # Z diagonal: -ε (FULL factor, not ε/2)
    for b in 1:nsen_s
        As_b = findfirst(==(slave_cps[b]), Pc); As_b === nothing && continue
        for c in 1:nsen_s
            As_c = findfirst(==(slave_cps[c]), Pc); As_c === nothing && continue
            for d in 1:n_dirs; Z[As_b+(d-1)*nlm, As_c+(d-1)*nlm] -= epss*R_s[b]*R_s[c]*gwJ; end
        end
        # Z off-diagonal: +ε/2 (HALF factor)
        for c in 1:nsen_m
            Am_c = findfirst(==(master_cps[c]), Pc); Am_c === nothing && continue
            for d in 1:n_dirs; Z[As_b+(d-1)*nlm, Am_c+(d-1)*nlm] += 0.5*epss*R_s[b]*R_m[c]*gwJ; end
        end
    end
    # Master Z rows: only off-diagonal (ε/2), NO P
    for bm in 1:nsen_m
        Am_b = findfirst(==(master_cps[bm]), Pc); Am_b === nothing && continue
        for c in 1:nsen_s
            As_c = findfirst(==(slave_cps[c]), Pc); As_c === nothing && continue
            for d in 1:n_dirs; Z[Am_b+(d-1)*nlm, As_c+(d-1)*nlm] += 0.5*epss*R_m[bm]*R_s[c]*gwJ; end
        end
        # NO Z_mm (no P term)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Run the comparison
# ═══════════════════════════════════════════════════════════════════════════════

function run_variant_ratio_sweep(;
    p_ord   = 2,
    epss    = 1e2,
    n_slave = 6,
    n_masters = [2, 3, 4, 5, 6, 7, 8],
    variants = [
        ("Current",    TwinMortarFormulation()),
        ("CorrTM",     CorrectedTM()),
        ("Full-C",     FullC_TM()),
        ("No-P",       NoP_TM()),
        ("FC-noP",     FullC_NoP_TM()),
        ("DualZ",      DualPassZ_TM()),
        ("FullEps",    FullEps_TM()),
    ]
)
    println("\n" * "="^100)
    @printf "  VARIANT RATIO SWEEP: p=%d, ε=%.0e, slave=%d×%d\n" p_ord epss n_slave n_slave
    println("="^100)

    @printf "\n  %5s %5s |" "n_m" "ratio"
    for (lb, _) in variants; @printf " %9s |" lb; end
    println()
    @printf "  %5s %5s-|" "-----" "-----"
    for _ in variants; @printf " %9s-|" "---------"; end
    println()

    for n_m in n_masters
        ratio = n_slave / n_m
        @printf "  %5d %5.2f |" n_m ratio
        for (lb, form) in variants
            try
                U, ID, npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref,
                    nen, nel, IEN, INC, E, nu, NQUAD, _ =
                    _cpt_solve(p_ord, 0;
                               n_x_lower_base=n_slave, n_y_lower_base=n_slave,
                               n_x_upper_base=n_m, n_y_upper_base=n_m,
                               epss=epss, formulation=form)
                rms_zz, _, _, _ =
                    stress_error_cpt(U, ID, npc, nsd, npd, p_mat, n_mat_ref,
                                     KV_ref, P_ref, B_ref, nen, nel, IEN, INC, E, nu, NQUAD)
                @printf " %9.2e |" rms_zz
            catch e
                @printf " %9s |" "ERR"
                @warn "$lb n_m=$n_m: $(sprint(showerror, e))"
            end
        end
        println()
    end
    println()
end

# Key comparison: Current (Z_II) vs Z_I (D-only C) vs MbarCZ (M̄ in C+Z) vs DualZ
for eps in [1e2, 1e6]
    run_variant_ratio_sweep(p_ord=2, epss=eps,
        variants=[
            ("Z_II",     TwinMortarFormulation()),
            ("Z_I",      CorrectedTM()),
            ("MbarCZ",   MbarCZ_TM()),
            ("DualZ",    DualPassZ_TM()),
        ])
end
