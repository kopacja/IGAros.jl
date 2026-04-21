# benchmark/run_flat_patch_test.jl
#
# 2D flat patch test: factorial table + ε sweep + NQUAD sweep + force moments.
# Uses BenchmarkFramework for standardised output.
#
# Usage:
#   julia -t auto --project=.. run_flat_patch_test.jl

import Pkg; Pkg.instantiate(; allow_autoprecomp=false)

include(joinpath(@__DIR__, "lib", "BenchmarkFramework.jl"))
include(joinpath(@__DIR__, "..", "examples", "cz_cancellation.jl"))

using Printf, Dates

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

const DEGREES  = [1, 2, 3, 4]
const N_S      = 4      # slave interface elements
const N_M      = 7      # master interface elements
const E_VAL    = 1000.0
const NU_VAL   = 0.0
const EPSS_DEF = 10.0   # default ε for two-pass methods
const EPS_RANGE = 10.0 .^ (-2:0.5:8)
const NQUAD_RANGE = 1:20

# ═══════════════════════════════════════════════════════════════════════════════
# Solver adapter
# ═══════════════════════════════════════════════════════════════════════════════

function flat_solve(p, _exp_unused;
                    formulation, strategy, epss,
                    NQUAD_mortar::Int = p + 2,
                    max_dof::Int = typemax(Int),
                    kwargs...)
    eps_use = formulation isa SinglePassFormulation ? 0.0 : epss
    r = flat_patch_test_hp(p, N_S, N_M;
            E=E_VAL, nu=NU_VAL, epss=eps_use,
            NQUAD_mortar=NQUAD_mortar,
            strategy=strategy, formulation=formulation)

    # Continuous L² displacement error (absolute)
    NQUAD_vol = p + 1
    d_abs, d_ref = l2_disp_error_flat(r.U, r.ID, r.npc, r.nsd, r.npd,
                                       r.p_mat, r.n_mat, r.KV, r.P, r.B,
                                       r.nen, r.nel, r.IEN, r.INC,
                                       NQUAD_vol, E_VAL)
    d_err = d_abs

    kappa = safe_kappa(r.K_bc, r.C, r.Z; max_dof=max_dof)

    return BenchmarkResult(l2_disp=d_err, energy=NaN, l2_stress=NaN,
                           kappa=kappa, ndof=r.neq, n_lam=size(r.C, 2))
end

# ═══════════════════════════════════════════════════════════════════════════════
# Mesh setup for force moments
# ═══════════════════════════════════════════════════════════════════════════════

function flat_setup(p, _exp_unused; kwargs...)
    nsd = 2; npd = 2; npc = 2

    if p == 1
        n = [2 N_S+1; 2 N_M+1]
        p_mat = [1 1; 1 1]
        KV = generate_knot_vectors(npc, npd, p_mat, n)
        ncp1 = 2 * (N_S + 1); ncp2 = 2 * (N_M + 1); ncp = ncp1 + ncp2
        B = zeros(ncp, 4)
        idx = 0
        for j in 0:N_S, i in 0:1
            idx += 1; B[idx, :] = [i * 0.5, j / N_S, 0.0, 1.0]
        end
        for j in 0:N_M, i in 0:1
            idx += 1; B[idx, :] = [0.5 + i * 0.5, j / N_M, 0.0, 1.0]
        end
        P = [collect(1:ncp1), collect(ncp1+1:ncp)]
        nnp_vec = [ncp1, ncp2]
    else
        p_mat = fill(p, npc, npd)
        n_init = fill(p + 1, npc, npd)
        KV_init = generate_knot_vectors(npc, npd, p_mat, n_init)
        ncp_p1 = (p + 1)^2
        B0 = zeros(2 * ncp_p1, 4)
        idx = 0
        for j in 0:p, i in 0:p
            idx += 1; B0[idx, :] = [i * 0.5 / p, j / p, 0.0, 1.0]
        end
        for j in 0:p, i in 0:p
            idx += 1; B0[idx, :] = [0.5 + i * 0.5 / p, j / p, 0.0, 1.0]
        end
        P_init = [collect(1:ncp_p1), collect(ncp_p1+1:2*ncp_p1)]

        s_s = 1.0 / N_S; s_m = 1.0 / N_M
        u_s = collect(s_s:s_s:1 - s_s/2)
        u_m = collect(s_m:s_m:1 - s_m/2)
        kref_data = Vector{Float64}[
            vcat([1.0, 2.0], u_s), vcat([2.0, 2.0], u_m)]

        B0_hack = copy(B0); B0_hack[P_init[1], 2] .+= 1000.0
        n, _, KV, B_hack, P = krefinement(
            nsd, npd, npc, n_init, p_mat, KV_init, B0_hack, P_init, kref_data)
        B = copy(B_hack)
        for i in axes(B, 1); B[i, 2] > 100.0 && (B[i, 2] -= 1000.0); end
        _, nnp_vec, _ = patch_metrics(npc, npd, p_mat, n)
    end

    pairs_full = [InterfacePair(1, 2, 2, 4), InterfacePair(2, 4, 1, 2)]
    pairs_sp   = [InterfacePair(1, 2, 2, 4)]
    nel_v, nnp_v, nen_v = patch_metrics(npc, npd, p_mat, n)

    return (p_mat=p_mat, n_mat=n, KV=KV, P=P, B=B,
            nnp=nnp_v, nsd=nsd, npd=npd,
            pairs_full=pairs_full, pairs_sp=pairs_sp)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

outdir = setup_output_dir("flat_patch_test", "factorial")
t_start = time()

println("=" ^ 90)
println("  2D flat patch test: systematic benchmark")
println("  Threads: $(Threads.nthreads()), Host: $(gethostname()), Date: $(now())")
println("=" ^ 90)

# ── 1. Factorial (one "exp level" = 0, all 6 methods × all degrees) ──────────
println("\n>>> Factorial study...")
conv = run_convergence(flat_solve, ALL_6_METHODS, DEGREES, [0];
    h_fn = (p, e) -> 1.0 / N_S,
    eps_fn = (p, e) -> EPSS_DEF,
    max_dof = typemax(Int))
write_csv(joinpath(outdir, "factorial.csv"), conv)

# ── 2. Force moments ─────────────────────────────────────────────────────────
println("\n>>> Force moments...")
moments = run_moments(flat_setup, ALL_6_METHODS, DEGREES, [0];
    dims=[2], NQUAD_mortar_fn=(p, e) -> p + 2)
write_csv(joinpath(outdir, "moments.csv"), moments)

# ── 3. ε sweep (TME, TMS, DPME, DPMS only) ──────────────────────────────────
println("\n>>> ε sweep...")
eps_methods = [m for m in ALL_6_METHODS if !is_single_pass(m)]
eps_rows = run_eps_sweep(flat_solve, eps_methods, DEGREES, 0, EPS_RANGE;
    max_dof = typemax(Int))
write_csv(joinpath(outdir, "eps_sweep.csv"), eps_rows)

# ── 4. NQUAD sweep (TME, DPME, SPME only) ────────────────────────────────────
println("\n>>> NQUAD sweep...")
nq_rows = run_nquad_sweep(flat_solve, NQUAD_METHODS, DEGREES, 0, NQUAD_RANGE;
    eps_fn = (p, e) -> EPSS_DEF)
write_csv(joinpath(outdir, "nquad_sweep.csv"), nq_rows)

# ── 5. Meta ───────────────────────────────────────────────────────────────────
write_meta(joinpath(outdir, "meta.toml");
    description="2D flat patch test: factorial + ε sweep + NQUAD sweep + moments",
    params=Dict("E" => E_VAL, "nu" => NU_VAL, "n_s" => N_S, "n_m" => N_M,
                "epss_default" => EPSS_DEF, "degrees" => DEGREES),
    outputs=["factorial.csv", "moments.csv", "eps_sweep.csv", "nquad_sweep.csv"],
    wallclock_seconds=time() - t_start)

println("\n>>> Results written to: $outdir")
println(">>> Done: $(now())")
