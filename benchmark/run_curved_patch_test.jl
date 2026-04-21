# benchmark/run_curved_patch_test.jl
#
# 3D curved patch test (polynomial-matched interface): factorial + ε sweep +
# NQUAD sweep + force moments.
#
# Usage:
#   julia -t auto --project=.. run_curved_patch_test.jl

import Pkg; Pkg.instantiate(; allow_autoprecomp=false)

include(joinpath(@__DIR__, "lib", "BenchmarkFramework.jl"))
include(joinpath(@__DIR__, "..", "examples", "poly_patch_test.jl"))

using Printf, Dates

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

const DEGREES   = [2, 3, 4]
const EXP_RANGE = 0:2
const E_VAL     = 1e5
const NU_VAL    = 0.3
const DELTA     = 0.15
const EPSS_DEF  = 1e6
const EPS_RANGE = 10.0 .^ (-2:0.5:8)
const NQUAD_RANGE = 2:10
const EPS_SWEEP_EXP  = 1
const NQUAD_SWEEP_EXP = 1
const SVD_MAX   = 15000

# ═══════════════════════════════════════════════════════════════════════════════
# Solver adapter
# ═══════════════════════════════════════════════════════════════════════════════

function cpt_solve(p, exp_level;
                   formulation, strategy, epss,
                   NQUAD_mortar::Int = p + 4,
                   max_dof::Int = SVD_MAX,
                   kwargs...)
    eps_use = formulation isa SinglePassFormulation ? 0.0 : epss
    r = solve_ppt(p, exp_level;
            E=E_VAL, nu=NU_VAL, δ=DELTA,
            epss=eps_use, NQUAD_mortar=NQUAD_mortar,
            strategy=strategy, formulation=formulation)

    kappa = safe_kappa(sparse(zeros(0,0)), sparse(zeros(0,0)), sparse(zeros(0,0)); max_dof=0)
    # kappa computed inside solve_ppt if possible — extract from r
    kappa = hasfield(typeof(r), :kappa) ? r.kappa : NaN

    return BenchmarkResult(l2_disp=NaN, energy=NaN, l2_stress=r.rms_zz,
                           kappa=kappa)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Mesh setup for force moments
# ═══════════════════════════════════════════════════════════════════════════════

function cpt_setup(p, exp_level; kwargs...)
    nsd = 3; npd = 3; npc = 2

    # Rebuild the mesh (same as solve_ppt)
    B0, P = ppt_geometry(p; L_x=1.0, L_y=1.0, L_z=1.0, δ=DELTA)
    p_mat = fill(p, npc, npd)
    n_mat = fill(p + 1, npc, npd)
    KV = generate_knot_vectors(npc, npd, p_mat, n_mat)

    B0_hack = copy(B0); B0_hack[P[1], 3] .+= 1000.0
    n_x = 2 * 2^exp_level; n_xl = 3 * 2^exp_level
    n_y = 2 * 2^exp_level; n_yl = 3 * 2^exp_level
    n_z = max(1, 2^exp_level)

    kref_data = Vector{Float64}[
        vcat([1.0,1.0], [i/n_xl for i in 1:n_xl-1]),
        vcat([1.0,2.0], [i/n_yl for i in 1:n_yl-1]),
        vcat([1.0,3.0], [i/n_z  for i in 1:n_z -1]),
        vcat([2.0,1.0], [i/n_x  for i in 1:n_x -1]),
        vcat([2.0,2.0], [i/n_y  for i in 1:n_y -1]),
        vcat([2.0,3.0], [i/n_z  for i in 1:n_z -1]),
    ]

    n_ref, _, KV_ref, B_hack, P_ref = krefinement(
        nsd, npd, npc, n_mat, p_mat, KV, B0_hack, P, kref_data)
    B = copy(B_hack)
    for i in axes(B, 1); B[i, 3] > 100.0 && (B[i, 3] -= 1000.0); end

    _, nnp_v, _ = patch_metrics(npc, npd, p_mat, n_ref)

    pairs_full = [InterfacePair(1, 6, 2, 1), InterfacePair(2, 1, 1, 6)]
    pairs_sp   = [InterfacePair(1, 6, 2, 1)]

    return (p_mat=p_mat, n_mat=n_ref, KV=KV_ref, P=P_ref, B=B,
            nnp=nnp_v, nsd=nsd, npd=npd,
            pairs_full=pairs_full, pairs_sp=pairs_sp)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

outdir = setup_output_dir("curved_patch_test", "polynomial_matched")
t_start = time()

println("=" ^ 90)
println("  Curved patch test (polynomial-matched): systematic benchmark")
println("  Threads: $(Threads.nthreads()), Host: $(gethostname()), Date: $(now())")
println("=" ^ 90)

# ── 1. Factorial (all 6 methods × all degrees × exp levels) ─────────────────
println("\n>>> Factorial study...")
conv = run_convergence(cpt_solve, ALL_6_METHODS, DEGREES, EXP_RANGE;
    h_fn = (p, e) -> 1.0 / (2 * 2^e),
    eps_fn = (p, e) -> EPSS_DEF,
    max_dof = SVD_MAX)
write_csv(joinpath(outdir, "factorial.csv"), conv)

# ── 2. Force moments ─────────────────────────────────────────────────────────
println("\n>>> Force moments...")
moments = run_moments(cpt_setup, ALL_6_METHODS, DEGREES, EXP_RANGE;
    dims=[1, 2, 3], NQUAD_mortar_fn=(p, e) -> p + 4)
write_csv(joinpath(outdir, "moments.csv"), moments)

# ── 3. ε sweep (TME, DPME only — most informative) ──────────────────────────
println("\n>>> ε sweep...")
eps_methods = [m for m in ALL_6_METHODS if !is_single_pass(m)]
eps_rows = run_eps_sweep(cpt_solve, eps_methods, DEGREES, EPS_SWEEP_EXP, EPS_RANGE;
    max_dof=SVD_MAX)
write_csv(joinpath(outdir, "eps_sweep.csv"), eps_rows)

# ── 4. NQUAD sweep ──────────────────────────────────────────────────────────
println("\n>>> NQUAD sweep...")
nq_rows = run_nquad_sweep(cpt_solve, NQUAD_METHODS, DEGREES, NQUAD_SWEEP_EXP, NQUAD_RANGE;
    eps_fn = (p, e) -> EPSS_DEF)
write_csv(joinpath(outdir, "nquad_sweep.csv"), nq_rows)

# ── 5. Meta ───────────────────────────────────────────────────────────────────
write_meta(joinpath(outdir, "meta.toml");
    description="Curved patch test (polynomial-matched interface): factorial + sweeps + moments",
    params=Dict("E" => E_VAL, "nu" => NU_VAL, "delta" => DELTA,
                "degrees" => DEGREES, "exp_range" => "0:2",
                "interface" => "degree-p polynomial (exact in B-spline space)"),
    outputs=["factorial.csv", "moments.csv", "eps_sweep.csv", "nquad_sweep.csv"],
    wallclock_seconds=time() - t_start)

println("\n>>> Results written to: $outdir")
println(">>> Done: $(now())")
