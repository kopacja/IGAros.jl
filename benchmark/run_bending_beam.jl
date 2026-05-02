# benchmark/run_bending_beam.jl
#
# 3D bending beam: convergence + ε sweep + NQUAD sweep + force moments.
#
# Usage:
#   julia -t auto --project=.. run_bending_beam.jl

import Pkg; Pkg.instantiate(; allow_autoprecomp=false)

include(joinpath(@__DIR__, "lib", "BenchmarkFramework.jl"))
include(joinpath(@__DIR__, "..", "examples", "bending_beam.jl"))
include(joinpath(@__DIR__, "..", "examples", "cz_cancellation.jl"))  # compute_force_moments

using Printf, Dates

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

const DEGREES   = [1, 2, 3, 4]
const EXP_RANGE = 0:3
const E_VAL     = 1000.0
const NU_VAL    = 0.0
const P_LOAD    = 10.0
const L_X       = 8.0
const L_Y       = 2.0
const L_Z       = 2.0
const EPS_RANGE = 10.0 .^ (-2:0.5:8)
const NQUAD_RANGE = 2:8
const EPS_SWEEP_EXP  = 1
const NQUAD_SWEEP_EXP = 1
const SVD_MAX   = 15000

# ═══════════════════════════════════════════════════════════════════════════════
# Solver adapter
# ═══════════════════════════════════════════════════════════════════════════════

function beam_solve(p, exp_level;
                    formulation, strategy, epss,
                    NQUAD_mortar::Int = p + 2,
                    max_dof::Int = SVD_MAX,
                    kwargs...)
    eps_use = formulation isa SinglePassFormulation ? 0.0 : epss

    r = if p == 1
        solve_beam_p1(exp_level;
            E=E_VAL, nu=NU_VAL, p_load=P_LOAD,
            l_x=L_X, l_y=L_Y, l_z=L_Z,
            epss=eps_use, NQUAD_mortar=NQUAD_mortar,
            strategy=strategy, formulation=formulation)
    else
        solve_beam(p, exp_level;
            E=E_VAL, nu=NU_VAL, p_load=P_LOAD,
            l_x=L_X, l_y=L_Y, l_z=L_Z,
            epss=eps_use, NQUAD_mortar=NQUAD_mortar,
            strategy=strategy, formulation=formulation)
    end

    kappa = safe_kappa(r.K_bc, r.C, r.Z; max_dof=max_dof)
    return BenchmarkResult(l2_disp=r.l2_abs, energy=r.en_abs, l2_stress=r.σ_abs,
                           kappa=kappa, ndof=r.neq, n_lam=size(r.C, 2))
end

function h_fn(p, exp)
    return 1.0 / (2 * 2^exp)
end

function eps_fn(p, exp)
    return p == 1 ? 1e5 : 1e6
end

# ═══════════════════════════════════════════════════════════════════════════════
# Mesh setup for force moments
# ═══════════════════════════════════════════════════════════════════════════════

function beam_setup(p, exp_level; kwargs...)
    nsd = 3; npd = 3; npc = 2

    B0, P = beam_geometry(p; l_x=L_X, l_y=L_Y, l_z=L_Z)
    p_mat = fill(p, npc, npd)
    n_mat = fill(p + 1, npc, npd)
    KV = generate_knot_vectors(npc, npd, p_mat, n_mat)

    B0_hack = copy(B0); B0_hack[P[1], 2] .+= 1000.0
    n_x = 1 * 2^exp_level; n_xl = 2 * 2^exp_level
    n_y = 2^exp_level; n_z = max(1, 2^exp_level)
    kref_data = Vector{Float64}[
        vcat([1.0,1.0], [i/n_xl for i in 1:n_xl-1]),
        vcat([1.0,2.0], [i/n_y  for i in 1:n_y -1]),
        vcat([1.0,3.0], [i/n_z  for i in 1:n_z -1]),
        vcat([2.0,1.0], [i/n_x  for i in 1:n_x -1]),
        vcat([2.0,2.0], [i/n_y  for i in 1:n_y -1]),
        vcat([2.0,3.0], [i/n_z  for i in 1:n_z -1]),
    ]

    n_ref, _, KV_ref, B_hack, P_ref = krefinement(
        nsd, npd, npc, n_mat, p_mat, KV, B0_hack, P, kref_data)
    B = copy(B_hack)
    for i in axes(B, 1); B[i, 2] > 100.0 && (B[i, 2] -= 1000.0); end
    _, nnp_v, _ = patch_metrics(npc, npd, p_mat, n_ref)

    pairs_full = [InterfacePair(1, 3, 2, 5), InterfacePair(2, 5, 1, 3)]
    pairs_sp   = [InterfacePair(1, 3, 2, 5)]

    return (p_mat=p_mat, n_mat=n_ref, KV=KV_ref, P=P_ref, B=B,
            nnp=nnp_v, nsd=nsd, npd=npd,
            pairs_full=pairs_full, pairs_sp=pairs_sp)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

outdir = setup_output_dir("bending_beam", "systematic")
t_start = time()

println("=" ^ 90)
println("  Bending beam: systematic benchmark")
println("  Threads: $(Threads.nthreads()), Host: $(gethostname()), Date: $(now())")
println("=" ^ 90)

# ── 1. Convergence ───────────────────────────────────────────────────────────
println("\n>>> Convergence study...")
conv = run_convergence(beam_solve, ALL_6_METHODS, DEGREES, EXP_RANGE;
    h_fn=h_fn, eps_fn=eps_fn, max_dof=SVD_MAX)
write_csv(joinpath(outdir, "convergence.csv"), conv)

# ── 2. Force moments ─────────────────────────────────────────────────────────
println("\n>>> Force moments...")
moments = run_moments(beam_setup, ALL_6_METHODS, [2, 3], [1];
    dims=[1, 2, 3], NQUAD_mortar_fn=(p, e) -> p + 2)
write_csv(joinpath(outdir, "moments.csv"), moments)

# ── 3. ε sweep ───────────────────────────────────────────────────────────────
println("\n>>> ε sweep...")
eps_methods = [m for m in ALL_6_METHODS if !is_single_pass(m)]
eps_rows = run_eps_sweep(beam_solve, eps_methods, [2, 3, 4], EPS_SWEEP_EXP, EPS_RANGE;
    max_dof=SVD_MAX)
write_csv(joinpath(outdir, "eps_sweep.csv"), eps_rows)

# ── 4. NQUAD sweep ──────────────────────────────────────────────────────────
println("\n>>> NQUAD sweep...")
nq_rows = run_nquad_sweep(beam_solve, NQUAD_METHODS, [2, 3], NQUAD_SWEEP_EXP, NQUAD_RANGE;
    eps_fn=eps_fn)
write_csv(joinpath(outdir, "nquad_sweep.csv"), nq_rows)

# ── 5. Meta ───────────────────────────────────────────────────────────────────
write_meta(joinpath(outdir, "meta.toml");
    description="Bending beam: convergence + ε sweep + NQUAD sweep + moments",
    params=Dict("E" => E_VAL, "nu" => NU_VAL, "p_load" => P_LOAD,
                "l_x" => L_X, "l_y" => L_Y, "l_z" => L_Z,
                "degrees" => DEGREES, "exp_range" => "0:3"),
    outputs=["convergence.csv", "moments.csv", "eps_sweep.csv", "nquad_sweep.csv"],
    wallclock_seconds=time() - t_start)

println("\n>>> Results written to: $outdir")
println(">>> Done: $(now())")
