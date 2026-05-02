# benchmark/run_pressurized_sphere.jl
#
# Pressurized sphere (3D, deltoidal): convergence + ε sweep + NQUAD sweep + moments.
#
# Usage:
#   julia -t auto --project=.. run_pressurized_sphere.jl

import Pkg; Pkg.instantiate(; allow_autoprecomp=false)

include(joinpath(@__DIR__, "lib", "BenchmarkFramework.jl"))
include(joinpath(@__DIR__, "..", "examples", "pressurized_sphere.jl"))
include(joinpath(@__DIR__, "..", "examples", "cz_cancellation.jl"))  # compute_force_moments

using Printf, Dates

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

const DEGREES   = [1, 2, 3, 4]
const EXP_RANGE = Dict(1 => 0:3, 2 => 0:3, 3 => 0:3, 4 => 0:3)
const E_VAL     = 1.0
const NU_VAL    = 0.3
const P_I       = 0.01
const R_I       = 1.0
const R_C       = 1.2
const R_O       = 1.4
const EPSS_DEF  = 1e4
const EPS_RANGE = 10.0 .^ (-2:1:7)
const NQUAD_RANGE = 2:8
const EPS_SWEEP_EXP  = 1
const NQUAD_SWEEP_EXP = 1
const SVD_MAX   = 15000

# Base mesh per degree
const BASE = Dict(
    1 => (n_out=2, n_in=3, n_rad=1),
    2 => (n_out=2, n_in=3, n_rad=1),
    3 => (n_out=1, n_in=2, n_rad=1),
    4 => (n_out=1, n_in=2, n_rad=1),
)

# ═══════════════════════════════════════════════════════════════════════════════
# Solver adapter
# ═══════════════════════════════════════════════════════════════════════════════

function sphere_solve(p, exp_level;
                      formulation, strategy, epss,
                      NQUAD_mortar::Int = p + 2,
                      max_dof::Int = SVD_MAX,
                      kwargs...)
    eps_use = formulation isa SinglePassFormulation ? 0.0 : epss
    bm = BASE[p]

    r = solve_sphere_deltoidal(p, exp_level;
            r_i=R_I, r_c=R_C, r_o=R_O, E=E_VAL, nu=NU_VAL, p_i=P_I,
            epss=eps_use, NQUAD_mortar=NQUAD_mortar,
            strategy=strategy, formulation=formulation,
            n_ang_outer_base=bm.n_out,
            n_ang_inner_base=bm.n_in,
            n_rad_base=bm.n_rad)

    kappa = safe_kappa(r.K_bc, r.C, r.Z; max_dof=max_dof)
    return BenchmarkResult(l2_disp=r.l2_abs, energy=r.en_abs, l2_stress=r.σ_abs,
                           kappa=kappa, ndof=r.neq, n_lam=size(r.C, 2))
end

function h_fn(p, exp)
    bm = BASE[p]
    return 1.0 / (bm.n_out * 2^exp)
end

function eps_fn(p, exp)
    return EPSS_DEF
end

# ═══════════════════════════════════════════════════════════════════════════════
# Mesh setup for force moments
# ═══════════════════════════════════════════════════════════════════════════════

function sphere_setup(p, exp_level; kwargs...)
    bm = BASE[p]; nsd = 3; npd = 3

    n_ang_outer = bm.n_out * 2^exp_level
    n_ang_inner = bm.n_in * 2^exp_level
    n_rad = bm.n_rad * 2^exp_level

    B, P, p_mat, n_mat, KV, npc = sphere_deltoidal(
        p, n_ang_inner, n_ang_outer, n_rad;
        r_i=R_I, r_c=R_C, r_o=R_O)
    _, nnp_v, _ = patch_metrics(npc, npd, p_mat, n_mat)

    # Inner→outer pairs (3 interface pairs for 3 kite patches)
    pairs_full = [InterfacePair(1, 6, 4, 1), InterfacePair(4, 1, 1, 6),
                  InterfacePair(2, 6, 5, 1), InterfacePair(5, 1, 2, 6),
                  InterfacePair(3, 6, 6, 1), InterfacePair(6, 1, 3, 6)]
    pairs_sp   = [InterfacePair(1, 6, 4, 1),
                  InterfacePair(2, 6, 5, 1),
                  InterfacePair(3, 6, 6, 1)]

    return (p_mat=p_mat, n_mat=n_mat, KV=KV, P=P, B=B,
            nnp=nnp_v, nsd=nsd, npd=npd,
            pairs_full=pairs_full, pairs_sp=pairs_sp)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

outdir = setup_output_dir("sphere", "deltoidal_systematic")
t_start = time()

println("=" ^ 90)
println("  Pressurized sphere (deltoidal): systematic benchmark")
println("  Threads: $(Threads.nthreads()), Host: $(gethostname()), Date: $(now())")
println("=" ^ 90)

# ── 1. Convergence ───────────────────────────────────────────────────────────
println("\n>>> Convergence study...")
conv = run_convergence(sphere_solve, ALL_6_METHODS, DEGREES, EXP_RANGE;
    h_fn=h_fn, eps_fn=eps_fn, max_dof=SVD_MAX)
write_csv(joinpath(outdir, "convergence.csv"), conv)

# ── 2. Force moments (p=2,3 at exp=1) ────────────────────────────────────────
# Note: for sphere with 3 interface pairs, moments are computed per pair.
# run_moments uses the first pair only — sufficient for δ₂ cancellation demo.
println("\n>>> Force moments...")
moments = run_moments(sphere_setup, ALL_6_METHODS, [2, 3], [1];
    dims=[1, 2, 3], NQUAD_mortar_fn=(p, e) -> p + 2)
write_csv(joinpath(outdir, "moments.csv"), moments)

# ── 3. ε sweep ───────────────────────────────────────────────────────────────
println("\n>>> ε sweep...")
eps_methods = [m for m in ALL_6_METHODS if !is_single_pass(m)]
eps_rows = run_eps_sweep(sphere_solve, eps_methods, [1, 2, 3, 4], EPS_SWEEP_EXP, EPS_RANGE;
    max_dof=SVD_MAX)
write_csv(joinpath(outdir, "eps_sweep.csv"), eps_rows)

# ── 4. NQUAD sweep ──────────────────────────────────────────────────────────
println("\n>>> NQUAD sweep...")
nq_rows = run_nquad_sweep(sphere_solve, NQUAD_METHODS, [2, 3], NQUAD_SWEEP_EXP, NQUAD_RANGE;
    eps_fn=eps_fn)
write_csv(joinpath(outdir, "nquad_sweep.csv"), nq_rows)

# ── 5. Meta ───────────────────────────────────────────────────────────────────
write_meta(joinpath(outdir, "meta.toml");
    description="Pressurized sphere (deltoidal): convergence + ε sweep + NQUAD sweep + moments",
    params=Dict("E" => E_VAL, "nu" => NU_VAL, "p_i" => P_I,
                "r_i" => R_I, "r_c" => R_C, "r_o" => R_O,
                "epss_default" => EPSS_DEF, "degrees" => DEGREES,
                "geometry" => "deltoidal icositetrahedron, Greville interpolation"),
    outputs=["convergence.csv", "moments.csv", "eps_sweep.csv", "nquad_sweep.csv"],
    wallclock_seconds=time() - t_start)

println("\n>>> Results written to: $outdir")
println(">>> Done: $(now())")
