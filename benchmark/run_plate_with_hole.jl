# benchmark/run_plate_with_hole.jl
#
# Plate with circular hole (2D): convergence + ε sweep + NQUAD sweep + moments.
# Uses BenchmarkFramework for standardised output.
#
# Usage:
#   julia -t auto --project=.. run_plate_with_hole.jl

import Pkg; Pkg.instantiate(; allow_autoprecomp=false)

include(joinpath(@__DIR__, "lib", "BenchmarkFramework.jl"))
include(joinpath(@__DIR__, "..", "examples", "plate_with_hole.jl"))

using Printf, Dates

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

const DEGREES    = [1, 2, 3, 4]
const EXP_RANGE  = Dict(1 => 0:5, 2 => 0:5, 3 => 0:5, 4 => 0:5)
const E_VAL      = 1e5
const NU_VAL     = 0.3
const TX_VAL     = 10.0
const R_VAL      = 1.0
const EPS_RANGE  = 10.0 .^ (-2:0.5:8)
const NQUAD_RANGE = 1:10
const EPS_SWEEP_EXP  = 3
const NQUAD_SWEEP_EXP = 2

# ═══════════════════════════════════════════════════════════════════════════════
# Solver adapter
# ═══════════════════════════════════════════════════════════════════════════════

function plate_solve(p, exp_level;
                     formulation, strategy, epss,
                     NQUAD_mortar::Int = 10,
                     max_dof::Int = typemax(Int),
                     kwargs...)
    eps_use = formulation isa SinglePassFormulation ? 0.0 : epss

    r = solve_plate_full(p, exp_level;
            E=E_VAL, nu=NU_VAL, Tx=TX_VAL, R=R_VAL,
            epss=eps_use, NQUAD_mortar=NQUAD_mortar,
            strategy=strategy, formulation=formulation)

    # r.l2_abs = L2 stress error, r.d_abs = L2 disp error, r.en_abs = energy error
    return BenchmarkResult(
        l2_disp=r.d_abs, energy=r.en_abs, l2_stress=r.l2_abs,
        ndof=0, n_lam=0)
end

# h = radial element size for outer (coarser) patch
function h_fn(p, exp)
    return (1/2) / 2^exp
end

function eps_fn(p, exp)
    h = h_fn(p, exp)
    return E_VAL * h
end

# ═══════════════════════════════════════════════════════════════════════════════
# Mesh setup for force moments
# ═══════════════════════════════════════════════════════════════════════════════

function plate_setup(p, exp_level; kwargs...)
    nsd = 2; npd = 2; npc = 2

    B0, P = plate_geometry(p; R=R_VAL, a=4.0, b=4.0)
    p_mat = fill(p, npc, npd)
    n_mat = fill(p + 1, npc, npd)
    KV = generate_knot_vectors(npc, npd, p_mat, n_mat)

    s1 = (1/3) / 2^exp_level
    s2 = (1/2) / 2^exp_level
    s2_nc = s2 / 2
    u_ang = collect(s1:s1:1.0 - s1/2)
    u_rad = collect(s2:s2:1.0 - s2/2)
    u_rad_nc = collect(s2_nc:s2_nc:1.0 - s2_nc/2)

    kref_data = Vector{Float64}[
        vcat([1.0, 1.0], u_ang),
        vcat([2.0, 1.0], u_ang),
        vcat([1.0, 2.0], u_rad_nc),
        vcat([2.0, 2.0], u_rad),
    ]

    B0_hack = copy(B0); B0_hack[P[1], 2] .+= 1000.0
    n_ref, _, KV_ref, B_hack, P_ref = krefinement(
        nsd, npd, npc, n_mat, p_mat, KV, B0_hack, P, kref_data)
    B = copy(B_hack)
    for i in axes(B, 1); B[i, 2] > 100.0 && (B[i, 2] -= 1000.0); end

    _, nnp_v, _ = patch_metrics(npc, npd, p_mat, n_ref)

    pairs_full = [InterfacePair(2, 2, 1, 4), InterfacePair(1, 4, 2, 2)]
    pairs_sp   = [InterfacePair(2, 2, 1, 4)]

    return (p_mat=p_mat, n_mat=n_ref, KV=KV_ref, P=P_ref, B=B,
            nnp=nnp_v, nsd=nsd, npd=npd,
            pairs_full=pairs_full, pairs_sp=pairs_sp)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

outdir = setup_output_dir("plate_with_hole", "systematic")
t_start = time()

println("=" ^ 90)
println("  Plate with hole: systematic benchmark")
println("  Threads: $(Threads.nthreads()), Host: $(gethostname()), Date: $(now())")
println("=" ^ 90)

# ── 1. Convergence (all 6 methods, all degrees) ─────────────────────────────
# Note: p=1 uses solve_plate_full which does k-refinement from p=1 Bezier.
# For the full factorial, all 6 methods work since both integration strategies
# are available in 2D.
println("\n>>> Convergence study...")
conv = run_convergence(plate_solve, ALL_6_METHODS, DEGREES, EXP_RANGE;
    h_fn=h_fn, eps_fn=eps_fn, max_dof=typemax(Int))
write_csv(joinpath(outdir, "convergence.csv"), conv)

# ── 2. Force moments (all 6 methods, p=2, one exp level) ─────────────────────
println("\n>>> Force moments...")
moments = run_moments(plate_setup, ALL_6_METHODS, [2, 3], [2];
    dims=[1], NQUAD_mortar_fn=(p, e) -> 10)
write_csv(joinpath(outdir, "moments.csv"), moments)

# ── 3. ε sweep (TME, TMS, DPME, DPMS at fixed exp) ─────────────────────────
println("\n>>> ε sweep...")
eps_methods = [m for m in ALL_6_METHODS if !is_single_pass(m)]
eps_rows = run_eps_sweep(plate_solve, eps_methods, [2, 3, 4], EPS_SWEEP_EXP, EPS_RANGE;
    max_dof=typemax(Int))
write_csv(joinpath(outdir, "eps_sweep.csv"), eps_rows)

# ── 4. NQUAD sweep (TME, DPME, SPME at fixed exp) ───────────────────────────
println("\n>>> NQUAD sweep...")
nq_rows = run_nquad_sweep(plate_solve, NQUAD_METHODS, [2, 3, 4], NQUAD_SWEEP_EXP, NQUAD_RANGE;
    eps_fn=eps_fn)
write_csv(joinpath(outdir, "nquad_sweep.csv"), nq_rows)

# ── 5. Meta ───────────────────────────────────────────────────────────────────
write_meta(joinpath(outdir, "meta.toml");
    description="Plate with hole: convergence + ε sweep + NQUAD sweep + moments",
    params=Dict("E" => E_VAL, "nu" => NU_VAL, "Tx" => TX_VAL, "R" => R_VAL,
                "degrees" => DEGREES, "exp_range" => "0:5"),
    outputs=["convergence.csv", "moments.csv", "eps_sweep.csv", "nquad_sweep.csv"],
    wallclock_seconds=time() - t_start)

println("\n>>> Results written to: $outdir")
println(">>> Done: $(now())")
