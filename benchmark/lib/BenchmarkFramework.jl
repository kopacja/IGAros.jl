# benchmark/lib/BenchmarkFramework.jl
#
# Standardised benchmark framework for the Twin Mortar paper.
# Provides method configurations, study runners, CSV writers,
# force-moment computation, and condition number with SVD guard.
#
# Usage (from a per-example runner script):
#   include(joinpath(@__DIR__, "lib", "BenchmarkFramework.jl"))

using IGAros
using LinearAlgebra, SparseArrays, Printf, Dates

# ═══════════════════════════════════════════════════════════════════════════════
# Method configurations
# ═══════════════════════════════════════════════════════════════════════════════

struct MethodConfig
    label::String
    formulation::FormulationStrategy
    strategy::IntegrationStrategy
end

const ALL_6_METHODS = [
    MethodConfig("TME",  TwinMortarFormulation(),  ElementBasedIntegration()),
    MethodConfig("TMS",  TwinMortarFormulation(),  SegmentBasedIntegration()),
    MethodConfig("DPME", DualPassFormulation(),    ElementBasedIntegration()),
    MethodConfig("DPMS", DualPassFormulation(),    SegmentBasedIntegration()),
    MethodConfig("SPME", SinglePassFormulation(),  ElementBasedIntegration()),
    MethodConfig("SPMS", SinglePassFormulation(),  SegmentBasedIntegration()),
]

const TWOPASS_ELEM = [m for m in ALL_6_METHODS if m.label in ("TME", "DPME")]
const TWOPASS_ALL  = [m for m in ALL_6_METHODS if !(m.formulation isa SinglePassFormulation)]
const ELEM_METHODS = [m for m in ALL_6_METHODS if m.strategy isa ElementBasedIntegration]
const NQUAD_METHODS = [m for m in ALL_6_METHODS if m.label in ("TME", "DPME", "SPME")]

is_single_pass(cfg::MethodConfig) = cfg.formulation isa SinglePassFormulation
is_two_pass(cfg::MethodConfig)    = !is_single_pass(cfg)

# ═══════════════════════════════════════════════════════════════════════════════
# SVD-guarded condition number
# ═══════════════════════════════════════════════════════════════════════════════

"""
    safe_kappa(K_bc, C, Z; max_dof=15000) -> Float64

Condition number κ(A) of the KKT matrix [K C; Cᵀ -Z].
Returns NaN if the system exceeds `max_dof` total DOFs.
"""
function safe_kappa(
    K_bc::SparseMatrixCSC{Float64,Int},
    C::AbstractMatrix{Float64},
    Z::AbstractMatrix{Float64};
    max_dof::Int = 15000
)::Float64
    ndof = size(K_bc, 1) + size(Z, 1)
    ndof > max_dof && return NaN
    try
        Kd = Matrix(K_bc); Cd = Matrix(C); Zd = Matrix(Z)
        A = [Kd Cd; Cd' -Zd]
        sv = svdvals(A)
        sv_min = sv[end]
        sv_min < eps(Float64) * sv[1] && return Inf
        return sv[1] / sv_min
    catch
        return NaN
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Force moments (2D and 3D)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    compute_moments(D, M, s_cps, m_cps, B; dims) -> (δ₀, δ₁, δ₂)

Force-moment equilibrium errors for one mortar half-pass.
- `D`:     slave-slave mass matrix (scalar)
- `M`:     slave-master mass matrix (scalar)
- `s_cps`: global CP indices for slave interface
- `m_cps`: global CP indices for master interface
- `B`:     global CP array (ncp × 4)
- `dims`:  coordinate dimensions to use (e.g., [2] for 2D, [1,2,3] for 3D)

Returns averaged δ₀, δ₁, δ₂ over the requested dimensions.
"""
function compute_moments(D, M, s_cps, m_cps, B; dims::Vector{Int}=[2])
    Dd = Matrix(D); Md = Matrix(M)
    ns = length(s_cps); nm = length(m_cps)
    ones_s = ones(ns); ones_m = ones(nm)
    λ = ones(ns)

    δ0_sum = 0.0; δ1_sum = 0.0; δ2_sum = 0.0
    for d in dims
        y_s = [B[cp, d] for cp in s_cps]
        y_m = [B[cp, d] for cp in m_cps]
        δ0_sum += dot(λ, Dd * ones_s - Md * ones_m)
        δ1_sum += dot(λ, Dd * y_s - Md * y_m)
        δ2_sum += dot(λ, Dd * (y_s .^ 2) - Md * (y_m .^ 2))
    end
    nd = length(dims)
    return (δ_0=δ0_sum/nd, δ_1=δ1_sum/nd, δ_2=δ2_sum/nd)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Standard result type
# ═══════════════════════════════════════════════════════════════════════════════

"""
    BenchmarkResult

Standard return from a solver adapter.  Not all fields are required —
use NaN/0 for unavailable values.
"""
Base.@kwdef struct BenchmarkResult
    l2_disp::Float64   = NaN
    energy::Float64    = NaN
    l2_stress::Float64 = NaN
    kappa::Float64     = NaN
    ndof::Int          = 0
    n_lam::Int         = 0
    wall_s::Float64    = NaN
end

# ═══════════════════════════════════════════════════════════════════════════════
# Convergence study
# ═══════════════════════════════════════════════════════════════════════════════

"""
    run_convergence(solve_fn, configs, degrees, exp_range;
                    h_fn, eps_fn, max_dof, kw...) -> Vector{NamedTuple}

Run h-refinement convergence for all (method, p, exp) combinations.
`solve_fn(p, exp; formulation, strategy, epss, kw...) -> BenchmarkResult`
"""
function run_convergence(
    solve_fn, configs, degrees, exp_range;
    h_fn, eps_fn,
    max_dof::Int = 15000,
    kw...
)
    rows = NamedTuple[]
    for p in degrees
        exps = exp_range isa Dict ? exp_range[p] : exp_range
        for e in exps, cfg in configs
            eps_use = is_single_pass(cfg) ? 0.0 : eps_fn(p, e)
            t0 = time()
            try
                r = solve_fn(p, e; formulation=cfg.formulation,
                             strategy=cfg.strategy, epss=eps_use,
                             max_dof=max_dof, kw...)
                dt = time() - t0
                push!(rows, (method=cfg.label, p=p, exp=e, h=h_fn(p, e),
                             ndof=r.ndof, n_lam=r.n_lam,
                             l2_disp=r.l2_disp, energy=r.energy,
                             l2_stress=r.l2_stress, kappa=r.kappa,
                             wall_s=dt))
                @printf("  %-6s p=%d exp=%d  L2d=%.3e  E=%.3e  σ=%.3e  κ=%.3e  (%.1fs)\n",
                        cfg.label, p, e, r.l2_disp, r.energy, r.l2_stress, r.kappa, dt)
            catch ex
                dt = time() - t0
                @printf("  %-6s p=%d exp=%d  ERROR (%.1fs): %s\n",
                        cfg.label, p, e, dt, string(ex)[1:min(80,end)])
            end
            flush(stdout)
        end
    end
    return rows
end

# ═══════════════════════════════════════════════════════════════════════════════
# ε sweep
# ═══════════════════════════════════════════════════════════════════════════════

function run_eps_sweep(
    solve_fn, configs, degrees, exp_level, eps_range;
    max_dof::Int = 15000, kw...
)
    rows = NamedTuple[]
    for p in degrees, eps in eps_range, cfg in configs
        t0 = time()
        try
            r = solve_fn(p, exp_level; formulation=cfg.formulation,
                         strategy=cfg.strategy, epss=Float64(eps),
                         max_dof=max_dof, kw...)
            dt = time() - t0
            push!(rows, (method=cfg.label, p=p, exp=exp_level, eps=eps,
                         l2_disp=r.l2_disp, energy=r.energy,
                         l2_stress=r.l2_stress, kappa=r.kappa,
                         wall_s=dt))
            @printf("  %-6s p=%d ε=%.1e  L2d=%.3e  κ=%.3e\n",
                    cfg.label, p, eps, r.l2_disp, r.kappa)
        catch ex
            @printf("  %-6s p=%d ε=%.1e  ERROR: %s\n",
                    cfg.label, p, eps, string(ex)[1:min(60,end)])
        end
        flush(stdout)
    end
    return rows
end

# ═══════════════════════════════════════════════════════════════════════════════
# NQUAD sweep
# ═══════════════════════════════════════════════════════════════════════════════

function run_nquad_sweep(
    solve_fn, configs, degrees, exp_level, nquad_range;
    eps_fn, kw...
)
    rows = NamedTuple[]
    for p in degrees, nq in nquad_range, cfg in configs
        eps_use = is_single_pass(cfg) ? 0.0 : eps_fn(p, exp_level)
        t0 = time()
        try
            r = solve_fn(p, exp_level; formulation=cfg.formulation,
                         strategy=cfg.strategy, epss=eps_use,
                         NQUAD_mortar=nq, kw...)
            dt = time() - t0
            push!(rows, (method=cfg.label, p=p, exp=exp_level, nquad=nq,
                         l2_disp=r.l2_disp, energy=r.energy, wall_s=dt))
            @printf("  %-6s p=%d NQ=%d  L2d=%.3e  E=%.3e\n",
                    cfg.label, p, nq, r.l2_disp, r.energy)
        catch ex
            @printf("  %-6s p=%d NQ=%d  ERROR: %s\n",
                    cfg.label, p, nq, string(ex)[1:min(60,end)])
        end
        flush(stdout)
    end
    return rows
end

# ═══════════════════════════════════════════════════════════════════════════════
# Force-moment study
# ═══════════════════════════════════════════════════════════════════════════════

"""
    run_moments(setup_fn, configs, degrees, exp_range;
                dims, NQUAD_mortar_fn, kw...) -> Vector{NamedTuple}

Compute δ₀, δ₁, δ₂ for each (method, p, exp).
`setup_fn(p, exp; kw...) -> (p_mat, n_mat, KV, P, B, nnp, nsd, npd, pairs_full, pairs_sp)`
"""
function run_moments(
    setup_fn, configs, degrees, exp_range;
    dims::Vector{Int} = [2],
    NQUAD_mortar_fn = (p, e) -> p + 2,
    kw...
)
    rows = NamedTuple[]
    for p in degrees
        exps = exp_range isa Dict ? exp_range[p] : exp_range
        for e in exps
            # Build mesh once
            s = setup_fn(p, e; kw...)
            nqm = NQUAD_mortar_fn(p, e)

            for cfg in configs
                try
                    pair1 = s.pairs_sp[1]  # always the first slave→master pair
                    D1, M12, s1, m1 = build_mortar_mass_matrices(
                        pair1, s.p_mat, s.n_mat, s.KV, s.P, s.B,
                        s.nnp, s.nsd, s.npd, nqm, cfg.strategy)
                    mom1 = compute_moments(D1, M12, s1, m1, s.B; dims=dims)

                    if is_two_pass(cfg) && length(s.pairs_full) >= 2
                        pair2 = s.pairs_full[2]
                        D2, M21, s2, m2 = build_mortar_mass_matrices(
                            pair2, s.p_mat, s.n_mat, s.KV, s.P, s.B,
                            s.nnp, s.nsd, s.npd, nqm, cfg.strategy)
                        mom2 = compute_moments(D2, M21, s2, m2, s.B; dims=dims)
                        push!(rows, (method=cfg.label, p=p, exp=e,
                                     d0_p1=mom1.δ_0, d1_p1=mom1.δ_1, d2_p1=mom1.δ_2,
                                     d0_p2=mom2.δ_0, d1_p2=mom2.δ_1, d2_p2=mom2.δ_2,
                                     d0_sum=mom1.δ_0+mom2.δ_0,
                                     d1_sum=mom1.δ_1+mom2.δ_1,
                                     d2_sum=mom1.δ_2+mom2.δ_2))
                    else
                        push!(rows, (method=cfg.label, p=p, exp=e,
                                     d0_p1=mom1.δ_0, d1_p1=mom1.δ_1, d2_p1=mom1.δ_2,
                                     d0_p2=NaN, d1_p2=NaN, d2_p2=NaN,
                                     d0_sum=mom1.δ_0, d1_sum=mom1.δ_1, d2_sum=mom1.δ_2))
                    end
                    @printf("  %-6s p=%d exp=%d  δ₂(p1)=%.3e  δ₂(sum)=%.3e\n",
                            cfg.label, p, e, mom1.δ_2,
                            is_two_pass(cfg) ? mom1.δ_2 + mom2.δ_2 : mom1.δ_2)
                catch ex
                    @printf("  %-6s p=%d exp=%d  δ₂ ERROR: %s\n",
                            cfg.label, p, e, string(ex)[1:min(60,end)])
                end
                flush(stdout)
            end
        end
    end
    return rows
end

# ═══════════════════════════════════════════════════════════════════════════════
# CSV / Meta writers
# ═══════════════════════════════════════════════════════════════════════════════

function _fmt(x::Float64)
    isnan(x) && return ""
    isinf(x) && return "Inf"
    return @sprintf("%.6e", x)
end
_fmt(x::Int) = string(x)
_fmt(x::String) = x

function write_csv(path::String, rows::Vector{<:NamedTuple})
    isempty(rows) && return
    cols = keys(rows[1])
    open(path, "w") do io
        println(io, join(cols, ","))
        for r in rows
            println(io, join([_fmt(getfield(r, c)) for c in cols], ","))
        end
    end
    println("  Wrote: $path ($(length(rows)) rows)")
end

function write_meta(path::String; description="", params=Dict(), extra=Dict(),
                                  outputs=String[], wallclock_seconds::Real=0)
    # Delegate to IGAros.write_meta_toml which captures the full [run], [code],
    # [cluster] blocks. `path` is .../results/<benchmark>/<run_id>/meta.toml;
    # derive benchmark from the directory structure.
    run_dir = dirname(path)
    benchmark = basename(dirname(run_dir))
    IGAros.write_meta_toml(
        run_dir;
        benchmark = benchmark,
        description = description,
        parameters = params,
        outputs = outputs,
        extras = extra,
        wallclock_seconds = wallclock_seconds,
    )
    println("  Wrote: $path")
end

"""
    setup_output_dir(example_name, description) -> String

Create timestamped results directory and update `current` symlink.
Returns the path to the new directory.
"""
function setup_output_dir(example_name::String, description::String="")
    # @__DIR__ is IGAros/benchmark/lib/; three ".." up lands in the
    # twin_mortar manuscript root so paper results live in
    # twin_mortar/results/<example>/, not IGAros/results/ (the latter
    # is for package-local test artifacts only — see root CLAUDE.md).
    base = joinpath(@__DIR__, "..", "..", "..", "results", example_name)
    mkpath(base)
    stamp = Dates.format(now(), "yyyy-mm-dd")
    suffix = isempty(description) ? "" : "_" * replace(description, " " => "_")
    dirname = stamp * suffix
    outdir = joinpath(base, dirname)
    mkpath(outdir)
    # Update symlink
    link = joinpath(base, "current")
    islink(link) && rm(link)
    symlink(dirname, link)
    return outdir
end
