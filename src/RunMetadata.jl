"""
    RunMetadata

Emit `meta.toml` run-provenance files for benchmark results under
`twin_mortar/results/<benchmark>/<run_id>/`. Captures git state of the
IGAros repo, SLURM environment when present, Julia version, and
caller-supplied parameters and output filenames.

The schema matches `results/meta.toml.template` in the parent manuscript
repo: `[run]`, `[code]`, `[cluster]`, `[parameters]`, `[outputs]`.
Additional top-level tables (e.g. `[geometry]`, `[material]`) may be
passed through `extras`.
"""
module RunMetadata

using Dates
using TOML

export write_meta_toml, capture_code_provenance, capture_cluster_env

const _DEFAULT_IGAROS_ROOT = normpath(joinpath(@__DIR__, ".."))

"""
    capture_code_provenance(igaros_root=<pkg root>) -> Dict

Return `[code]` block values (commit short SHA, branch, dirty flag,
Julia version) by shelling out to `git -C igaros_root`. Returns empty
strings / `false` for fields that cannot be resolved so that
`TOML.print` never fails at write time.
"""
function capture_code_provenance(igaros_root::AbstractString = _DEFAULT_IGAROS_ROOT)
    return Dict(
        "igaros_commit" => _git(igaros_root, "rev-parse", "--short", "HEAD"),
        "igaros_branch" => _git(igaros_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "igaros_dirty"  => !isempty(_git(igaros_root, "status", "--porcelain")),
        "julia_version" => string(VERSION),
    )
end

"""
    capture_cluster_env(; wallclock_seconds=0) -> Dict

Return `[cluster]` block values from SLURM environment variables when
present, falling back to `gethostname()` and zeros off-cluster.
`wallclock_seconds` is caller-measured (`t0 = time(); …; time() - t0`).
"""
function capture_cluster_env(; wallclock_seconds::Real = 0)
    return Dict(
        "host"              => get(ENV, "SLURMD_NODENAME", _hostname()),
        "partition"         => get(ENV, "SLURM_JOB_PARTITION", ""),
        "cpus_per_task"     => _parse_int(get(ENV, "SLURM_CPUS_PER_TASK", "0")),
        "slurm_jobid"       => _parse_int(get(ENV, "SLURM_JOB_ID", "0")),
        "threads"           => Threads.nthreads(),
        "wallclock_seconds" => round(Int, wallclock_seconds),
    )
end

"""
    write_meta_toml(run_dir; benchmark, description,
                             parameters=Dict(), outputs=String[],
                             extras=Dict(),
                             igaros_root=<pkg root>,
                             wallclock_seconds=0,
                             figures_regenerated_by="plot.jl") -> String

Write `run_dir/meta.toml` with `[run]`, `[code]`, `[cluster]`,
`[parameters]`, `[outputs]` plus any tables in `extras`. Returns the
absolute path of the written file. Errors if `run_dir` does not exist —
callers are expected to `mkpath(run_dir)` first.

Values in `parameters` and `extras` must be TOML-serialisable: numbers,
strings, bools, arrays of those, or nested dicts keyed by string.
"""
function write_meta_toml(run_dir::AbstractString;
                         benchmark::AbstractString,
                         description::AbstractString,
                         parameters::AbstractDict = Dict{String,Any}(),
                         outputs = String[],
                         extras::AbstractDict = Dict{String,Any}(),
                         igaros_root::AbstractString = _DEFAULT_IGAROS_ROOT,
                         wallclock_seconds::Real = 0,
                         figures_regenerated_by::AbstractString = "plot.jl")
    isdir(run_dir) || error("run_dir does not exist: $run_dir")

    data = Dict{String,Any}(
        "run" => Dict(
            "date"        => string(Dates.today()),
            "time"        => Dates.format(now(), "HH:MM:SS"),
            "benchmark"   => String(benchmark),
            "run_id"      => basename(normpath(run_dir)),
            "description" => String(description),
        ),
        "code"    => capture_code_provenance(igaros_root),
        "cluster" => capture_cluster_env(; wallclock_seconds),
        "parameters" => Dict{String,Any}(String(k) => v for (k, v) in parameters),
        "outputs" => Dict{String,Any}(
            "csvs"                   => collect(String, outputs),
            "figures_regenerated_by" => String(figures_regenerated_by),
        ),
    )

    for (k, v) in extras
        data[String(k)] = v
    end

    path = joinpath(run_dir, "meta.toml")
    open(path, "w") do io
        TOML.print(io, data; sorted = true)
    end
    return path
end

# ─── internals ─────────────────────────────────────────────────────────────

function _git(root::AbstractString, args::AbstractString...)
    try
        strip(read(Cmd(`git -C $root $args`), String))
    catch
        ""
    end
end

function _parse_int(s::AbstractString)
    v = tryparse(Int, strip(s))
    return v === nothing ? 0 : v
end

_hostname() = try
    gethostname()
catch
    "unknown"
end

end # module
