# src/PlotError.jl — Convergence figures for CMAME paper.
#
# Produces six files in the twin_mortar/ root (next to main.tex):
#   convergence_plate_nonconforming.{pdf,tex}
#   convergence_plate_conforming.{pdf,tex}
#   convergence_cylinders.{pdf,tex}
#
# Usage (from IGAros/):
#   julia --project=. src/PlotError.jl

import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "examples", "plate_with_hole.jl"))
include(joinpath(@__DIR__, "..", "examples", "concentric_cylinders.jl"))

using PGFPlotsX, LaTeXStrings, Printf

# ── Visual style (shared) ────────────────────────────────────────────────────
const DEGREES = [2, 3, 4]
const MARKS   = Dict(2 => "*",          3 => "square*",   4 => "triangle*")
const LSTYLES = Dict(2 => "solid",      3 => "dashed",    4 => "dotted")
const LABELS  = Dict(2 => L"p=2",      3 => L"p=3",      4 => L"p=4")

# ── Generic error collector ──────────────────────────────────────────────────
"""
    collect_errors(solve_fn, degrees, exp_range) -> Dict{Int, Vector{Float64}}

Call `solve_fn(p, exp) -> (rel, abs)` for every (p, exp) combination and
return a dict of absolute L2 errors indexed by degree.
"""
function collect_errors(solve_fn, degrees, exp_range)
    errs = Dict{Int, Vector{Float64}}()
    for p in degrees
        errs[p] = Float64[]
        for e in exp_range
            _, abs_err = solve_fn(p, e)
            push!(errs[p], abs_err)
            @printf("  p=%d  exp=%d  ||e||_L2 = %.4e\n", p, e, abs_err)
        end
    end
    return errs
end

# ── Generic axis builder ─────────────────────────────────────────────────────
"""
    make_axis(h_vals, errs, degrees, title_str) -> Axis

Build a log-log convergence Axis. For each degree p, draws the data curve
plus a reference O(h^p) slope line anchored to the last two mesh levels.
"""
function make_axis(h_vals, errs, degrees, title_str)
    elements = []

    for p in degrees
        e_vec = errs[p]
        n     = length(e_vec)

        # Data curve (appears in legend)
        push!(elements,
            Plot(
                PGFPlotsX.Options(
                    "black"      => nothing,
                    "mark"       => MARKS[p],
                    LSTYLES[p]   => nothing,
                    "mark size"  => "2pt",
                    "line width" => "1pt"),
                Coordinates(h_vals[1:n], e_vec)))

        # Reference slope O(h^p) — last two mesh levels, excluded from legend
        h_ref = h_vals[n-1:n]
        e_ref = e_vec[n-1] .* (h_ref ./ h_ref[1]) .^ p
        push!(elements,
            Plot(
                PGFPlotsX.Options(
                    "gray"           => nothing,
                    "no marks"       => nothing,
                    "thin"           => nothing,
                    "densely dotted" => nothing,
                    "forget plot"    => nothing),
                Coordinates(h_ref, e_ref)))
    end

    @pgf Axis(
        {
            xmode            = "log",
            ymode            = "log",
            xlabel           = L"h",
            ylabel           = L"\|e_\sigma\|_{L^2(\Omega)}",
            title            = title_str,
            width            = "7.5cm",
            height           = "6.5cm",
            legend_pos       = "north west",
            legend_style     = "font=\\footnotesize, row sep=-2pt",
            tick_label_style = "font=\\footnotesize",
            label_style      = "font=\\small",
            title_style      = "font=\\small",
            grid             = "major",
            grid_style       = "{gray!25, line width=0.4pt}",
        },
        elements...,
        Legend([LABELS[p] for p in degrees]...)
    )
end

# ── Save helper ──────────────────────────────────────────────────────────────
const OUTDIR = joinpath(@__DIR__, "..", "..")   # twin_mortar/ root

function save_figure(ax, stem)
    pgfsave(joinpath(OUTDIR, "$stem.pdf"), ax)
    pgfsave(joinpath(OUTDIR, "$stem.tex"), ax)
    println("Saved: $stem.{pdf,tex}")
end

# ════════════════════════════════════════════════════════════════════════════
# 1. Plate with hole  (exp = 0:5,  h = s2 = 0.5/2^exp)
# ════════════════════════════════════════════════════════════════════════════
const PLATE_EXPS = 0:5
const H_PLATE    = [0.5 / 2^e for e in PLATE_EXPS]

println("\n── Plate with hole: non-conforming ──────────────────────────────────────")
errs_plate_nc = collect_errors( (p,e)->solve_plate(p, e; conforming=false), DEGREES, PLATE_EXPS)

println("\n── Plate with hole: conforming ──────────────────────────────────────────")
errs_plate_c = collect_errors( (p,e)->solve_plate(p, e; conforming=true), DEGREES, PLATE_EXPS)

save_figure(make_axis(H_PLATE, errs_plate_nc, DEGREES, "Non-conforming"),
            "convergence_plate_nonconforming")
save_figure(make_axis(H_PLATE, errs_plate_c,  DEGREES, "Conforming"),
            "convergence_plate_conforming")

# ════════════════════════════════════════════════════════════════════════════
# 2. Concentric cylinders  (exp = 0:4,  h = s_rad = 0.5/2^exp,  ε = 1e4)
#    p=4 is included; it stalls at exp≥3 due to conditioning — visible
#    in the figure as the expected pre-asymptotic plateau for high p.
# ════════════════════════════════════════════════════════════════════════════
const CYL_EXPS = 0:4
const H_CYL    = [0.5 / 2^e for e in CYL_EXPS]

println("\n── Concentric cylinders: non-conforming  (ε = 1e4) ─────────────────────")
errs_cyl = collect_errors((p,e)->solve_cylinder(p, e; epss=1e4), DEGREES, CYL_EXPS)

save_figure(make_axis(H_CYL, errs_cyl, DEGREES, "Concentric cylinders"),
            "convergence_cylinders")
