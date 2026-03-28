# src/PlotError.jl — Convergence figures for CMAME paper.
#
# Produces PDF/TeX files in the twin_mortar/ root (next to main.tex).
#
# Usage (from IGAros/):
#   julia --project=. src/PlotError.jl                   # all cases
#   julia --project=. src/PlotError.jl plate_nc          # single case
#   julia --project=. src/PlotError.jl cylinders sphere  # two cases
#
# Available cases: plate_nc  plate_c  cylinders  sphere

import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "examples", "plate_with_hole.jl"))
include(joinpath(@__DIR__, "..", "examples", "concentric_cylinders.jl"))
include(joinpath(@__DIR__, "..", "examples", "pressurized_sphere.jl"))
include(joinpath(@__DIR__, "..", "examples", "bending_beam.jl"))
include(joinpath(@__DIR__, "..", "examples", "cz_cancellation.jl"))

using PGFPlotsX, LaTeXStrings, Printf

# ── Color palette (Okabe-Ito, colorblind-safe) ──────────────────────────────
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"""
\definecolor{oiBlue}{RGB}{0,114,178}
\definecolor{oiOrange}{RGB}{230,159,0}
\definecolor{oiGreen}{RGB}{0,158,115}
\definecolor{oiVermilion}{RGB}{213,94,0}
""")

# ── Visual style (shared) ────────────────────────────────────────────────────
const MARKS   = Dict(1 => "diamond*", 2 => "*",      3 => "square*",   4 => "triangle*")
const LSTYLES = Dict(1 => "solid",    2 => "solid",  3 => "dashed",    4 => "dotted")
const LABELS  = Dict(1 => L"p=1",    2 => L"p=2",   3 => L"p=3",      4 => L"p=4")
# p=1: vermilion, p=2: blue, p=3: green, p=4: orange
const COLORS  = Dict(1 => "oiVermilion", 2 => "oiBlue", 3 => "oiGreen", 4 => "oiOrange")

# Legacy 6-config colors: SP=vermilion, TM=blue, DP=green (paired by formulation)
const CFG_COLORS = ["oiVermilion", "oiVermilion", "oiBlue", "oiBlue", "oiGreen", "oiGreen"]

# ── 4-method labeling for §7 (SPMS, SPME, DPM, TM) ────────────────────────
const METHODS_4 = [
    ("SPMS", SinglePassFormulation(),  SegmentBasedIntegration()),
    ("SPME", SinglePassFormulation(),  ElementBasedIntegration()),
    ("DPM",  DualPassFormulation(),    SegmentBasedIntegration()),
    ("TM",   TwinMortarFormulation(),  ElementBasedIntegration()),
]
const M4_COLORS = ["oiVermilion", "oiOrange", "oiGreen", "oiBlue"]
const M4_MARKS  = ["triangle*", "diamond*", "pentagon*", "*"]
const M4_STYLES = ["densely dashed", "dashdotted", "dashdotted", "solid"]

# ── Generic error collector ──────────────────────────────────────────────────
"""
    collect_errors(solve_fn, degrees, exp_range) -> Dict{Int, Vector{Float64}}

Call `solve_fn(p, exp) -> (rel, abs)` for every (p, exp) combination and
return a dict of absolute L2 errors indexed by degree.
`exp_range` may be a UnitRange (same for all p) or a Dict{Int,<:AbstractRange}.
"""
function collect_errors(solve_fn, degrees, exp_range)
    errs = Dict{Int, Vector{Float64}}()
    for p in degrees
        exps = exp_range isa Dict ? exp_range[p] : exp_range
        errs[p] = Float64[]
        for e in exps
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
`h_vals` may be a Vector{Float64} (same for all p) or a Dict{Int,Vector{Float64}}.
"""
function make_axis(h_vals, errs, degrees, title_str;
                   ylabel_str = L"\|e_\sigma\|_{L^2(\Omega)}")
    elements = []

    for p in degrees
        h_vec = h_vals isa Dict ? h_vals[p] : h_vals
        e_vec = errs[p]
        n     = length(e_vec)

        # Data curve (appears in legend)
        push!(elements,
            Plot(
                PGFPlotsX.Options(
                    COLORS[p]    => nothing,
                    "mark"       => MARKS[p],
                    LSTYLES[p]   => nothing,
                    "mark size"  => "2pt",
                    "line width" => "1pt"),
                Coordinates(h_vec[1:n], e_vec)))

        # Reference slope O(h^p) — anchored at last two mesh levels, excluded from legend
        h_ref = h_vec[n-1:n]
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
            ylabel           = ylabel_str,
            title            = title_str,
            width            = "7.5cm",
            height           = "6.5cm",
            legend_pos       = "south east",
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
    pgfsave(joinpath(OUTDIR, "$stem.tex"), ax)   # always succeeds (plain text)
    try
        pgfsave(joinpath(OUTDIR, "$stem.pdf"), ax)
        println("Saved: $stem.{pdf,tex}")
    catch e
        @warn "PDF save failed (no LaTeX?): $e"
        println("Saved: $stem.tex  (PDF skipped)")
    end
end

# ════════════════════════════════════════════════════════════════════════════
# Case runners
# ════════════════════════════════════════════════════════════════════════════

function run_plate_nc()
    println("\n── Plate with hole: non-conforming (p=1,2,3,4) ───────────────────────")
    degrees   = [1, 2, 3, 4]
    exp_per_p = Dict(1 => 0:5, 2 => 0:5, 3 => 0:5, 4 => 0:5)
    plate_solver = function(p, e)
        if p == 1
            return solve_plate_p1(e)
        else
            return solve_plate(p, e; conforming=false)
        end
    end
    h_per_p = Dict(p => [0.5 / 2^e for e in exp_per_p[p]] for p in degrees)
    errs = collect_errors(plate_solver, degrees, exp_per_p)
    save_figure(make_axis(h_per_p, errs, degrees, "Plate with hole"), "convergence_plate_nonconforming")
end

function run_plate_c()
    println("\n── Plate with hole: conforming ──────────────────────────────────────────")
    exps = 0:5
    h    = [0.5 / 2^e for e in exps]
    errs = collect_errors((p,e)->solve_plate(p, e; conforming=true), [2,3,4], exps)
    save_figure(make_axis(h, errs, [2,3,4], "Conforming"), "convergence_plate_conforming")
end

function run_plate_rate_table()
    println("\n── Plate with hole: convergence rate table (L² stress + energy norm) ──")
    degrees   = [1, 2, 3, 4]
    exp_per_p = Dict(1 => 0:5, 2 => 0:5, 3 => 0:5, 4 => 0:5)

    for p in degrees
        exps = collect(exp_per_p[p])
        l2_errs = Float64[]; en_errs = Float64[]

        for e in exps
            if p == 1
                # p=1: direct CP geometry, only L² stress available
                _, l2a = solve_plate_p1(e)
                push!(l2_errs, l2a)
                push!(en_errs, NaN)
            else
                r = solve_plate_full(p, e; conforming=false)
                push!(l2_errs, r.l2_abs)
                push!(en_errs, r.en_abs)
            end
        end

        @printf("\n  p = %d:\n", p)
        @printf("  %6s  %14s  %8s  %14s  %8s\n", "exp", "||e||_L2", "rate", "||e||_E", "rate")
        @printf("  %s\n", "─"^56)
        for (i, e) in enumerate(exps)
            l2r = i > 1 ? log2(l2_errs[i-1] / l2_errs[i]) : NaN
            enr = i > 1 && !isnan(en_errs[i]) ? log2(en_errs[i-1] / en_errs[i]) : NaN
            @printf("  %6d  %14.4e  %8.2f  %14.4e  %8.2f\n",
                    e, l2_errs[i], l2r, en_errs[i], enr)
        end
    end
end

function run_plate_formulation()
    println("\n── Plate formulation comparison: SPMS, SPME, TM — p=2 ────────────────")

    p_ord     = 2
    exp_range = 0:5
    h_vals    = [0.5 / 2^e for e in exp_range]

    # ε settings: SP methods don't use ε; TM uses adaptive E*s2
    epss_sp = 0.0    # ignored by single-pass
    epss_tm = 0.0    # 0 = auto-scale in solve_plate_full

    methods = [
        ("SPMS", SinglePassFormulation(),  SegmentBasedIntegration(),  epss_sp),
        ("SPME", SinglePassFormulation(),  ElementBasedIntegration(), epss_sp),
        ("TM",   TwinMortarFormulation(),  ElementBasedIntegration(), epss_tm),
    ]

    l2_data = [Float64[] for _ in methods]
    en_data = [Float64[] for _ in methods]

    for (k, (label, form, strat, eps_k)) in enumerate(methods)
        for e in exp_range
            r = solve_plate_full(p_ord, e;
                    conforming=false, strategy=strat, formulation=form,
                    epss=eps_k)
            push!(l2_data[k], r.l2_abs)
            push!(en_data[k], r.en_abs)
            @printf("  %-8s exp=%d  L2=%.4e  E=%.4e\n", label, e, r.l2_abs, r.en_abs)
        end
    end

    colors = ["oiVermilion", "oiOrange", "oiBlue"]
    marks  = ["triangle*", "diamond*", "*"]
    styles = ["densely dashed", "dashdotted", "solid"]
    labels_m = [m[1] for m in methods]

    function _axis(data, ylabel_str, title_str)
        elements = []
        for (k, lbl) in enumerate(labels_m)
            push!(elements, Plot(
                PGFPlotsX.Options(
                    colors[k]     => nothing,
                    "mark"        => marks[k],
                    styles[k]     => nothing,
                    "mark size"   => "2pt",
                    "line width"  => "1pt"),
                Coordinates(h_vals, data[k])))
        end

        # O(h^p) slope anchored at last two TM points
        tm = data[3]; nn = length(tm)
        h_ref = h_vals[nn-1:nn]
        e_ref = tm[nn-1] .* (h_ref ./ h_ref[1]) .^ p_ord
        push!(elements, Plot(
            PGFPlotsX.Options("gray" => nothing, "no marks" => nothing,
                "thin" => nothing, "densely dotted" => nothing,
                "forget plot" => nothing),
            Coordinates(h_ref, e_ref)))

        @pgf Axis(
            {
                xmode            = "log",
                ymode            = "log",
                xlabel           = L"h",
                ylabel           = ylabel_str,
                title            = title_str,
                width            = "9cm",
                height           = "7cm",
                legend_pos       = "outer north east",
                legend_style     = "font=\\footnotesize, row sep=-2pt",
                tick_label_style = "font=\\footnotesize",
                label_style      = "font=\\small",
                title_style      = "font=\\small",
                grid             = "major",
                grid_style       = "{gray!25, line width=0.4pt}",
            },
            elements...,
            Legend(labels_m...)
        )
    end

    ax_l2 = _axis(l2_data,
        L"\|e_\sigma\|_{L^2}",
        "Plate with hole (\$p=$p_ord\$, \$L^2\$ stress)")
    ax_en = _axis(en_data,
        L"\|e\|_E",
        "Plate with hole (\$p=$p_ord\$, energy norm)")

    save_figure(ax_l2, "convergence_plate_formulation_l2")
    save_figure(ax_en, "convergence_plate_formulation_energy")
end

function run_cylinders()
    # p=1: direct mesh (CPs on arcs), ε=1e9
    # p≥2: NURBS exact circle + krefinement, ε=1e6
    println("\n── Concentric cylinders: non-conforming  (p=1: ε=1e9, p≥2: ε=1e6) ─────")
    degrees   = [1, 2, 3, 4]
    exp_per_p = Dict(1 => 0:5, 2 => 0:5, 3 => 0:4, 4 => 0:3)
    # solve_cylinder now returns 4-tuple; extract (stress_rel, stress_abs) for the plot
    cyl_solver = function(p, e)
        if p == 1
            l2r, l2a, _, _ = solve_cylinder_p1(e; epss=1e9)
        else
            l2r, l2a, _, _ = solve_cylinder(p, e; epss=1e6)
        end
        return l2r, l2a
    end
    h_per_p = Dict(p => [0.5 / 2^e for e in exp_per_p[p]] for p in degrees)
    errs    = collect_errors(cyl_solver, degrees, exp_per_p)
    save_figure(make_axis(h_per_p, errs, degrees, "Concentric cylinders"), "convergence_cylinders")
end

function run_sphere()
    # p=1: exp 0..3, ε=1e9 — direct mesh (CPs on sphere surface), n_ang_o=2^(exp+1)
    # p=2: exp 0..4, p=3: exp 0..3, p=4: exp 0..3 — ε=1e6, n_ang_o=2^exp
    println("\n── Pressurized sphere: non-conforming  (p=1: ε=1e9, p≥2: ε=1e6) ──────")
    degrees    = [1, 2, 3, 4]
    exp_per_p  = Dict(1 => 0:3, 2 => 0:4, 3 => 0:3, 4 => 0:3)
    sphere_solver = function(p, e)
        p == 1 && return solve_sphere_p1(e; epss=1e9)
        return solve_sphere(p, e; epss=1e6)
    end
    # h = angular element size for outer patch
    # p≥2: n_ang_o = 2^e  → h = 1/2^e
    # p=1: n_ang_o = 2^(e+1) → h = 1/2^(e+1) = 0.5/2^e
    h_per_p = Dict(
        1 => [1.0 / 2^(e+1) for e in exp_per_p[1]],
        2 => [1.0 / 2^e     for e in exp_per_p[2]],
        3 => [1.0 / 2^e     for e in exp_per_p[3]],
        4 => [1.0 / 2^e     for e in exp_per_p[4]],
    )
    errs = collect_errors(sphere_solver, degrees, exp_per_p)
    save_figure(make_axis(h_per_p, errs, degrees, "Pressurized sphere"), "convergence_sphere")
end

function run_sphere_l2disp()
    println("\n── Pressurized sphere: L2 displacement error (p=1,2,3,4) ──────────")
    degrees    = [1, 2, 3, 4]
    exp_per_p  = Dict(1 => 0:3, 2 => 0:4, 3 => 0:3, 4 => 0:3)
    sphere_solver = function(p, e)
        r = solve_sphere_diag(p, e; epss = p == 1 ? 1.0 : 1e6)
        return r.l2_rel, r.l2_abs
    end
    h_per_p = Dict(
        1 => [1.0 / 2^(e+1) for e in exp_per_p[1]],
        2 => [1.0 / 2^e     for e in exp_per_p[2]],
        3 => [1.0 / 2^e     for e in exp_per_p[3]],
        4 => [1.0 / 2^e     for e in exp_per_p[4]],
    )
    errs = collect_errors(sphere_solver, degrees, exp_per_p)
    save_figure(make_axis(h_per_p, errs, degrees, "Pressurized sphere";
                          ylabel_str = L"\|e_u\|_{L^2(\Omega)}"),
                "convergence_sphere_l2disp")
end

function run_formulation_comparison_plots()
    println("\n── Formulation comparison: SPMS, SPME, DPM, TM — p=1 cylinders ────────")

    exp_range = 0:5
    h_vals    = [0.5 / 2^e for e in exp_range]
    # Method-specific ε: SP methods don't use ε; DPM and TM need large ε
    # but DPM (segment-based) is more sensitive to over-stabilisation.
    epss_map = Dict("SPMS" => 0.0, "SPME" => 0.0, "DPM" => 1e3, "TM" => 1e9)

    # ── Collect errors (4 methods) ────────────────────────────────────────────
    l2_data = [Float64[] for _ in METHODS_4]
    en_data = [Float64[] for _ in METHODS_4]

    for (k, (label, form, strat)) in enumerate(METHODS_4)
        epss = epss_map[label]
        for e in exp_range
            l2, _, _, _, en, _ = solve_cylinder_p1(e;
                strategy=strat, formulation=form, epss=epss)
            push!(l2_data[k], l2)
            push!(en_data[k], en)
            @printf("  %-8s exp=%d  L2=%.4e  E=%.4e\n", label, e, l2, en)
        end
    end

    # ── Single-mesh reference ─────────────────────────────────────────────────
    l2_single = Float64[]; en_single = Float64[]
    for e in exp_range
        l2, _, _, _, en, _ = solve_cylinder_p1_single(e)
        push!(l2_single, l2); push!(en_single, en)
        @printf("  %-11s exp=%d  L2=%.4e  E=%.4e\n", "Single", e, l2, en)
    end

    # ── Axis builder ──────────────────────────────────────────────────────────
    function make_axis_fc(err_data, sm_vec, ylabel_str, title_str)
        elements = []
        legend_labels = String[]

        for (k, (cfg_label, _, _)) in enumerate(METHODS_4)
            push!(elements, Plot(
                PGFPlotsX.Options(
                    M4_COLORS[k]  => nothing,
                    "mark"        => M4_MARKS[k],
                    M4_STYLES[k]  => nothing,
                    "mark size"   => "2pt",
                    "line width"  => "1pt"),
                Coordinates(h_vals, err_data[k])))
            push!(legend_labels, cfg_label)
        end

        # Single-mesh reference (gray, x marks)
        push!(elements, Plot(
            PGFPlotsX.Options(
                "gray"       => nothing,
                "mark"       => "x",
                "solid"      => nothing,
                "mark size"  => "3pt",
                "line width" => "1pt"),
            Coordinates(h_vals, sm_vec)))
        push!(legend_labels, "Single mesh")

        # O(h^1) slope anchored at last two TM points
        tm_vec = err_data[4]; n = length(tm_vec)
        h_ref = h_vals[n-1:n]
        e_ref = tm_vec[n-1] .* (h_ref ./ h_ref[1]) .^ 1
        push!(elements, Plot(
            PGFPlotsX.Options("gray" => nothing, "no marks" => nothing,
                "thin" => nothing, "densely dotted" => nothing,
                "forget plot" => nothing),
            Coordinates(h_ref, e_ref)))

        @pgf Axis(
            {
                xmode            = "log",
                ymode            = "log",
                xlabel           = L"h",
                ylabel           = ylabel_str,
                title            = title_str,
                width            = "9cm",
                height           = "7cm",
                legend_pos       = "outer north east",
                legend_style     = "font=\\footnotesize, row sep=-2pt",
                tick_label_style = "font=\\footnotesize",
                label_style      = "font=\\small",
                title_style      = "font=\\small",
                grid             = "major",
                grid_style       = "{gray!25, line width=0.4pt}",
            },
            elements...,
            Legend(legend_labels...)
        )
    end

    ax_l2 = make_axis_fc(l2_data, l2_single,
        L"\|e_\sigma\|_{L^2} / \|\sigma_{\rm ex}\|_{L^2}",
        "Formulation comparison (\$p=1\$, \$L^2\$ stress)")
    ax_en = make_axis_fc(en_data, en_single,
        L"\|e\|_E / \|u_{\rm ex}\|_E",
        "Formulation comparison (\$p=1\$, energy norm)")

    save_figure(ax_l2, "convergence_formulation_l2")
    save_figure(ax_en, "convergence_formulation_energy")
end

function run_beam_convergence()
    println("\n── Bending beam: 4-method comparison (SPMS, SPME, DPM, TM), p=1, curved ──")

    exp_range = 0:3
    epss_map  = Dict("SPMS" => 0.0, "SPME" => 0.0, "DPM" => 1e3, "TM" => 1e5)
    h_vals    = [1.0 / 2^e for e in exp_range]
    n_lo = 3; n_up = 2   # 4:3 ratio

    disp_data = [Float64[] for _ in METHODS_4]

    for (k, (label, form, strat)) in enumerate(METHODS_4)
        epss = epss_map[label]
        for e in exp_range
            err_abs = try
                _, ea = solve_beam_p1(e;
                    epss=epss, strategy=strat, formulation=form,
                    n_x_lower_base=n_lo, n_x_upper_base=n_up, curved=true)
                ea
            catch ex
                @warn "  $label exp=$e FAILED: $ex"
                NaN
            end
            push!(disp_data[k], err_abs)
            @printf("  %-8s exp=%d  L2=%.4e\n", label, e, err_abs)
        end
    end

    elements = []
    legend_labels = String[]

    for (k, (cfg_label, _, _)) in enumerate(METHODS_4)
        push!(elements, Plot(
            PGFPlotsX.Options(
                M4_COLORS[k]  => nothing,
                "mark"        => M4_MARKS[k],
                M4_STYLES[k]  => nothing,
                "mark size"   => "2pt",
                "line width"  => "1pt"),
            Coordinates(h_vals, disp_data[k])))
        push!(legend_labels, cfg_label)
    end

    # O(h^2) reference slope anchored at last two TM points
    tm_vec = disp_data[4]; n = length(tm_vec)
    h_ref = h_vals[n-1:n]
    e_ref = tm_vec[n-1] .* (h_ref ./ h_ref[1]) .^ 2
    push!(elements, Plot(
        PGFPlotsX.Options("gray" => nothing, "no marks" => nothing,
            "thin" => nothing, "densely dotted" => nothing,
            "forget plot" => nothing),
        Coordinates(h_ref, e_ref)))

    ax = @pgf Axis(
        {
            xmode            = "log",
            ymode            = "log",
            xlabel           = L"h",
            ylabel           = L"\|e_u\|_{L^2} / \|u_{\rm ex}\|_{L^2}",
            title            = "Bending beam (\$p=1\$, curved interface)",
            width            = "9cm",
            height           = "7cm",
            legend_pos       = "outer north east",
            legend_style     = "font=\\footnotesize, row sep=-2pt",
            tick_label_style = "font=\\footnotesize",
            label_style      = "font=\\small",
            title_style      = "font=\\small",
            grid             = "major",
            grid_style       = "{gray!25, line width=0.4pt}",
        },
        elements...,
        Legend(legend_labels...)
    )

    save_figure(ax, "convergence_beam")
end

# ════════════════════════════════════════════════════════════════════════════
# Dispatch: run all cases, or only those listed in ARGS
# ════════════════════════════════════════════════════════════════════════════

function run_conditioning_plot()
    println("\n── System conditioning: κ(A) vs h, 4 methods (SPMS, SPME, DPM, TM) ────")

    exp_range = 0:3
    epss_map  = Dict("SPMS" => 0.0, "SPME" => 0.0, "DPM" => 1e3, "TM" => 1e9)
    h_vals    = [0.5 / 2^e for e in exp_range]

    elements = []
    legend_labels = String[]

    for (k, (cfg_label, form, strat)) in enumerate(METHODS_4)
        epss = epss_map[cfg_label]
        kv = Float64[]
        for e in exp_range
            κ = try
                k_val, _ = cylinder_p1_kappa(e; formulation=form, strategy=strat, epss=epss)
                k_val
            catch ex
                @warn "  $cfg_label exp=$e FAILED: $ex"
                NaN
            end
            push!(kv, κ)
            @printf("  %-8s exp=%d  κ=%.3e\n", cfg_label, e, κ)
        end
        push!(elements, Plot(
            PGFPlotsX.Options(
                M4_COLORS[k]  => nothing,
                "mark"        => M4_MARKS[k],
                M4_STYLES[k]  => nothing,
                "mark size"   => "2pt",
                "line width"  => "1pt"),
            Coordinates(h_vals, kv)))
        push!(legend_labels, cfg_label)
    end

    # Machine-precision reference κ = 10^16
    push!(elements, Plot(
        PGFPlotsX.Options(
            "gray"           => nothing,
            "no marks"       => nothing,
            "thin"           => nothing,
            "densely dotted" => nothing,
            "forget plot"    => nothing),
        Coordinates([h_vals[1], h_vals[end]], [1e16, 1e16])))

    ax = @pgf Axis(
        {
            xmode            = "log",
            ymode            = "log",
            xlabel           = L"h",
            ylabel           = L"\kappa(\mathbf{A})",
            title            = "System condition number (\$p=1\$)",
            width            = "9cm",
            height           = "7cm",
            legend_pos       = "outer north east",
            legend_style     = "font=\\footnotesize, row sep=-2pt",
            tick_label_style = "font=\\footnotesize",
            label_style      = "font=\\small",
            title_style      = "font=\\small",
            grid             = "major",
            grid_style       = "{gray!25, line width=0.4pt}",
            ymin             = 1e8,
            ymax             = 1e26,
        },
        elements...,
        Legend(legend_labels...),
    )

    save_figure(ax, "convergence_conditioning")
end

function run_nquad_sweep_disp()
    println("\n── NQUAD mortar sweep: L2-displacement vs mortar quadrature order ──────")

    p_ord     = 2
    exp_level = 3
    epss      = 1e6
    nquad_vec = collect(1:p_ord+5)     # 1..7 for p=2

    configs = [
        ("SPMS", SinglePassFormulation(),  SegmentBasedIntegration()),
        ("SPME", SinglePassFormulation(),  ElementBasedIntegration()),
        ("DPM",  DualPassFormulation(),    SegmentBasedIntegration()),
        ("TM",   TwinMortarFormulation(),  ElementBasedIntegration()),
    ]
    cfg_marks  = M4_MARKS
    cfg_styles = M4_STYLES
    cfg_colors = M4_COLORS

    # ── Collect data ──────────────────────────────────────────────────────────
    errs_data = Dict{String, Vector{Float64}}()
    for (label, form, strat) in configs
        errs_data[label] = Float64[]
        for nq in nquad_vec
            _, _, d_rel, _ = solve_cylinder(p_ord, exp_level;
                epss=epss, NQUAD_mortar=nq, strategy=strat, formulation=form)
            push!(errs_data[label], d_rel)
            @printf("  %-8s NQUAD=%d  L2_disp=%.4e\n", label, nq, d_rel)
        end
    end

    # ── Build axis ────────────────────────────────────────────────────────────
    elements = []
    legend_labels = String[]

    for (k, (label, _, _)) in enumerate(configs)
        push!(elements, Plot(
            PGFPlotsX.Options(
                cfg_colors[k]  => nothing,
                "mark"         => cfg_marks[k],
                cfg_styles[k]  => nothing,
                "mark size"    => "2pt",
                "line width"   => "1pt"),
            Coordinates(nquad_vec, errs_data[label])))
        push!(legend_labels, label)
    end

    # Vertical reference line at NQUAD = p (sufficient quadrature threshold)
    y_lims = extrema(vcat(values(errs_data)...))
    push!(elements, Plot(
        PGFPlotsX.Options(
            "gray" => nothing, "no marks" => nothing,
            "thin" => nothing, "densely dotted" => nothing,
            "forget plot" => nothing),
        Coordinates([p_ord, p_ord], [y_lims[1]*0.5, y_lims[2]*2])))

    ax = @pgf Axis(
        {
            xmode            = "linear",
            ymode            = "log",
            xlabel           = L"N_{\rm GP}^{\rm mortar}",
            ylabel           = L"\|e_u\|_{L^2} / \|u_{\rm ex}\|_{L^2}",
            title            = "Integration accuracy: \$p=$p_ord\$, exp=$exp_level",
            width            = "9cm",
            height           = "7cm",
            legend_pos       = "outer north east",
            legend_style     = "font=\\footnotesize, row sep=-2pt",
            tick_label_style = "font=\\footnotesize",
            label_style      = "font=\\small",
            title_style      = "font=\\small",
            grid             = "major",
            grid_style       = "{gray!25, line width=0.4pt}",
            xtick            = "{1,2,3,4,5,6,7}",
        },
        elements...,
        Legend(legend_labels...),
    )

    save_figure(ax, "convergence_nquad_disp")
end

function run_disp_convergence_p2()
    println("\n── L2-displacement convergence: p=2, 4 methods (SPMS, SPME, DPM, TM) ──")

    p_ord     = 2
    exp_range = 0:5
    epss      = 1e6
    h_vals    = [0.5 / 2^e for e in exp_range]

    # ── Collect errors ────────────────────────────────────────────────────────
    disp_data = [Float64[] for _ in METHODS_4]

    for (k, (label, form, strat)) in enumerate(METHODS_4)
        for e in exp_range
            _, _, d_rel, _ = solve_cylinder(p_ord, e;
                epss=epss, strategy=strat, formulation=form)
            push!(disp_data[k], d_rel)
            @printf("  %-8s exp=%d  L2_disp=%.4e\n", label, e, d_rel)
        end
    end

    # ── Build axis ────────────────────────────────────────────────────────────
    elements = []
    legend_labels = String[]

    for (k, (cfg_label, _, _)) in enumerate(METHODS_4)
        push!(elements, Plot(
            PGFPlotsX.Options(
                M4_COLORS[k]  => nothing,
                "mark"        => M4_MARKS[k],
                M4_STYLES[k]  => nothing,
                "mark size"   => "2pt",
                "line width"  => "1pt"),
            Coordinates(h_vals, disp_data[k])))
        push!(legend_labels, cfg_label)
    end

    # O(h^{p+1}) = O(h^3) reference slope anchored at last two TM points
    tm_vec = disp_data[4]; n = length(tm_vec)
    h_ref = h_vals[n-1:n]
    e_ref = tm_vec[n-1] .* (h_ref ./ h_ref[1]) .^ (p_ord + 1)
    push!(elements, Plot(
        PGFPlotsX.Options("gray" => nothing, "no marks" => nothing,
            "thin" => nothing, "densely dotted" => nothing,
            "forget plot" => nothing),
        Coordinates(h_ref, e_ref)))

    ax = @pgf Axis(
        {
            xmode            = "log",
            ymode            = "log",
            xlabel           = L"h",
            ylabel           = L"\|e_u\|_{L^2} / \|u_{\rm ex}\|_{L^2}",
            title            = "Displacement convergence (\$p=2\$)",
            width            = "9cm",
            height           = "7cm",
            legend_pos       = "outer north east",
            legend_style     = "font=\\footnotesize, row sep=-2pt",
            tick_label_style = "font=\\footnotesize",
            label_style      = "font=\\small",
            title_style      = "font=\\small",
            grid             = "major",
            grid_style       = "{gray!25, line width=0.4pt}",
        },
        elements...,
        Legend(legend_labels...)
    )

    save_figure(ax, "convergence_disp_p2")
end

function run_beam_normal_strategy()
    println("\n── Bending beam: DualMortar × 3 NormalStrategies (elem-based, p=1, curved, 4:3) ─")

    exp_range = 0:3
    epss      = 1e5
    h_vals    = [1.0 / 2^e for e in exp_range]

    strategies = [
        ("Slave",   SlaveNormal()),
        ("Master",  MasterNormal()),
        ("Average", AverageNormal()),
    ]
    strat_colors = ["oiBlue",  "oiOrange", "oiGreen"]
    strat_marks  = ["*",       "square*",  "diamond*"]
    strat_styles = ["solid",   "dashed",   "dashdotted"]

    n_lo = 3; n_up = 2   # 4:3 ratio: lower has 3·2^e, upper has 2·2^e x-elements

    disp_data = [Float64[] for _ in strategies]

    for (k, (label, ns)) in enumerate(strategies)
        for e in exp_range
            _, err_abs = solve_beam_p1(e;
                epss=epss, strategy=ElementBasedIntegration(),
                formulation=DualPassFormulation(), normal_strategy=ns,
                n_x_lower_base=n_lo, n_x_upper_base=n_up, curved=true)
            push!(disp_data[k], err_abs)
            @printf("  %-8s exp=%d  L2=%.4e\n", label, e, err_abs)
        end
    end

    elements = []
    legend_labels = String[]

    for (k, (label, _)) in enumerate(strategies)
        push!(elements, Plot(
            PGFPlotsX.Options(
                strat_colors[k] => nothing,
                "mark"          => strat_marks[k],
                strat_styles[k] => nothing,
                "mark size"     => "2pt",
                "line width"    => "1pt"),
            Coordinates(h_vals, disp_data[k])))
        push!(legend_labels, "DP-Elm ($label)")
    end

    # O(h^2) reference slope anchored at last two Average points
    avg = disp_data[3]; n = length(avg)
    h_ref = h_vals[n-1:n]
    e_ref = avg[n-1] .* (h_ref ./ h_ref[1]) .^ 2
    push!(elements, Plot(
        PGFPlotsX.Options("gray" => nothing, "no marks" => nothing,
            "thin" => nothing, "densely dotted" => nothing,
            "forget plot" => nothing),
        Coordinates(h_ref, e_ref)))

    ax = @pgf Axis(
        {
            xmode            = "log",
            ymode            = "log",
            xlabel           = L"h",
            ylabel           = L"\|e_u\|_{L^2} / \|u_{\rm ex}\|_{L^2}",
            title            = "Normal strategy: DualMortar, \$p=1\$, curved arc",
            width            = "9cm",
            height           = "7cm",
            legend_pos       = "outer north east",
            legend_style     = "font=\\footnotesize, row sep=-2pt",
            tick_label_style = "font=\\footnotesize",
            label_style      = "font=\\small",
            title_style      = "font=\\small",
            grid             = "major",
            grid_style       = "{gray!25, line width=0.4pt}",
        },
        elements...,
        Legend(legend_labels...)
    )

    save_figure(ax, "convergence_beam_normal_strategy")
end

# ════════════════════════════════════════════════════════════════════════════
# NEW: ε-sensitivity plot (cylinders, p=1,2,3,4)
# ════════════════════════════════════════════════════════════════════════════

function run_eps_sensitivity()
    println("\n── ε-sensitivity: concentric cylinders, p=1,2,3,4 ─────────────────────")

    degrees   = [1, 2, 3, 4]
    exp_level = 3
    eps_range = 10 .^ (-2:0.5:7)

    elements = []
    legend_labels = String[]

    for p_ord in degrees
        errs = Float64[]
        for eps in eps_range
            try
                rel, _ = if p_ord == 1
                    solve_cylinder_p1(exp_level; epss=Float64(eps))
                else
                    solve_cylinder(p_ord, exp_level; epss=Float64(eps))
                end
                push!(errs, rel)
            catch
                push!(errs, NaN)
            end
            @printf("  p=%d  ε=%.1e  err=%.4e\n", p_ord, eps, errs[end])
        end

        push!(elements, Plot(
            PGFPlotsX.Options(
                COLORS[p_ord] => nothing,
                "mark"        => MARKS[p_ord],
                LSTYLES[p_ord] => nothing,
                "mark size"   => "2pt",
                "line width"  => "1pt"),
            Coordinates(collect(eps_range), errs)))
        push!(legend_labels, "p=$p_ord")
    end

    ax = @pgf Axis(
        {
            xmode            = "log",
            ymode            = "log",
            xlabel           = L"\varepsilon",
            ylabel           = L"\|e_\sigma\|_{L^2} / \|\sigma_{\rm ex}\|_{L^2}",
            title            = "\\varepsilon-sensitivity (exp=$exp_level)",
            width            = "9cm",
            height           = "7cm",
            legend_pos       = "outer north east",
            legend_style     = "font=\\footnotesize, row sep=-2pt",
            tick_label_style = "font=\\footnotesize",
            label_style      = "font=\\small",
            title_style      = "font=\\small",
            grid             = "major",
            grid_style       = "{gray!25, line width=0.4pt}",
            unbounded_coords = "jump",
        },
        elements...,
        Legend(legend_labels...)
    )

    save_figure(ax, "convergence_eps_sensitivity")
end

# ════════════════════════════════════════════════════════════════════════════
# NEW: Lagrange multiplier λ^(s) comparison (flat patch test, p=1, 4 methods)
# ════════════════════════════════════════════════════════════════════════════

function run_patch_lambda()
    println("\n── Lagrange multiplier λ^(s): flat patch test, p=1, 4 methods ─────────")

    n_s = 4; n_m = 7   # enough CPs to see oscillation
    E   = 1000.0

    configs = [
        ("SPMS", SinglePassFormulation(),  SegmentBasedIntegration(), 0.0),
        ("SPME", SinglePassFormulation(),  ElementBasedIntegration(), 0.0),
        ("DPM",  DualPassFormulation(),    SegmentBasedIntegration(), 1.0),
        ("TM",   TwinMortarFormulation(),  ElementBasedIntegration(), 1.0),
    ]

    elements = []
    legend_labels = String[]

    for (k, (label, form, strat, eps)) in enumerate(configs)
        r = flat_patch_test(n_s, n_m; E=E, epss=eps, NQUAD_mortar=3,
                            strategy=strat, formulation=form)
        _, Lambda = solve_mortar(r.K_bc, r.C, r.Z, r.F_bc)

        # Lambda is blocked by direction: [CP1..CPn for d=1, CP1..CPn for d=2]
        # Extract slave-side normal component (x in code = normal to vertical interface)
        if form isa SinglePassFormulation
            n_lam = length(Lambda) ÷ 2   # 2 dirs × n_slave_cps
            lam_s_n = Lambda[1:n_lam]    # d=1 block (normal direction)
        else
            # TM/DPM: slave block + master block, each blocked by direction
            n_lam = length(Lambda) ÷ 4   # 2 dirs × 2 sides × n_cps
            lam_s_n = Lambda[1:n_lam]    # slave, d=1 block
        end

        # Interface coordinate of slave CPs (Patch 1, facet 2 = x=0.5)
        # Code interface runs y ∈ [0,1]; paper geometry runs x ∈ [0, L1=4]
        L1 = 4.0
        y_slave = Float64[]
        for cp in r.Pc
            if r.B[cp, 1] ≈ 0.5 && cp <= 2*(n_s+1)   # Patch 1 CPs at interface
                push!(y_slave, r.B[cp, 2] * L1)        # scale to paper coords
            end
        end
        sort!(y_slave)

        n_pts = min(length(lam_s_n), length(y_slave))
        @printf("  %-6s n_lam=%d, n_pts=%d, λ_n=%s\n", label, length(lam_s_n), length(y_slave),
                string(round.(lam_s_n, sigdigits=4)))

        push!(elements, Plot(
            PGFPlotsX.Options(
                M4_COLORS[k]  => nothing,
                "mark"        => M4_MARKS[k],
                M4_STYLES[k]  => nothing,
                "mark size"   => "2pt",
                "line width"  => "1pt"),
            Coordinates(y_slave[1:n_pts], lam_s_n[1:n_pts])))
        push!(legend_labels, label)
    end

    # Exact multiplier reference (|λ| = 1.0 everywhere for unit traction)
    push!(elements, Plot(
        PGFPlotsX.Options(
            "gray"           => nothing,
            "no marks"       => nothing,
            "thin"           => nothing,
            "densely dotted" => nothing,
            "forget plot"    => nothing),
        Coordinates([0.0, 4.0], [1.0, 1.0])))

    ax = @pgf Axis(
        {
            xlabel           = L"x",
            ylabel           = L"\|\boldsymbol{\lambda}^{(\mathrm{s})}\|",
            title            = "Lagrange multiplier (\$p=1\$)",
            width            = "9cm",
            height           = "7cm",
            legend_pos       = "outer north east",
            legend_style     = "font=\\footnotesize, row sep=-2pt",
            tick_label_style = "font=\\footnotesize",
            label_style      = "font=\\small",
            title_style      = "font=\\small",
            grid             = "major",
            grid_style       = "{gray!25, line width=0.4pt}",
        },
        elements...,
        Legend(legend_labels...)
    )

    save_figure(ax, "convergence_patch_lambda")
end

# ════════════════════════════════════════════════════════════════════════════
# §7.3 Concentric cylinders: SPMS vs TM convergence (L² disp + energy norm)
# ════════════════════════════════════════════════════════════════════════════

"""
    _cyl_solve(p, e; formulation, strategy, epss) -> (l2_disp_abs, en_abs)

Helper: solve cylinder at given (p, e) and return L² displacement and energy
norm absolute errors.
"""
function _cyl_solve(p::Int, e::Int;
    formulation::FormulationStrategy = TwinMortarFormulation(),
    strategy::IntegrationStrategy    = ElementBasedIntegration(),
    epss::Float64                    = 1e6,
)
    if p == 1
        _, _, _, d_abs, _, en_abs = solve_cylinder_p1(e;
            epss=epss, strategy=strategy, formulation=formulation)
    else
        _, _, _, d_abs, _, en_abs = solve_cylinder(p, e;
            epss=epss, strategy=strategy, formulation=formulation)
    end
    return d_abs, en_abs
end

function run_cyl_spms_tm()
    println("\n── Concentric cylinders: SPMS vs TM convergence (L² disp + energy) ─────")
    degrees   = [1, 2, 3, 4]
    # TM: full range; SPMS: truncated where conditioning fails
    exp_tm   = Dict(1 => 0:5, 2 => 0:5, 3 => 0:4, 4 => 0:3)
    exp_spms = Dict(1 => 0:5, 2 => 0:3, 3 => 0:1, 4 => 0:2)
    h_tm     = Dict(p => [0.5 / 2^e for e in exp_tm[p]]   for p in degrees)
    h_spms   = Dict(p => [0.5 / 2^e for e in exp_spms[p]] for p in degrees)

    # ε settings per method (SPMS: ε irrelevant for single-pass; TM: tuned per p)
    epss_spms = Dict(1 => 0.0, 2 => 0.0, 3 => 0.0, 4 => 0.0)
    epss_tm   = Dict(1 => 1e2, 2 => 1e6, 3 => 1e6, 4 => 1e6)

    # ── Collect errors ────────────────────────────────────────────────────────
    spms_disp = Dict{Int, Vector{Float64}}()
    spms_en   = Dict{Int, Vector{Float64}}()
    tm_disp   = Dict{Int, Vector{Float64}}()
    tm_en     = Dict{Int, Vector{Float64}}()

    for p in degrees
        spms_disp[p] = Float64[]; spms_en[p] = Float64[]
        for e in exp_spms[p]
            d, en = _cyl_solve(p, e;
                formulation=SinglePassFormulation(),
                strategy=SegmentBasedIntegration(),
                epss=epss_spms[p])
            push!(spms_disp[p], d); push!(spms_en[p], en)
            @printf("  SPMS  p=%d  exp=%d  disp=%.4e  energy=%.4e\n", p, e, d, en)
        end

        tm_disp[p] = Float64[]; tm_en[p] = Float64[]
        for e in exp_tm[p]
            d, en = _cyl_solve(p, e;
                formulation=TwinMortarFormulation(),
                strategy=ElementBasedIntegration(),
                epss=epss_tm[p])
            push!(tm_disp[p], d); push!(tm_en[p], en)
            @printf("  TM    p=%d  exp=%d  disp=%.4e  energy=%.4e\n", p, e, d, en)
        end
    end

    # ── Axis builder (reusable for each subfigure) ─────────────────────────────
    function _conv_axis(errs, h_map, ylabel_str, title_str;
                        ylims::Union{Nothing,Tuple{Float64,Float64}} = nothing)
        elements = []
        for p in degrees
            h_vec = h_map[p]; e_vec = errs[p]; n = length(e_vec)
            push!(elements, Plot(
                PGFPlotsX.Options(
                    COLORS[p] => nothing, "mark" => MARKS[p],
                    LSTYLES[p] => nothing, "mark size" => "2pt",
                    "line width" => "1pt"),
                Coordinates(h_vec[1:n], e_vec)))
            # O(h^p) reference slope (need ≥2 points)
            if n >= 2
                h_ref = h_vec[n-1:n]
                e_ref = e_vec[n-1] .* (h_ref ./ h_ref[1]) .^ p
                push!(elements, Plot(
                    PGFPlotsX.Options("gray" => nothing, "no marks" => nothing,
                        "thin" => nothing, "densely dotted" => nothing,
                        "forget plot" => nothing),
                    Coordinates(h_ref, e_ref)))
            end
        end
        opts = PGFPlotsX.Options(
            "xmode"            => "log",
            "ymode"            => "log",
            "xlabel"           => L"h",
            "ylabel"           => ylabel_str,
            "title"            => title_str,
            "width"            => "7.5cm",
            "height"           => "6.5cm",
            "legend pos"       => "south east",
            "legend style"     => "font=\\footnotesize, row sep=-2pt",
            "tick label style" => "font=\\footnotesize",
            "label style"      => "font=\\small",
            "title style"      => "font=\\small",
            "grid"             => "major",
            "grid style"       => "{gray!25, line width=0.4pt}",
        )
        if ylims !== nothing
            opts["ymin"] = ylims[1]
            opts["ymax"] = ylims[2]
        end
        @pgf Axis(opts, elements..., Legend([LABELS[p] for p in degrees]...))
    end

    # ── Shared y-axis ranges so (a) and (b) subfigures are directly comparable ─
    all_disp = vcat(values(spms_disp)..., values(tm_disp)...)
    all_en   = vcat(values(spms_en)...,   values(tm_en)...)
    yl_disp  = (10.0^floor(log10(minimum(all_disp))), 10.0^ceil(log10(maximum(all_disp))))
    yl_en    = (10.0^floor(log10(minimum(all_en))),   10.0^ceil(log10(maximum(all_en))))

    # ── Generate 4 figures ─────────────────────────────────────────────────────
    save_figure(_conv_axis(spms_disp, h_spms, L"\|e_u\|_{L^2(\Omega)}", "SPMS"; ylims=yl_disp), "convergence_cyl_spms_l2")
    save_figure(_conv_axis(tm_disp,   h_tm,   L"\|e_u\|_{L^2(\Omega)}", "TM";   ylims=yl_disp), "convergence_cyl_tm_l2")
    save_figure(_conv_axis(spms_en,   h_spms, L"\|e\|_E",               "SPMS"; ylims=yl_en),   "convergence_cyl_spms_energy")
    save_figure(_conv_axis(tm_en,     h_tm,   L"\|e\|_E",               "TM";   ylims=yl_en),   "convergence_cyl_tm_energy")

    # ── Rate table (TM only — SPMS too short for meaningful rates at high p) ──
    println("\n── Concentric cylinders: TM convergence rate table ──")
    for p in degrees
        exps = collect(exp_tm[p])
        println("\n  p = $p:")
        @printf("  %5s  %14s  %6s  %14s  %6s\n",
                "exp", "||e_u||_L2", "rate", "||e||_E", "rate")
        @printf("  %s\n", "─"^54)
        prev_d = NaN; prev_e = NaN
        for (i, e) in enumerate(exps)
            rd = isnan(prev_d) ? NaN : log2(prev_d / tm_disp[p][i])
            re = isnan(prev_e) ? NaN : log2(prev_e / tm_en[p][i])
            @printf("  %5d  %14.4e  %6.2f  %14.4e  %6.2f\n",
                    e, tm_disp[p][i], rd, tm_en[p][i], re)
            prev_d = tm_disp[p][i]; prev_e = tm_en[p][i]
        end
    end
end

# ════════════════════════════════════════════════════════════════════════════
# §7.3 Force-moment table (concentric cylinders)
# ════════════════════════════════════════════════════════════════════════════

function run_cyl_moments()
    println("\n── Concentric cylinders: force-moment analysis (p=2, exp=2) ──")

    p_ord = 2; exp_level = 2
    d = _cyl_setup(p_ord, exp_level; epss=1e6)

    pair1 = d.pairs_tm[1]   # pass 1: slave=inner, master=outer
    pair2 = d.pairs_tm[2]   # pass 2: slave=outer, master=inner
    nq = p_ord + 1   # NQUAD = p+1

    methods = [
        ("SPMS", nq, SegmentBasedIntegration(),  false),
        ("SPME", nq, ElementBasedIntegration(),  false),
        ("DPM",  nq, SegmentBasedIntegration(),  true),
        ("TM",   nq, ElementBasedIntegration(),  true),
    ]

    @printf("Moments with uniform test multiplier λ = 1, dim=1 (x)\n")
    @printf("NQUAD_mortar = %d\n\n", nq)
    @printf("%-8s  %12s  %12s  %14s  %14s  %14s\n",
            "Method", "δ₀", "δ₁", "δ₂ (pass 1)", "δ₂ (pass 2)", "δ₂ (sum)")
    @printf("%s\n", "─"^80)

    for (label, nq_use, strat, two_pass) in methods
        D1, M12, s1, m1 = build_mortar_mass_matrices(
            pair1, d.p_mat, d.n_mat_ref, d.KV_ref, d.P_ref, d.B_ref,
            d.nnp, d.nsd, d.npd, nq_use, strat)
        mom1 = compute_force_moments(D1, M12, s1, m1, d.B_ref; dim=1)

        if two_pass
            D2, M21, s2, m2 = build_mortar_mass_matrices(
                pair2, d.p_mat, d.n_mat_ref, d.KV_ref, d.P_ref, d.B_ref,
                d.nnp, d.nsd, d.npd, nq_use, strat)
            mom2 = compute_force_moments(D2, M21, s2, m2, d.B_ref; dim=1)
            δ2_sum = mom1.δ_2 + mom2.δ_2
            @printf("%-8s  %12.2e  %12.2e  %+14.4e  %+14.4e  %+14.4e\n",
                    label, abs(mom1.δ_0), abs(mom1.δ_1), mom1.δ_2, mom2.δ_2, δ2_sum)
        else
            @printf("%-8s  %12.2e  %12.2e  %+14.4e  %14s  %14s\n",
                    label, abs(mom1.δ_0), abs(mom1.δ_1), mom1.δ_2, "—", "—")
        end
    end
end

# ════════════════════════════════════════════════════════════════════════════
# §7.3 Cylinder conditioning: SPMS(p=1..4) vs TM(p=1..4) vs h
# ════════════════════════════════════════════════════════════════════════════

function run_cyl_conditioning()
    println("\n── Concentric cylinders: κ(A) for SPMS & TM, p=1,2,3,4 ──────────────")
    degrees = [1, 2, 3, 4]
    # exp ranges must stay within tractable SVD sizes
    exp_range = Dict(1 => 0:4, 2 => 0:3, 3 => 0:2, 4 => 0:2)

    epss_tm   = Dict(1 => 1e2, 2 => 1e6, 3 => 1e6, 4 => 1e6)

    # ── Collect data ──────────────────────────────────────────────────────────
    kappa_spms = Dict{Int, Vector{Float64}}()
    kappa_tm   = Dict{Int, Vector{Float64}}()
    h_map      = Dict{Int, Vector{Float64}}()

    for p in degrees
        kappa_spms[p] = Float64[]
        kappa_tm[p]   = Float64[]
        h_map[p]      = [0.5 / 2^e for e in exp_range[p]]

        for e in exp_range[p]
            # SPMS: no ε-stabilisation (saddle-point)
            κs, _ = try
                cylinder_kappa(p, e;
                    formulation=SinglePassFormulation(),
                    strategy=SegmentBasedIntegration(),
                    epss=0.0)
            catch ex
                @warn "  SPMS p=$p exp=$e FAILED: $ex"
                (NaN, 0)
            end
            push!(kappa_spms[p], κs)
            @printf("  SPMS(p=%d)  exp=%d  κ=%.3e\n", p, e, κs)

            # TM: with ε-regularisation
            κt, _ = try
                cylinder_kappa(p, e;
                    formulation=TwinMortarFormulation(),
                    strategy=ElementBasedIntegration(),
                    epss=epss_tm[p])
            catch ex
                @warn "  TM   p=$p exp=$e FAILED: $ex"
                (NaN, 0)
            end
            push!(kappa_tm[p], κt)
            @printf("  TM(p=%d)    exp=%d  κ=%.3e\n", p, e, κt)
        end
    end

    # ── Build PGFPlots axis ───────────────────────────────────────────────────
    elements = []
    legend_labels = String[]

    # SPMS curves (dashed)
    for p in degrees
        push!(elements, Plot(
            PGFPlotsX.Options(
                COLORS[p]    => nothing,
                "mark"       => MARKS[p],
                "densely dashed" => nothing,
                "mark size"  => "2pt",
                "line width" => "1pt"),
            Coordinates(h_map[p], kappa_spms[p])))
        push!(legend_labels, "SPMS (\$p=$p\$)")
    end

    # TM curves (solid)
    for p in degrees
        push!(elements, Plot(
            PGFPlotsX.Options(
                COLORS[p]    => nothing,
                "mark"       => MARKS[p],
                "solid"      => nothing,
                "mark size"  => "2.5pt",
                "line width" => "1.2pt"),
            Coordinates(h_map[p], kappa_tm[p])))
        push!(legend_labels, "TM (\$p=$p\$)")
    end

    # Machine-precision reference κ = 10^16
    all_h = vcat(values(h_map)...)
    push!(elements, Plot(
        PGFPlotsX.Options(
            "gray"           => nothing,
            "no marks"       => nothing,
            "thin"           => nothing,
            "densely dotted" => nothing,
            "forget plot"    => nothing),
        Coordinates([minimum(all_h), maximum(all_h)], [1e16, 1e16])))

    ax = @pgf Axis(
        {
            xmode            = "log",
            ymode            = "log",
            xlabel           = L"h",
            ylabel           = L"\kappa(\mathbf{A})",
            width            = "9cm",
            height           = "7cm",
            legend_columns   = 2,
            legend_style     = raw"font=\scriptsize, row sep=-2pt, fill opacity=0.85, text opacity=1, draw=gray!50, at={(0.5,-0.22)}, anchor=north",
            tick_label_style = "font=\\footnotesize",
            label_style      = "font=\\small",
            title_style      = "font=\\small",
            grid             = "major",
            grid_style       = "{gray!25, line width=0.4pt}",
            ymin             = 1e4,
            ymax             = 1e22,
        },
        elements...,
        Legend(legend_labels...),
    )

    save_figure(ax, "convergence_cyl_conditioning")
end

# ════════════════════════════════════════════════════════════════════════════
# §7.3 Cylinder conditioning vs ε: TM(p=1..4) curves + SPMS(p=1..4) lines
# ════════════════════════════════════════════════════════════════════════════

function run_cyl_cond_eps()
    println("\n── Concentric cylinders: κ(A) vs ε, TM + SPMS reference, p=1,2,3,4 ──")
    degrees   = [1, 2, 3, 4]
    exp_level = 3                          # same as eps_sensitivity (Fig. 12)
    eps_range = [10.0^k for k in -2:7]

    # ── SPMS reference κ (one value per p, no ε dependence) ──────────────────
    kappa_spms = Dict{Int, Float64}()
    for p in degrees
        κs, _ = cylinder_kappa(p, exp_level;
            formulation=SinglePassFormulation(),
            strategy=SegmentBasedIntegration(),
            epss=0.0)
        kappa_spms[p] = κs
        @printf("  SPMS(p=%d)  exp=%d  κ=%.3e\n", p, exp_level, κs)
    end

    # ── TM κ vs ε ────────────────────────────────────────────────────────────
    kappa_tm = Dict{Int, Vector{Float64}}()
    for p in degrees
        kappa_tm[p] = Float64[]
        for eps in eps_range
            κt, _ = try
                cylinder_kappa(p, exp_level;
                    formulation=TwinMortarFormulation(),
                    strategy=ElementBasedIntegration(),
                    epss=eps)
            catch ex
                @warn "  TM p=$p eps=$eps FAILED: $ex"
                (NaN, 0)
            end
            push!(kappa_tm[p], κt)
            @printf("  TM(p=%d)    eps=%.0e  κ=%.3e\n", p, eps, κt)
        end
    end

    # ── Build axis ────────────────────────────────────────────────────────────
    elements = []
    legend_labels = String[]

    # TM curves (solid, with markers)
    for p in degrees
        push!(elements, Plot(
            PGFPlotsX.Options(
                COLORS[p]    => nothing,
                "mark"       => MARKS[p],
                LSTYLES[p]   => nothing,
                "mark size"  => "2pt",
                "line width" => "1pt"),
            Coordinates(eps_range, kappa_tm[p])))
        push!(legend_labels, "TM (\$p=$p\$)")
    end

    # SPMS horizontal reference lines (dashed, no markers)
    for p in degrees
        push!(elements, Plot(
            PGFPlotsX.Options(
                COLORS[p]        => nothing,
                "no marks"       => nothing,
                "densely dashed" => nothing,
                "line width"     => "0.8pt"),
            Coordinates([eps_range[1], eps_range[end]], [kappa_spms[p], kappa_spms[p]])))
        push!(legend_labels, "SPMS (\$p=$p\$)")
    end

    ax = @pgf Axis(
        {
            xmode            = "log",
            ymode            = "log",
            xlabel           = L"\varepsilon",
            ylabel           = L"\kappa(\mathbf{A})",
            width            = "9cm",
            height           = "7cm",
            legend_columns   = 2,
            legend_style     = raw"font=\scriptsize, row sep=-2pt, fill opacity=0.85, text opacity=1, draw=gray!50, at={(0.5,-0.22)}, anchor=north",
            tick_label_style = "font=\\footnotesize",
            label_style      = "font=\\small",
            title_style      = "font=\\small",
            grid             = "major",
            grid_style       = "{gray!25, line width=0.4pt}",
        },
        elements...,
        Legend(legend_labels...),
    )

    save_figure(ax, "convergence_cyl_cond_eps")
end

# ════════════════════════════════════════════════════════════════════════════
# Dispatch
# ════════════════════════════════════════════════════════════════════════════

const ALL_CASES = ["plate_nc", "plate_c", "plate_formulation", "plate_rate_table", "cylinders", "sphere", "sphere_l2disp", "formulation", "conditioning", "nquad_disp", "disp_p2", "beam", "beam_normal_strategy", "eps_sensitivity", "patch_lambda", "cyl_spms_tm", "cyl_moments", "cyl_conditioning", "cyl_cond_eps"]
const CASE_FNS  = Dict(
    "plate_nc"             => run_plate_nc,
    "plate_c"              => run_plate_c,
    "plate_formulation"    => run_plate_formulation,
    "plate_rate_table"     => run_plate_rate_table,
    "cylinders"            => run_cylinders,
    "sphere"               => run_sphere,
    "sphere_l2disp"        => run_sphere_l2disp,
    "formulation"          => run_formulation_comparison_plots,
    "conditioning"         => run_conditioning_plot,
    "nquad_disp"           => run_nquad_sweep_disp,
    "disp_p2"              => run_disp_convergence_p2,
    "beam"                 => run_beam_convergence,
    "beam_normal_strategy" => run_beam_normal_strategy,
    "eps_sensitivity"      => run_eps_sensitivity,
    "patch_lambda"         => run_patch_lambda,
    "cyl_spms_tm"          => run_cyl_spms_tm,
    "cyl_moments"          => run_cyl_moments,
    "cyl_conditioning"     => run_cyl_conditioning,
    "cyl_cond_eps"         => run_cyl_cond_eps,
)

requested = isempty(ARGS) ? ALL_CASES : ARGS

for case in requested
    if haskey(CASE_FNS, case)
        CASE_FNS[case]()
    else
        @warn "Unknown case '$case'. Available: " * join(ALL_CASES, ", ")
    end
end
