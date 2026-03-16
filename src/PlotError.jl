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

# Formulation colors: SP=vermilion, TM=blue, DP=green (6 configs, paired by formulation)
const CFG_COLORS = ["oiVermilion", "oiVermilion", "oiBlue", "oiBlue", "oiGreen", "oiGreen"]

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
function make_axis(h_vals, errs, degrees, title_str)
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
            ylabel           = L"\|e_\sigma\|_{L^2(\Omega)}",
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
    println("\n── Plate with hole: non-conforming ──────────────────────────────────────")
    exps = 0:5
    h    = [0.5 / 2^e for e in exps]
    errs = collect_errors((p,e)->solve_plate(p, e; conforming=false), [2,3,4], exps)
    save_figure(make_axis(h, errs, [2,3,4], "Non-conforming"), "convergence_plate_nonconforming")
end

function run_plate_c()
    println("\n── Plate with hole: conforming ──────────────────────────────────────────")
    exps = 0:5
    h    = [0.5 / 2^e for e in exps]
    errs = collect_errors((p,e)->solve_plate(p, e; conforming=true), [2,3,4], exps)
    save_figure(make_axis(h, errs, [2,3,4], "Conforming"), "convergence_plate_conforming")
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

function run_formulation_comparison_plots()
    println("\n── Formulation comparison: SP vs TM × Seg vs Elm, 2:1 and 3:2 ratios ──")

    exp_range = 0:5
    h_vals    = [0.5 / 2^e for e in exp_range]
    epss      = 1e9

    configs = [
        ("SP-Seg", SinglePassFormulation(), SegmentBasedIntegration()),
        ("SP-Elm", SinglePassFormulation(), ElementBasedIntegration()),
        ("TM-Seg", TwinMortarFormulation(), SegmentBasedIntegration()),
        ("TM-Elm", TwinMortarFormulation(), ElementBasedIntegration()),
        ("DP-Seg", DualPassFormulation(),   SegmentBasedIntegration()),
        ("DP-Elm", DualPassFormulation(),   ElementBasedIntegration()),
    ]
    cfg_marks  = ["triangle*", "diamond*", "*", "square*", "pentagon*", "otimes"]
    cfg_styles = ["densely dashed", "densely dashed", "solid", "solid", "dashdotted", "dashdotted"]
    # SP=vermilion, TM=blue, DP=green; 2:1 full color, 3:2 at 60% intensity
    ratio_intensities = ["100", "60"]

    # Two mesh ratios: 2:1 (full color) and 3:2 (60% intensity)
    ratios = [
        ("2:1", 3, 6),
        ("3:2", 2, 3),
    ]

    # ── Collect errors ────────────────────────────────────────────────────────
    # l2_data[ri][k] and en_data[ri][k] = Vector of errors for ratio ri, config k
    l2_data = [[Float64[] for _ in configs] for _ in ratios]
    en_data = [[Float64[] for _ in configs] for _ in ratios]

    for (ri, (ratio_label, p2base, p1base)) in enumerate(ratios)
        for (k, (label, form, strat)) in enumerate(configs)
            for e in exp_range
                l2, _, en, _ = solve_cylinder_p1(e;
                    strategy=strat, formulation=form, epss=epss,
                    n_ang_p2_base=p2base, n_ang_p1_base=p1base)
                push!(l2_data[ri][k], l2)
                push!(en_data[ri][k], en)
                @printf("  %s %-8s  exp=%d  L2=%.4e  E=%.4e\n",
                        ratio_label, label, e, l2, en)
            end
        end
    end

    # ── Single-mesh reference (no interface; one mesh ratio not applicable) ───
    l2_single = Float64[]; en_single = Float64[]
    for e in exp_range
        l2, _, en, _ = solve_cylinder_p1_single(e)
        push!(l2_single, l2); push!(en_single, en)
        @printf("  %-11s  exp=%d  L2=%.4e  E=%.4e\n", "Single", e, l2, en)
    end

    # ── Axis builder ──────────────────────────────────────────────────────────
    function make_axis_fc(l2_or_en_data, sm_vec, ylabel_str, title_str)
        elements = []
        legend_labels = String[]

        for (ri, (ratio_label, _, _)) in enumerate(ratios)
            for (k, (cfg_label, _, _)) in enumerate(configs)
                e_vec = l2_or_en_data[ri][k]
                # full color for 2:1 ratio, 60% intensity for 3:2
                color = ri == 1 ? CFG_COLORS[k] : "$(CFG_COLORS[k])!60!white"
                push!(elements, Plot(
                    PGFPlotsX.Options(
                        color         => nothing,
                        "mark"        => cfg_marks[k],
                        cfg_styles[k] => nothing,
                        "mark size"   => "2pt",
                        "line width"  => "1pt"),
                    Coordinates(h_vals, e_vec)))
                push!(legend_labels, "$cfg_label ($ratio_label)")
            end
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

        # O(h^1) slope anchored at last two TM-Elm (2:1) points
        tm21 = l2_or_en_data[1][4]; n = length(tm21)
        h_ref = h_vals[n-1:n]
        e_ref = tm21[n-1] .* (h_ref ./ h_ref[1]) .^ 1
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
        "Formulation comparison (\$L^2\$)")
    ax_en = make_axis_fc(en_data, en_single,
        L"\|e\|_E / \|u_{\rm ex}\|_E",
        "Formulation comparison (energy norm)")

    save_figure(ax_l2, "convergence_formulation_l2")
    save_figure(ax_en, "convergence_formulation_energy")
end

function run_beam_convergence()
    println("\n── Bending beam p=1 curved arc: L2 displacement, 6 formulations × 2 ratios ─")

    exp_range = 0:3
    epss      = 1e5   # safe for p=1 (avoids ε-resonance bands)
    h_vals    = [1.0 / 2^e for e in exp_range]   # proxy: coarser-patch x element size

    configs = [
        ("SP-Seg", SinglePassFormulation(), SegmentBasedIntegration()),
        ("SP-Elm", SinglePassFormulation(), ElementBasedIntegration()),
        ("TM-Seg", TwinMortarFormulation(), SegmentBasedIntegration()),
        ("TM-Elm", TwinMortarFormulation(), ElementBasedIntegration()),
        ("DP-Seg", DualPassFormulation(),   SegmentBasedIntegration()),
        ("DP-Elm", DualPassFormulation(),   ElementBasedIntegration()),
    ]
    cfg_marks  = ["triangle*", "diamond*", "*", "square*", "pentagon*", "otimes"]
    cfg_styles = ["densely dashed", "densely dashed", "solid", "solid", "dashdotted", "dashdotted"]

    ratios = [
        #("2:1", 2, 1),   # n_x_lower_base=2, n_x_upper_base=1
        #("3:2", 3, 2),    # n_x_lower_base=3, n_x_upper_base=2
        ("4:3", 3, 2),    # n_x_lower_base=3, n_x_upper_base=2
    ]

    disp_data = [[Float64[] for _ in configs] for _ in ratios]

    for (ri, (ratio_label, n_lo, n_up)) in enumerate(ratios)
        for (k, (label, form, strat)) in enumerate(configs)
            for e in exp_range
                _, err_abs = solve_beam_p1(e;
                    epss=epss, strategy=strat, formulation=form,
                    n_x_lower_base=n_lo, n_x_upper_base=n_up, curved=true)
                push!(disp_data[ri][k], err_abs)
                @printf("  %s %-8s exp=%d  L2=%.4e\n", ratio_label, label, e, err_abs)
            end
        end
    end

    elements = []
    legend_labels = String[]

    for (ri, (ratio_label, _, _)) in enumerate(ratios)
        for (k, (cfg_label, _, _)) in enumerate(configs)
            push!(elements, Plot(
                PGFPlotsX.Options(
                    CFG_COLORS[k] => nothing,
                    "mark"        => cfg_marks[k],
                    cfg_styles[k] => nothing,
                    "mark size"   => "2pt",
                    "line width"  => "1pt"),
                Coordinates(h_vals, disp_data[ri][k])))
            push!(legend_labels, "$cfg_label ($ratio_label)")
        end
    end

    # O(h^2) reference slope (optimal, p+1=2): anchored at last two TM-Elm (2:1) points
    tm21 = disp_data[1][4]; n = length(tm21)
    h_ref = h_vals[n-1:n]
    e_ref = tm21[n-1] .* (h_ref ./ h_ref[1]) .^ 2
    push!(elements, Plot(
        PGFPlotsX.Options("gray" => nothing, "no marks" => nothing,
            "thin" => nothing, "densely dotted" => nothing,
            "forget plot" => nothing),
        Coordinates(h_ref, e_ref)))

    # O(h^1) reference slope (variational crime, p=1): anchored at last two SP-Elm (2:1) points
    spelm21 = disp_data[1][2]; n2 = length(spelm21)
    h_ref2 = h_vals[n2-1:n2]
    e_ref2 = spelm21[n2-1] .* (h_ref2 ./ h_ref2[1]) .^ 1
    push!(elements, Plot(
        PGFPlotsX.Options("gray" => nothing, "no marks" => nothing,
            "thin" => nothing, "dashed" => nothing,
            "forget plot" => nothing),
        Coordinates(h_ref2, e_ref2)))

    ax = @pgf Axis(
        {
            xmode            = "log",
            ymode            = "log",
            xlabel           = L"h",
            ylabel           = L"\|e_u\|_{L^2} / \|u_{\rm ex}\|_{L^2}",
            title            = "Bending beam: curved arc, \$p=1\$",
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
    println("\n── System conditioning: κ(A) vs h, 2:1 and 3:2 ratios ─────────────────")

    exp_range = 0:3          # dense SVD feasible up to exp=3 (neq ~ 2600)
    epss      = 1e9

    ratios = [
        ("2:1", 3, 6, [0.5/2^e for e in exp_range]),   # n_ang_p2_base=3, n_ang_p1_base=6
        ("3:2", 2, 3, [0.5/2^e for e in exp_range]),   # n_ang_p2_base=2, n_ang_p1_base=3
    ]

    configs = [
        ("SP-Seg", SinglePassFormulation(), SegmentBasedIntegration()),
        ("SP-Elm", SinglePassFormulation(), ElementBasedIntegration()),
        ("TM-Seg", TwinMortarFormulation(), SegmentBasedIntegration()),
        ("TM-Elm", TwinMortarFormulation(), ElementBasedIntegration()),
        ("DP-Seg", DualPassFormulation(),   SegmentBasedIntegration()),
        ("DP-Elm", DualPassFormulation(),   ElementBasedIntegration()),
    ]
    cfg_marks  = ["triangle*", "diamond*", "*", "square*", "pentagon*", "otimes"]
    cfg_styles = ["densely dashed", "densely dashed", "solid", "solid", "dashdotted", "dashdotted"]
    elements = []
    legend_labels = String[]

    for (ri, (ratio_label, p2base, p1base, h_vals)) in enumerate(ratios)
        for (k, (cfg_label, form, strat)) in enumerate(configs)
            kv = Float64[]
            for e in exp_range
                κ, _ = cylinder_p1_kappa(e; formulation=form, strategy=strat, epss=epss,
                                          n_ang_p2_base=p2base, n_ang_p1_base=p1base)
                push!(kv, κ)
                @printf("  %s %-8s  exp=%d  κ=%.3e\n", ratio_label, cfg_label, e, κ)
            end
            # full color for 2:1 ratio, 60% intensity for 3:2
            color = ri == 1 ? CFG_COLORS[k] : "$(CFG_COLORS[k])!60!white"
            push!(elements, Plot(
                PGFPlotsX.Options(
                    color          => nothing,
                    "mark"         => cfg_marks[k],
                    cfg_styles[k]  => nothing,
                    "mark size"    => "2pt",
                    "line width"   => "1pt"),
                Coordinates(h_vals, kv)))
            push!(legend_labels, "$cfg_label ($ratio_label)")
        end
    end

    # Machine-precision reference κ = 10^16
    h_all = ratios[1][4]
    push!(elements, Plot(
        PGFPlotsX.Options(
            "gray"           => nothing,
            "no marks"       => nothing,
            "thin"           => nothing,
            "densely dotted" => nothing,
            "forget plot"    => nothing),
        Coordinates([h_all[1], h_all[end]], [1e16, 1e16])))

    ax = @pgf Axis(
        {
            xmode            = "log",
            ymode            = "log",
            xlabel           = L"h",
            ylabel           = L"\kappa(\mathbf{A})",
            title            = "System condition number",
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
        ("TM-Seg", TwinMortarFormulation(), SegmentBasedIntegration()),
        ("TM-Elm", TwinMortarFormulation(), ElementBasedIntegration()),
        ("DP-Seg", DualPassFormulation(),   SegmentBasedIntegration()),
        ("DP-Elm", DualPassFormulation(),   ElementBasedIntegration()),
    ]
    cfg_marks  = ["*", "square*", "pentagon*", "otimes"]
    cfg_styles = ["solid", "solid", "dashdotted", "dashdotted"]
    cfg_colors = ["black", "black", "gray", "gray"]

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
    println("\n── L2-displacement convergence: p=2, 6 formulations, 2:1 and 3:2 ratios ──")

    p_ord     = 2
    exp_range = 0:5
    epss      = 1e6
    h_vals    = [0.5 / 2^e for e in exp_range]

    configs = [
        ("SP-Seg", SinglePassFormulation(), SegmentBasedIntegration()),
        ("SP-Elm", SinglePassFormulation(), ElementBasedIntegration()),
        ("TM-Seg", TwinMortarFormulation(), SegmentBasedIntegration()),
        ("TM-Elm", TwinMortarFormulation(), ElementBasedIntegration()),
        ("DP-Seg", DualPassFormulation(),   SegmentBasedIntegration()),
        ("DP-Elm", DualPassFormulation(),   ElementBasedIntegration()),
    ]
    cfg_marks  = ["triangle*", "diamond*", "*", "square*", "pentagon*", "otimes"]
    cfg_styles = ["densely dashed", "densely dashed", "solid", "solid", "dashdotted", "dashdotted"]

    ratios = [
        ("2:1", 3, 6),
        ("3:2", 2, 3),
    ]

    # ── Collect errors ────────────────────────────────────────────────────────
    disp_data = [[Float64[] for _ in configs] for _ in ratios]

    for (ri, (ratio_label, p2base, p1base)) in enumerate(ratios)
        for (k, (label, form, strat)) in enumerate(configs)
            for e in exp_range
                _, _, d_rel, _ = solve_cylinder(p_ord, e;
                    epss=epss, strategy=strat, formulation=form,
                    n_ang_p2_base=p2base, n_ang_p1_base=p1base)
                push!(disp_data[ri][k], d_rel)
                @printf("  %s %-8s exp=%d  L2_disp=%.4e\n", ratio_label, label, e, d_rel)
            end
        end
    end

    # ── Build axis ────────────────────────────────────────────────────────────
    elements = []
    legend_labels = String[]

    for (ri, (ratio_label, _, _)) in enumerate(ratios)
        for (k, (cfg_label, _, _)) in enumerate(configs)
            # full color for 2:1 ratio, 60% intensity for 3:2
            color = ri == 1 ? CFG_COLORS[k] : "$(CFG_COLORS[k])!60!white"
            push!(elements, Plot(
                PGFPlotsX.Options(
                    color         => nothing,
                    "mark"        => cfg_marks[k],
                    cfg_styles[k] => nothing,
                    "mark size"   => "2pt",
                    "line width"  => "1pt"),
                Coordinates(h_vals, disp_data[ri][k])))
            push!(legend_labels, "$cfg_label ($ratio_label)")
        end
    end

    # O(h^{p+1}) = O(h^3) reference slope anchored at last two TM-Elm (2:1) points
    tm21 = disp_data[1][4]; n = length(tm21)
    h_ref = h_vals[n-1:n]
    e_ref = tm21[n-1] .* (h_ref ./ h_ref[1]) .^ (p_ord + 1)
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

const ALL_CASES = ["plate_nc", "plate_c", "cylinders", "sphere", "formulation", "conditioning", "nquad_disp", "disp_p2", "beam", "beam_normal_strategy"]
const CASE_FNS  = Dict(
    "plate_nc"             => run_plate_nc,
    "plate_c"              => run_plate_c,
    "cylinders"            => run_cylinders,
    "sphere"               => run_sphere,
    "formulation"          => run_formulation_comparison_plots,
    "conditioning"         => run_conditioning_plot,
    "nquad_disp"           => run_nquad_sweep_disp,
    "disp_p2"              => run_disp_convergence_p2,
    "beam"                 => run_beam_convergence,
    "beam_normal_strategy" => run_beam_normal_strategy,
)

requested = isempty(ARGS) ? ALL_CASES : ARGS

for case in requested
    if haskey(CASE_FNS, case)
        CASE_FNS[case]()
    else
        @warn "Unknown case '$case'. Available: " * join(ALL_CASES, ", ")
    end
end
