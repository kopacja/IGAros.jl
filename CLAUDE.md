# IGAros.jl — Claude Code Guidelines

## Project Overview

**IsoGeometric Analysis** research package in Julia — NURBS-based structural mechanics with multi-patch non-conforming mesh coupling via the **Twin Mortar method**. This repository is the numerical engine behind the CMAME manuscript in the parent directory. Ported from a legacy MATLAB research codebase; SOLID-designed with a full unit-test suite.

- **Julia version**: 1.12
- **Dependencies**: Julia standard library only (`LinearAlgebra`, `SparseArrays`)
- **Scope**: 2D linear elasticity with multi-patch NURBS meshes (plus a 3D mortar integration module)

**Cross-references:**
- Parent manuscript methodology, SLURM workflow, and reproducibility conventions: [`../CLAUDE.md`](../CLAUDE.md)
- Paper-side design decisions (Z sign, C/2 factor, formulation): `../Docs/Paper/Architecture/`
- Code-side design decisions (integration strategies, module split): `../Docs/IGAros/Architecture/`

## Architecture

```
IGAros/
├── Project.toml / Manifest.toml    # package manifest — do not hand-edit Manifest
├── src/                            # package source
│   ├── IGAros.jl                   # module root
│   ├── BSplines.jl                 # find_span, basis_funs, derivatives
│   ├── KnotVectors.jl              # generate_knot_vector, k-refinement
│   ├── Quadrature.jl               # gauss_rule, tensor-product rules
│   ├── Connectivity.jl             # INC/IEN/ID/LM arrays
│   ├── Materials.jl                # LinearElastic
│   ├── StrainDisplacement.jl       # B-matrix
│   ├── Geometry.jl                 # NURBS shape fns + Jacobians
│   ├── Assembly.jl                 # element_stiffness, build_stiffness_matrix
│   ├── BoundaryConditions.jl       # Dirichlet, Neumann
│   ├── Solver.jl                   # linear_solve
│   ├── MortarGeometry.jl           # closest_point_1d/2d
│   ├── MortarAssembly.jl           # interface DOFs, C/D/M/Z matrices
│   ├── MortarIntegration.jl        # element-based & segment-based integration
│   └── MortarSolver.jl             # KKT solve for Twin Mortar tying
├── examples/                       # benchmark drivers, produce data in ../results/
├── benchmark/                      # performance benchmarks + SLURM scripts
├── test/                           # unit tests (runtests.jl)
├── results/                        # package-local test artifacts (not paper results)
└── legacy/                         # reference MATLAB code for port verification
```

**Key architectural principles:**

- **Separation of concerns by module** — each `src/*.jl` covers one responsibility (basis, quadrature, assembly, ...). Functions at the module boundaries have documented signatures; internal helpers are `_prefixed`.
- **Explicit data passing** — no module-level mutable state. Connectivity arrays, quadrature rules, and assembled matrices are passed as arguments.
- **Shared kernels** — element-based and segment-based mortar integration both use `_accumulate_mortar!` to populate C/D/M/Z; differ only in the quadrature point set.
- **KKT system form** — `[K C; Cᵀ Z]` with Z stored positive definite. See `Key Conventions` below.

## Code Conventions

### Julia style

- **Indentation**: 4 spaces
- **Function names**: `snake_case`
- **Type names**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Module-private helpers**: leading underscore (`_accumulate_mortar!`)
- **Mutating functions**: trailing `!` per Julia convention
- **Type annotations**: on function signatures at module boundaries; not required for internal closures
- **Docstrings**: triple-quoted, at least on exported functions — describe inputs, outputs, and any non-obvious invariant

### File organization

- One primary responsibility per `src/*.jl` file
- Test file per source file: `test/<SourceName>Tests.jl`
- `test/runtests.jl` orchestrates all suites

### Key Conventions (load-bearing — do not break silently)

These are numerical invariants that other code depends on. Changing them requires a journal entry in `../Docs/Paper/Journal/Fixes.md` and re-running all affected benchmarks.

- **B-matrix layout**: 4 columns `[x, y, z=0, w]`. **Weight is the LAST column** (not the first).
- **INC anchor**: `INC[A][d] = nc_d ∈ [1, n_d]`. Element span is `[kv[nc_d], kv[nc_d+1]]`.
- **Stiffness symmetry**: only approximate equality — test with `K ≈ K' atol=1e-10`, not exact equality.
- **Z sign convention**: stored **positive definite**, `+ε[D̄,M̄;M̄ᵀ,D̄]`. KKT block is `[K C; Cᵀ Z]`. See parent [`../CLAUDE.md`](../CLAUDE.md) "Key Project Decisions".
- **C/2 factor**: TM and DP kernels assemble C with factor 1/2 so that `λ^(s) = ±σ_app` (full physical traction), not `±σ/2`.
- **CPP boundary check**: `closest_point_2d` clamps to `[0,1]²`. After CPP convergence, check parametric boundary + physical gap > tol, skip out-of-domain GPs. Without this: systematic integration error on overhanging geometries.
- **Rim/mortar quadrature consistency**: for element-based mortar, rim subtraction must use the same NQUAD and same skip logic as mortar assembly. Mismatch dominates displacement error.

## Do Not Edit Directly

- **`Manifest.toml`** — Julia dependency lockfile. Update via `Pkg.update()` / `Pkg.add(...)` only. Do not `rsync --delete` to the cluster (desyncs with remote lockfile and breaks precompile).
- **`legacy/`** — reference MATLAB code kept for port verification. Read-only; if a bug is found, fix in Julia, **do not patch MATLAB**.
- **`results/`** inside IGAros (the package-local one) — only for test artifacts and benchmark outputs. Paper results live in `../results/` at the parent level.

## Build & Test

```bash
# From IGAros/ directory

# Install / resolve dependencies
julia --project=. -e 'import Pkg; Pkg.instantiate()'

# Run full test suite
julia --project=. test/runtests.jl

# Run a specific example (produces results in ../results/<benchmark>/)
julia --project=. examples/<script>.jl

# Precompile (useful after Pkg changes)
julia --project=. -e 'import Pkg; Pkg.precompile()'
```

**Current test state (2026-04-10):** 1227 tests pass.

### Test categories

- **Unit** (module-level): basis fn values vs. hand-computed, quadrature exactness, connectivity sanity
- **Integration** (assembly-level): stiffness symmetry, patch-test exactness, KKT solve consistency
- **Benchmark drivers** (`examples/`): end-to-end convergence against analytical solutions

### Writing new tests

- New source file → new `test/<SourceName>Tests.jl` + register in `runtests.jl`
- Use `@testset` blocks with descriptive names
- Test for **behavior at documented boundaries** (boundary of parametric domain, overhanging geometry, degree edge cases p=1 vs p≥2)
- Regression tests for **bugs that were caught once** — name the test after the bug (`test_cpp_out_of_domain_skip`, `test_z_positive_definite_after_assembly`)

### Cluster runs

Heavy experiments (NQUAD sweeps, large ε × NQUAD grids, convergence studies) must go through SLURM. See [`../CLAUDE.md`](../CLAUDE.md) "Kraken HPC Cluster" for partitions, submission patterns, and the skip-study guard convention for reusable scripts.

## Git Workflow

Full software-engineering discipline — this is where things can silently break numbers in the paper.

- **Main branch**: `main`
- **Feature branches** for all non-trivial work: `feature/<short>`, `fix/<short>`, `refactor/<short>`, `benchmark/<short>`, `perf/<short>`
- **PR self-review required** for any change touching `src/` or changing numerical output
- **Direct-to-main only** for docstrings, README, test comments, benchmark script tweaks
- **Before merging**: all tests green, no new warnings, affected benchmarks re-run (if numerics-affecting)
- **Commit messages**: imperative, category prefix — `feat:`, `fix:`, `refactor:`, `test:`, `bench:`, `perf:`, `docs:`

### When paper claims are affected

If a code change alters a number that appears in the paper:

1. Land the IGAros PR first.
2. Re-run the affected benchmark; update `meta.toml` with the new commit SHA.
3. Regenerate the paper figure via the documented `plot.jl` / `render.py`.
4. Open a paper-side PR referencing the IGAros commit SHA in the description.
5. Journal the fix in `../Docs/Paper/Journal/Fixes.md` (not just `Docs/IGAros/Journal/`) because the paper is affected.

### Submission tags

At each paper submission, tag at the commit that produced the numbers:

```bash
git tag paper-cmame-submit-v1 -m "Submitted to CMAME, version 1"
git push origin paper-cmame-submit-v1
```

## Task Workflow

Same lifecycle as the paper repo (Pick → Analyze → Implement → Review → Ship) — see parent [`../CLAUDE.md`](../CLAUDE.md) "Task Workflow". IGAros specifics:

- **Pick** from `../Docs/IGAros/Feature Backlog.md` or `../Docs/IGAros/Tech Debt.md`
- **Implement** includes writing tests — TDD for anything non-trivial
- **Review** includes `julia test/runtests.jl` green + affected benchmark re-run if numerics-sensitive
- **Ship** — journal in `../Docs/IGAros/Journal/Features.md` or `Refactoring.md` (`Fixes.md` on the paper side if paper is affected)

## Engineering Principles

Solo-dev research code with a shipped-paper consumer — rigor where it buys you reproducibility.

### Design principles

- **SOLID, KISS, DRY, YAGNI** — no over-engineering, no premature abstraction, but also no half-done refactors
- **Pure functions by default** — explicit state in, explicit state out; avoid closures over mutable state
- **Fail fast at boundaries** — check arguments at the entry to each module function; inside helpers trust your invariants
- **One reason to change per module** — Single Responsibility at file granularity
- **Julia-specific**:
  - Prefer immutable `struct` over mutable unless mutation is the point
  - Use `@views` / `@inbounds` only where profiled, not by reflex
  - Preallocate in tight loops; avoid repeated allocation for small arrays
  - Multiple dispatch over type flags — e.g. `integrate(::ElementBasedIntegration, ...)` vs `integrate(::SegmentBasedIntegration, ...)`

### Testable code

- Prefer pure functions (basis evaluation, quadrature rules, connectivity builders)
- Dependencies passed as arguments, not read from module globals
- Integration strategies as types (`ElementBasedIntegration`, `SegmentBasedIntegration`) → easy to mock / swap
- **Coverage is not the goal** — test the non-trivial logic, the documented boundaries, and every bug we've ever caught

### Commit discipline

- Logical units: one feature / fix / refactor per commit
- Never commit failing tests to `main`
- Before commit: tests pass, no new warnings, no `@debug`/`println` dead code

### Definition of Done

See parent [`../CLAUDE.md`](../CLAUDE.md) "Definition of Done — IGAros code change". Summary:

- Build passes, all tests green, no new warnings
- Affected benchmarks re-run if numerics-affecting
- Journal entry written
- Kanban card moved to Done
- Committed (direct or via merged PR)

### Feature Capture Rule

Same as parent — if during work a new bug / tech-debt item / feature idea surfaces, write it as a kanban card and return to the current task. Do not tangent off.

## Development Documentation

Obsidian vault at `../Docs/`. Code-relevant subfolders:

- `../Docs/IGAros/Feature Backlog.md` — planned code features (Kanban)
- `../Docs/IGAros/Tech Debt.md` — refactors / cleanup (Kanban)
- `../Docs/IGAros/Architecture/` — design decisions:
  - `Module Overview.md` — what each `src/*.jl` is for
  - `Integration Strategies.md` — element-based vs segment-based mortar
  - `Key Conventions.md` — B-matrix layout, INC anchor, stiffness symmetry, Z sign, C/2
- `../Docs/IGAros/Journal/`
  - `Features.md` — feature / benchmark implementations
  - `Refactoring.md` — architectural changes, test coverage, cleanup

See parent [`../CLAUDE.md`](../CLAUDE.md) "Development Journal" for entry format and routing rules.

## Package Documentation

User-facing docs remain in `README.md` (quick start, module table, examples pointer). Internal design documentation lives in `../Docs/IGAros/Architecture/` so the same design-doc conventions apply across paper and code.
