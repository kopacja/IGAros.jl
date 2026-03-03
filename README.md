# IGAros.jl

**IsoGeometric Analysis** research package in Julia — NURBS-based structural mechanics with multi-patch non-conforming mesh coupling via the Twin Mortar method.

## Overview

IGAros.jl implements the core building blocks of Isogeometric Analysis (IGA) for 2D linear elasticity:

- B-spline / NURBS basis functions and knot-vector operations
- Multi-patch connectivity (INC / IEN / ID / LM arrays)
- k-refinement (h-refinement + degree elevation)
- Gauss quadrature (tensor product rules)
- Stiffness matrix assembly and Neumann traction loading
- Dirichlet boundary conditions
- **Twin Mortar method** for non-conforming multi-patch coupling over curved interfaces

The package is ported from a MATLAB research codebase and uses only Julia standard-library dependencies (`LinearAlgebra`, `SparseArrays`).

## Installation

```julia
import Pkg
Pkg.develop(path="path/to/IGAros")  # local development
```

Or once registered:

```julia
import Pkg
Pkg.add("IGAros")
```

## Quick Start

```julia
using IGAros

# Generate a clamped knot vector of degree p with n control points
KV = generate_knot_vector(p, n)

# B-spline basis functions and first derivatives at parameter u
i  = find_span(n, p, u, KV)
N  = basis_funs(i, u, p, KV)
dN = bspline_basis_and_deriv(p, i, u, KV)

# Gauss quadrature rule (nq points in 1D, tensor-product in npd dimensions)
rule = gauss_product(nq, npd)

# Linear elastic material (plane stress)
mat = LinearElastic(E, nu, :plane_stress)
D   = elastic_constants(mat, nsd)
```

See [`examples/`](examples/) for complete multi-patch solve workflows.

## Examples

### Plate with a Circular Hole

[`examples/plate_with_hole.jl`](examples/plate_with_hole.jl) — Quarter-domain, two non-conforming NURBS patches joined along the 135° diagonal. Loaded by the exact Kirsch traction; convergence measured in the L2 stress norm against the Kirsch analytical solution.

```bash
julia examples/plate_with_hole.jl
```

### Concentric Cylinders

[`examples/concentric_cylinders.jl`](examples/concentric_cylinders.jl) — Two-patch quarter annulus with a curved (circular arc) non-conforming interface. Loaded by external pressure; compared against the Lamé exact solution. Includes an ε-sensitivity sweep for the Twin Mortar stabilization parameter.

```bash
julia examples/concentric_cylinders.jl
```

## Module Structure

| File | Contents |
|------|----------|
| [`BSplines.jl`](src/BSplines.jl) | `find_span`, `basis_funs`, `bspline_basis_and_deriv` |
| [`KnotVectors.jl`](src/KnotVectors.jl) | `generate_knot_vector(s)`, `knot_insertion`, `krefinement` |
| [`Quadrature.jl`](src/Quadrature.jl) | `gauss_rule`, `gauss_product` |
| [`Connectivity.jl`](src/Connectivity.jl) | `build_inc`, `build_ien`, `build_id`, `build_lm`, `patch_metrics` |
| [`Geometry.jl`](src/Geometry.jl) | `shape_function` (NURBS shape fns + Jacobian) |
| [`Materials.jl`](src/Materials.jl) | `LinearElastic`, `elastic_constants` |
| [`StrainDisplacement.jl`](src/StrainDisplacement.jl) | `strain_displacement_matrix` |
| [`Assembly.jl`](src/Assembly.jl) | `element_stiffness`, `build_stiffness_matrix` |
| [`BoundaryConditions.jl`](src/BoundaryConditions.jl) | `dirichlet_bc_control_points`, `enforce_dirichlet`, `segment_load` |
| [`Solver.jl`](src/Solver.jl) | `linear_solve` |
| [`MortarGeometry.jl`](src/MortarGeometry.jl) | `eval_boundary_point`, `closest_point_1d` |
| [`MortarAssembly.jl`](src/MortarAssembly.jl) | `InterfacePair`, `build_interface_cps`, `build_mortar_coupling` |
| [`MortarSolver.jl`](src/MortarSolver.jl) | `solve_mortar` (KKT system for Twin Mortar tying) |

## Twin Mortar Method

The Twin Mortar method couples non-conforming NURBS patches by enforcing interface compatibility weakly through Lagrange multipliers. It assembles a KKT system:

```
[ K    C  ] [U]   [F]
[ C^T  -Z ] [λ] = [0]
```

where `C` is the mortar coupling matrix, `Z` is a stabilization matrix, and `λ` are the interface Lagrange multipliers. Both conforming and non-conforming meshes (including curved interfaces) are supported.

## Running Tests

```julia
using Pkg
Pkg.test("IGAros")
```

## License

Research code — see repository for license details.
