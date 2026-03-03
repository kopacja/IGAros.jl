module IGAros

using LinearAlgebra
using SparseArrays

include("BSplines.jl")
include("Quadrature.jl")
include("KnotVectors.jl")
include("Connectivity.jl")
include("Materials.jl")
include("StrainDisplacement.jl")
include("Geometry.jl")
include("Assembly.jl")
include("BoundaryConditions.jl")
include("Solver.jl")
include("MortarGeometry.jl")
include("MortarAssembly.jl")
include("MortarSolver.jl")

export
    # BSplines
    find_span, basis_funs, bspline_basis_and_deriv,
    # Quadrature
    gauss_rule, gauss_product,
    # KnotVectors
    generate_knot_vector, generate_knot_vectors, knot_insertion,
    multiple_knot_insertion, krefinement,
    # Connectivity
    nurbs_coords, build_inc, build_ien, build_id, build_lm, build_ind,
    patch_metrics,
    # Materials
    MaterialModel, LinearElastic, elastic_constants,
    # StrainDisplacement
    strain_displacement_matrix,
    # Geometry
    shape_function,
    # Assembly
    element_stiffness, build_stiffness_matrix, build_updated_geometry,
    # BoundaryConditions
    dirichlet_bc_control_points, get_segment_patch, enforce_dirichlet, segment_load,
    # Solver
    linear_solve,
    # MortarGeometry
    eval_boundary_point, closest_point_1d,
    # MortarAssembly
    InterfacePair, build_interface_cps, build_mortar_coupling,
    # MortarSolver
    solve_mortar

end
