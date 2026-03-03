using Test

@testset "IGAros" begin
    include("test_bsplines.jl")
    include("test_quadrature.jl")
    include("test_knot_vectors.jl")
    include("test_connectivity.jl")
    include("test_materials.jl")
    include("test_geometry.jl")
    include("test_assembly.jl")
    include("test_mortar.jl")
end
