# Materials.jl
# Abstract material interface and linear elastic constitutive model.
# Ported from MATLAB: getMatrixOfElasticConstants.m

"""
    MaterialModel

Abstract type for constitutive models. Any subtype must implement:
- `elastic_constants(m::MaterialModel, nsd::Int) -> Matrix{Float64}`
"""
abstract type MaterialModel end

"""
    LinearElastic(E, nu, stress_type)

Isotropic linear elastic material.

- `E`: Young's modulus
- `nu`: Poisson's ratio
- `stress_type`: one of `:plane_stress`, `:plane_strain`, `:axisymmetric`, `:three_d`
"""
struct LinearElastic <: MaterialModel
    E::Float64
    nu::Float64
    stress_type::Symbol
end

"""
    elastic_constants(m::LinearElastic, nsd) -> Matrix{Float64}

Return the Voigt-notation elasticity matrix D for the given material and spatial dimension.

For nsd=2:
- `:plane_stress`  → 3×3
- `:plane_strain`  → 3×3
- `:axisymmetric`  → 4×4

For nsd=3 → 6×6.

Ported from getMatrixOfElasticConstants.m.
"""
function elastic_constants(m::LinearElastic, nsd::Int)::Matrix{Float64}
    E  = m.E
    nu = m.nu
    λ  = nu * E / ((1 + nu) * (1 - 2nu))
    μ  = E / (2 * (1 + nu))

    if nsd == 3
        return [λ+2μ  λ     λ     0  0  0
                λ     λ+2μ  λ     0  0  0
                λ     λ     λ+2μ  0  0  0
                0     0     0     μ  0  0
                0     0     0     0  μ  0
                0     0     0     0  0  μ]

    elseif nsd == 2
        if m.stress_type == :plane_stress
            return E / (1 - nu^2) * [1    nu        0
                                     nu   1         0
                                     0    0   (1-nu)/2]

        elseif m.stress_type == :plane_strain
            return E / ((1+nu)*(1-2nu)) *
                   [(1-nu)  nu       0
                    nu      (1-nu)   0
                    0       0        (1-2nu)/2]

        elseif m.stress_type == :axisymmetric
            # Full 3D → pick [rr, zz, tt, rz] (D33 + D3 block from MATLAB)
            D3d = [λ+2μ  λ     λ     0  0  0
                   λ     λ+2μ  λ     0  0  0
                   λ     λ     λ+2μ  0  0  0
                   0     0     0     μ  0  0
                   0     0     0     0  μ  0
                   0     0     0     0  0  μ]
            D33 = [D3d[1,1]  D3d[1,2]  D3d[1,4]
                   D3d[2,1]  D3d[2,2]  D3d[2,4]
                   D3d[4,1]  D3d[4,2]  D3d[4,4]]
            D3  = [D3d[1,3]; D3d[2,3]; D3d[4,3]]
            D44 = D3d[3,3]
            return [D33  D3
                    D3'  fill(D44,1,1)]
        else
            error("Unknown stress type: $(m.stress_type)")
        end
    else
        error("elastic_constants: unsupported nsd = $nsd")
    end
end
