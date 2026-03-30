# render_sphere_3patch.jl
#
# Export the 3-patch deltoidal sphere for Blender rendering.
#
# NOTE: The deltoidal icositetrahedron tiles are degree 4 in each surface
# direction, so the minimum polynomial degree is p=4.
#
# Usage:
#   cd IGAros/examples
#   julia --project=.. render_sphere_3patch.jl
#
# Then in Blender:
#   blender --python sphere3_render_import.py

include("pressurized_sphere_3patch.jl")
include("export_blender.jl")

# ── Parameters ────────────────────────────────────────────────────────────────

p_ord = 4          # minimum for deltoidal tiles
exp_level = 0      # coarsest refinement
r_i = 1.0; r_o = 1.2
E = 1.0; nu = 0.3; p_i_load = 1.0

nsd = 3; npd = 3; npc = 3

# ── Build geometry ────────────────────────────────────────────────────────────

B0, P = sphere_geometry_3patch(p_ord; r_i=r_i, r_o=r_o)
p_geom = max(p_ord, 4)
p_mat = fill(p_geom, npc, npd)
n_ang = p_geom + 1
n_rad = p_geom + 1
n_mat = fill(0, npc, npd)
for pc in 1:npc
    n_mat[pc, :] = [n_ang, n_ang, n_rad]
end

KV = [[vcat(zeros(p_geom+1), ones(p_geom+1)) for _ in 1:3] for _ in 1:npc]

# h-refinement
n_base = 4
n_elem     = n_base * 2^exp_level
n_rad_elem = 2^exp_level

u_surf = Float64[i/n_elem for i in 1:n_elem-1]
u_rad  = Float64[i/n_rad_elem for i in 1:n_rad_elem-1]

kref_data = Vector{Float64}[]
for t in 1:npc
    push!(kref_data, vcat([Float64(t), 1.0], u_surf))
    push!(kref_data, vcat([Float64(t), 2.0], u_surf))
    push!(kref_data, vcat([Float64(t), 3.0], u_rad))
end

n_mat_ref, _, KV_ref, B_ref, P_ref = krefinement(
    nsd, npd, npc, n_mat, p_mat, KV, B0, P, kref_data
)

# ── Radial stress field (exact Lamé solution) ────────────────────────────────

function radial_stress(x, y, z)
    r = sqrt(x^2 + y^2 + z^2)
    r < 1e-14 && return 0.0
    sig = lame_stress_sphere(x, y, z; p_i=p_i_load, r_i=r_i, r_o=r_o)
    rhat = [x, y, z] / r
    return dot(rhat, sig * rhat)
end

# ── Export for Blender ────────────────────────────────────────────────────────

export_faces = Dict(
    pc => [1, 2, 3, 4, 5, 6] for pc in 1:npc
)

println("Exporting Blender files for 3-patch sphere, p=$p_ord, exp=$exp_level...")
export_blender("sphere3_render", npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref;
               faces=export_faces, n_vis=6, scalar_fn=radial_stress)

println("\nDone! Run:  blender --python sphere3_render_import.py")
