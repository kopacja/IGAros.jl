# render_sphere.jl
#
# Export the 2-patch p=1 pressurized sphere for Blender rendering.
#
# Usage:
#   cd IGAros/examples
#   julia --project=.. render_sphere.jl
#
# Then in Blender:
#   blender --python sphere_render_import.py

include("pressurized_sphere.jl")
include("export_blender.jl")

# ── Parameters ────────────────────────────────────────────────────────────────

p_ord = 1
exp_level = 0
r_i = 1.0; r_c = 1.2; r_o = 1.4

nsd = 3; npd = 3; npc = 2

n_ang_o = 2^(exp_level + 2)                   # 4
n_ang_i = round(Int, 2.0 * n_ang_o)           # 8 (non-conforming 2:1)
n_rad   = 2^exp_level                          # 1

# ── Build geometry ────────────────────────────────────────────────────────────

B_ref, P_ref = sphere_geometry_direct_p1(n_ang_i, n_ang_o, n_rad;
                                          r_i=r_i, r_c=r_c, r_o=r_o)

p_mat     = fill(p_ord, npc, npd)
n_mat_ref = [n_ang_i+1  n_ang_i+1  n_rad+1;
             n_ang_o+1  n_ang_o+1  n_rad+1]

kv_i   = open_uniform_kv(n_ang_i, 1)
kv_o   = open_uniform_kv(n_ang_o, 1)
kv_rad = open_uniform_kv(n_rad,   1)
KV_ref = Vector{Vector{Vector{Float64}}}([
    [kv_i, kv_i, kv_rad],
    [kv_o, kv_o, kv_rad]
])

# ── Radial stress field (exact Lamé solution) ────────────────────────────────
p_i_load = 0.01
function radial_stress(x, y, z)
    r = sqrt(x^2 + y^2 + z^2)
    r < 1e-14 && return 0.0
    sig = lame_stress_sphere(x, y, z; p_i=p_i_load, r_i=r_i, r_o=r_o)
    rhat = [x, y, z] / r
    return dot(rhat, sig * rhat)   # σ_rr
end

# ── Export for Blender ────────────────────────────────────────────────────────
# All 6 boundary faces per patch
export_faces = Dict(
    pc => [1, 2, 3, 4, 5, 6] for pc in 1:npc
)

println("Exporting Blender files for p=$p_ord, exp=$exp_level sphere...")
export_blender("sphere_render", npc, nsd, npd, p_mat, n_mat_ref, KV_ref, P_ref, B_ref;
               faces=export_faces, n_vis=4, scalar_fn=radial_stress)

println("\nDone! Next steps:")
println("  1. Open Blender (≥ 4.0)")
println("  2. Run:  blender --python sphere_render_import.py")
println("  3. Adjust camera angle in viewport")
println("  4. F12 to render")
