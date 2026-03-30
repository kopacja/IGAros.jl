# sphere_render_import.py — Auto-generated Blender import script
# Run: blender --python sphere_render_import.py
# Or:  File > Scripting > Open > Run Script

import bpy
import os
import math
from pathlib import Path

# ─── Clean default scene ──────────────────────────────────────────────
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Delete default collection objects
for obj in bpy.data.objects:
    bpy.data.objects.remove(obj, do_unlink=True)

script_dir = str(Path(bpy.data.filepath).parent) if bpy.data.filepath else os.getcwd()

# ─── Materials ────────────────────────────────────────────────────────

def make_surface_material(name="IGA_Surface", color=(0.15, 0.35, 0.65, 1.0)):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Metallic"].default_value = 0.1
    bsdf.inputs["Roughness"].default_value = 0.4
    bsdf.inputs["Specular IOR Level"].default_value = 0.5
    return mat

def make_vertex_color_material(name="IGA_Scalar"):
    """Material that reads vertex colors (from OBJ 'v x y z r g b')."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    tree = mat.node_tree
    bsdf = tree.nodes["Principled BSDF"]
    bsdf.inputs["Metallic"].default_value = 0.05
    bsdf.inputs["Roughness"].default_value = 0.35
    bsdf.inputs["Specular IOR Level"].default_value = 0.5
    # Color Attribute node → Base Color (layer name set at import time)
    ca = tree.nodes.new('ShaderNodeVertexColor')
    ca.location = (-300, 300)
    ca.layer_name = ""  # will be set after import
    tree.links.new(ca.outputs['Color'], bsdf.inputs['Base Color'])
    return mat

def make_wireframe_material(name="CP_Net", color=(0.9, 0.3, 0.1, 1.0)):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 0.8
    return mat

def make_cp_material(name="CP_Points", color=(0.85, 0.08, 0.08, 1.0)):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Metallic"].default_value = 0.6
    bsdf.inputs["Roughness"].default_value = 0.3
    return mat

mat_surface = make_surface_material()
mat_scalar  = make_vertex_color_material()
mat_cpnet   = make_wireframe_material()
mat_cp      = make_cp_material()

# ─── Import OBJ files ────────────────────────────────────────────────

def import_obj(filepath, collection_name):
    """Import OBJ and move to named collection."""
    if not os.path.isabs(filepath):
        filepath = os.path.join(script_dir, filepath)
    if not os.path.exists(filepath):
        print(f"  SKIP (not found): {filepath}")
        return None

    # Create collection if needed
    if collection_name not in bpy.data.collections:
        col = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(col)
    col = bpy.data.collections[collection_name]

    before = set(bpy.data.objects.keys())
    bpy.ops.wm.obj_import(filepath=filepath)
    after = set(bpy.data.objects.keys())
    new_objs = after - before

    for name in new_objs:
        obj = bpy.data.objects[name]
        # Move to our collection
        for c in obj.users_collection:
            c.objects.unlink(obj)
        col.objects.link(obj)

    return [bpy.data.objects[n] for n in new_objs]

# Import patch surfaces
patch_files = [f for f in ["sphere_render_patch_1.obj", "sphere_render_patch_2.obj", "sphere_render_cpnet_1.obj", "sphere_render_cpnet_2.obj", "sphere_render_cpts.obj"] if "_patch_" in f]
for pf in patch_files:
    objs = import_obj(pf, "Patches")
    if objs:
        for obj in objs:
            obj.data.materials.clear()
            # Use vertex-color material if the mesh has color attributes
            if obj.data.color_attributes:
                layer_name = obj.data.color_attributes[0].name
                mat_vc = make_vertex_color_material(name=f"IGA_Scalar_{obj.name}")
                # Set the layer name to match what the OBJ importer created
                for node in mat_vc.node_tree.nodes:
                    if node.type == 'VERTEX_COLOR':
                        node.layer_name = layer_name
                obj.data.materials.append(mat_vc)
            else:
                obj.data.materials.append(mat_surface)
            # Smooth shading
            for poly in obj.data.polygons:
                poly.use_smooth = True

# Import control nets
cpnet_files = [f for f in ["sphere_render_patch_1.obj", "sphere_render_patch_2.obj", "sphere_render_cpnet_1.obj", "sphere_render_cpnet_2.obj", "sphere_render_cpts.obj"] if "_cpnet_" in f]
for cf in cpnet_files:
    objs = import_obj(cf, "ControlNet")
    if objs:
        for obj in objs:
            # Convert edges to renderable tubes via Geometry Nodes
            # (Wireframe modifier doesn't work on OBJ line elements)
            gn_mod = obj.modifiers.new(name="EdgeTubes", type='NODES')
            tree = bpy.data.node_groups.new("CP_Net_Tubes", 'GeometryNodeTree')
            tree.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
            tree.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')

            inp = tree.nodes.new('NodeGroupInput')
            inp.location = (-400, 0)
            out = tree.nodes.new('NodeGroupOutput')
            out.location = (400, 0)

            # Mesh to Curve → Curve to Mesh with circle profile
            m2c = tree.nodes.new('GeometryNodeMeshToCurve')
            m2c.location = (-200, 0)

            circle = tree.nodes.new('GeometryNodeCurvePrimitiveCircle')
            circle.location = (-200, -200)
            circle.inputs['Resolution'].default_value = 8
            circle.inputs['Radius'].default_value = 0.006

            c2m = tree.nodes.new('GeometryNodeCurveToMesh')
            c2m.location = (0, 0)

            tree.links.new(inp.outputs[0], m2c.inputs['Mesh'])
            tree.links.new(m2c.outputs['Curve'], c2m.inputs['Curve'])
            tree.links.new(circle.outputs['Curve'], c2m.inputs['Profile Curve'])
            tree.links.new(c2m.outputs['Mesh'], out.inputs[0])

            gn_mod.node_group = tree
            obj.data.materials.clear()
            obj.data.materials.append(mat_cpnet)

# Import control points
cpts_file = [f for f in ["sphere_render_patch_1.obj", "sphere_render_patch_2.obj", "sphere_render_cpnet_1.obj", "sphere_render_cpnet_2.obj", "sphere_render_cpts.obj"] if "_cpts" in f]
for cf in cpts_file:
    objs = import_obj(cf, "ControlPoints")
    if objs:
        for obj in objs:
            # Convert vertices to spheres via geometry nodes
            gn_mod = obj.modifiers.new(name="PointSpheres", type='NODES')
            # Create a simple geometry node tree: Mesh to Points → Instance on Points
            tree = bpy.data.node_groups.new("CP_Spheres", 'GeometryNodeTree')
            tree.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
            tree.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')

            inp = tree.nodes.new('NodeGroupInput')
            inp.location = (-400, 0)
            out = tree.nodes.new('NodeGroupOutput')
            out.location = (400, 0)

            m2p = tree.nodes.new('GeometryNodeMeshToPoints')
            m2p.location = (-200, 0)
            m2p.inputs['Radius'].default_value = 0.015

            iop = tree.nodes.new('GeometryNodeInstanceOnPoints')
            iop.location = (0, 0)

            sphere = tree.nodes.new('GeometryNodeMeshIcoSphere')
            sphere.location = (-200, -200)
            sphere.inputs['Radius'].default_value = 0.015
            sphere.inputs['Subdivisions'].default_value = 3

            tree.links.new(inp.outputs[0], m2p.inputs['Mesh'])
            tree.links.new(m2p.outputs['Points'], iop.inputs['Points'])
            tree.links.new(sphere.outputs['Mesh'], iop.inputs['Instance'])
            tree.links.new(iop.outputs['Instances'], out.inputs[0])

            gn_mod.node_group = tree
            obj.data.materials.clear()
            obj.data.materials.append(mat_cp)

# ─── Shadow-catcher ground plane ─────────────────────────────────────
# White studio floor that catches shadows but is otherwise invisible.

bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, -0.01))
ground = bpy.context.active_object
ground.name = "ShadowCatcher"
ground.is_shadow_catcher = True
# White diffuse material (only visible via shadows)
mat_ground = bpy.data.materials.new(name="Ground")
mat_ground.use_nodes = True
bsdf_g = mat_ground.node_tree.nodes["Principled BSDF"]
bsdf_g.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
bsdf_g.inputs["Roughness"].default_value = 1.0
ground.data.materials.append(mat_ground)

# ─── Lighting (3-point studio setup) ─────────────────────────────────

# Key light — main shadow caster
key = bpy.data.lights.new(name="Key", type='AREA')
key.energy = 300
key.size = 4.0
key.color = (1.0, 0.98, 0.95)
key_obj = bpy.data.objects.new("Key", key)
bpy.context.scene.collection.objects.link(key_obj)
key_obj.location = (4.0, -3.0, 5.0)
key_obj.rotation_euler = (math.radians(45), 0, math.radians(30))

# Fill light
fill = bpy.data.lights.new(name="Fill", type='AREA')
fill.energy = 100
fill.size = 5.0
fill.color = (0.85, 0.9, 1.0)
fill_obj = bpy.data.objects.new("Fill", fill)
bpy.context.scene.collection.objects.link(fill_obj)
fill_obj.location = (-4.0, -2.0, 3.5)
fill_obj.rotation_euler = (math.radians(60), 0, math.radians(-45))

# Rim / back light
rim = bpy.data.lights.new(name="Rim", type='SPOT')
rim.energy = 200
rim.spot_size = math.radians(60)
rim.color = (1.0, 0.95, 0.9)
rim_obj = bpy.data.objects.new("Rim", rim)
bpy.context.scene.collection.objects.link(rim_obj)
rim_obj.location = (1.0, 4.0, 4.0)
rim_obj.rotation_euler = (math.radians(-150), 0, math.radians(10))

# ─── Camera with depth of field ──────────────────────────────────────

import mathutils

cam = bpy.data.cameras.new("Camera")
cam.lens = 85           # portrait lens — nice DoF compression
cam.dof.use_dof = True
cam.dof.aperture_fstop = 4.0   # moderate depth of field
cam_obj = bpy.data.objects.new("Camera", cam)
bpy.context.scene.collection.objects.link(cam_obj)
cam_obj.location = (5.0, -5.0, 4.0)   # pulled back for full view

# Point camera at origin
direction = mathutils.Vector((0, 0, 0.3)) - cam_obj.location
rot_quat = direction.to_track_quat('-Z', 'Y')
cam_obj.rotation_euler = rot_quat.to_euler()

# Focus on origin
cam.dof.focus_distance = direction.length

bpy.context.scene.camera = cam_obj

# ─── Render settings ─────────────────────────────────────────────────

scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = 256
scene.cycles.use_denoising = True
scene.render.resolution_x = 2400
scene.render.resolution_y = 1800
scene.render.film_transparent = False   # white background with shadows

# ─── World (studio white) ────────────────────────────────────────────
world = bpy.data.worlds.new("StudioWhite")
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
bg.inputs["Strength"].default_value = 1.0
scene.world = world

print("\n✓ Scene ready. Adjust camera, then Render → Render Image (F12).")
print("  Collections: Patches, ControlNet, ControlPoints")
print("  Toggle visibility per collection in the Outliner.")
print("  Ground plane casts shadows on white background.")
