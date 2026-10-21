r"""blender_sim_batch.py — 参数化批量仿真（基于 blender_sim01.py v3）
====================================================================
无头跑法：
  blender -b -P blender_sim_batch.py -- out=sample04 seed=11 crack_len=0.8 crack_x0=-0.3 cam_h=1.1 spall_r=0.10 spall_cx=0.30 spall_cy=-0.20 hole_r=0.09 hole_cx=-0.25 hole_cy=0.18 hole_depth=0.035
输出：<OUT>/rgb.png cloud.ply meta.json truth.json（约定与 late_fusion 一致）
真值：crack_length_m=直线长；spall_area_sqm=πr²；hole_area_sqm=πr² + hole_depth_m
"""
import bpy
import json
import math
import os
import random
import sys
from mathutils import Matrix

# ---------- 参数 ----------
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
P = {}
for a in argv:
    if "=" in a:
        k, v = a.split("=", 1)
        P[k.strip()] = v.strip()


def gp(k, d, t=float):
    try:
        return t(P.get(k, d))
    except Exception:
        return t(d)


OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), gp("out", "sample04", str))
SEED = int(gp("seed", 11))
CRACK_LEN = gp("crack_len", 1.0)
CRACK_X0 = gp("crack_x0", -0.5)
CRACK_X1 = CRACK_X0 + CRACK_LEN
CAM_H = gp("cam_h", 1.2)
CAM_X = gp("cam_x", 0.1)
CAM_Y = gp("cam_y", -0.05)
SPALL_R = gp("spall_r", 0.12)
SPALL_C = (gp("spall_cx", 0.35), gp("spall_cy", -0.25))
HOLE_R = gp("hole_r", 0.09)
HOLE_C = (gp("hole_cx", -0.25), gp("hole_cy", 0.18))
HOLE_DEPTH = gp("hole_depth", 0.035)
W, H = 1280, 960
LENS, SENSOR_W = 35.0, 36.0
CRACK_HALF_W = 0.020

rng = random.Random(SEED)
os.makedirs(OUT, exist_ok=True)

# ---------- 1. 清空 ----------
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

# ---------- 2. 桥面板（含裂缝沟槽/剥落坑/坑洞） ----------
bpy.ops.mesh.primitive_grid_add(x_subdivisions=400, y_subdivisions=300, size=4.0)
deck = bpy.context.active_object
deck.name = "BridgeDeck"
deck.scale = (1.0, 0.75, 1.0)
bpy.ops.object.transform_apply(scale=True)
mesh = deck.data
for v in mesh.vertices:
    x, y = v.co.x, v.co.y
    if abs(y) < CRACK_HALF_W and CRACK_X0 <= x <= CRACK_X1:
        v.co.z -= 0.010
    d = math.hypot(x - SPALL_C[0], y - SPALL_C[1])
    if d < SPALL_R:
        v.co.z -= 0.02 * (1.0 - d / SPALL_R)
    dh = math.hypot(x - HOLE_C[0], y - HOLE_C[1])
    if dh < HOLE_R:
        v.co.z -= HOLE_DEPTH * (1.0 - (dh / HOLE_R) ** 2)

# ---------- 3. 混凝土材质 ----------
mat = bpy.data.materials.new("Concrete")
mat.use_nodes = True
nodes, links = mat.node_tree.nodes, mat.node_tree.links
nodes.clear()
out = nodes.new("ShaderNodeOutputMaterial")
bsdf = nodes.new("ShaderNodeBsdfPrincipled")
tex_coord = nodes.new("ShaderNodeTexCoord")
mapping = nodes.new("ShaderNodeMapping")
mapping.inputs["Scale"].default_value = (3.0, 3.0, 1.0)
noise_big = nodes.new("ShaderNodeTexNoise")
noise_big.inputs["Scale"].default_value = 12.0
noise_big.inputs["Detail"].default_value = 4.0
noise_fine = nodes.new("ShaderNodeTexNoise")
noise_fine.inputs["Scale"].default_value = 60.0
noise_fine.inputs["Detail"].default_value = 2.0
ramp = nodes.new("ShaderNodeValToRGB")
ramp.color_ramp.elements[0].position = 0.35
ramp.color_ramp.elements[0].color = (0.52, 0.52, 0.50, 1.0)
ramp.color_ramp.elements[1].position = 0.65
ramp.color_ramp.elements[1].color = (0.70, 0.70, 0.68, 1.0)
mix = nodes.new("ShaderNodeMix")
mix.data_type = "RGBA"
mix.inputs["Factor"].default_value = 0.3
links.new(tex_coord.outputs["UV"], mapping.inputs["Vector"])
links.new(mapping.outputs["Vector"], noise_big.inputs["Vector"])
links.new(mapping.outputs["Vector"], noise_fine.inputs["Vector"])
links.new(noise_big.outputs["Fac"], ramp.inputs["Fac"])
links.new(ramp.outputs["Color"], mix.inputs["A"])
links.new(noise_fine.outputs["Fac"], mix.inputs["B"])
links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
bsdf.inputs["Roughness"].default_value = 0.95
links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
deck.data.materials.append(mat)

# ---------- 4. 裂缝条（波浪，独立seed） ----------
NSEG = 48
_cxs = [CRACK_X0 + (CRACK_X1 - CRACK_X0) * i / NSEG for i in range(NSEG + 1)]
_crack_rng = random.Random(SEED * 100 + 7)
_cyc = [0.006 * math.sin(x * 9.0) + _crack_rng.uniform(-0.005, 0.005) for x in _cxs]
_chw = [_crack_rng.uniform(0.006, 0.012) for _ in _cxs]
_cverts = []
for _x, _yc, _hw in zip(_cxs, _cyc, _chw):
    _cverts.append((_x, _yc - _hw, 0.0015))
    _cverts.append((_x, _yc + _hw, 0.0015))
_cfaces = [(2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2) for i in range(NSEG)]
_crack_mesh = bpy.data.meshes.new("CrackStripMesh")
_crack_mesh.from_pydata(_cverts, [], _cfaces)
_crack_mesh.update()
crack_strip = bpy.data.objects.new("CrackStrip", _crack_mesh)
bpy.context.collection.objects.link(crack_strip)
cmat = bpy.data.materials.new("CrackDark")
cmat.use_nodes = True
_cn, _cl = cmat.node_tree.nodes, cmat.node_tree.links
_cn.clear()
_cout = _cn.new("ShaderNodeOutputMaterial")
cbsdf = _cn.new("ShaderNodeBsdfPrincipled")
_ctex = _cn.new("ShaderNodeTexCoord")
_cmap = _cn.new("ShaderNodeMapping")
_cmap.inputs["Scale"].default_value = (8.0, 2.0, 1.0)
_cnoise = _cn.new("ShaderNodeTexNoise")
_cnoise.inputs["Scale"].default_value = 25.0
_cnoise.inputs["Detail"].default_value = 3.0
_cramp = _cn.new("ShaderNodeValToRGB")
_cramp.color_ramp.elements[0].position = 0.3
_cramp.color_ramp.elements[0].color = (0.07, 0.07, 0.07, 1.0)
_cramp.color_ramp.elements[1].position = 0.7
_cramp.color_ramp.elements[1].color = (0.20, 0.19, 0.18, 1.0)
_cl.new(_ctex.outputs["Generated"], _cmap.inputs["Vector"])
_cl.new(_cmap.outputs["Vector"], _cnoise.inputs["Vector"])
_cl.new(_cnoise.outputs["Fac"], _cramp.inputs["Fac"])
_cl.new(_cramp.outputs["Color"], cbsdf.inputs["Base Color"])
cbsdf.inputs["Roughness"].default_value = 1.0
_cl.new(cbsdf.outputs["BSDF"], _cout.inputs["Surface"])
crack_strip.data.materials.append(cmat)

# ---------- 5. 剥落盘 + 坑洞盘（深色，更暗更深） ----------
bpy.ops.mesh.primitive_cylinder_add(radius=SPALL_R, depth=0.002,
                                    location=(SPALL_C[0], SPALL_C[1], 0.001))
spall_disc = bpy.context.active_object
spall_disc.name = "SpallDisc"
smat = bpy.data.materials.new("SpallDark")
smat.use_nodes = True
sbsdf = next(n for n in smat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
sbsdf.inputs["Base Color"].default_value = (0.25, 0.22, 0.20, 1.0)
sbsdf.inputs["Roughness"].default_value = 1.0
spall_disc.data.materials.append(smat)

bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=HOLE_R, depth=0.002,
                                    location=(HOLE_C[0], HOLE_C[1], 0.001 - HOLE_DEPTH + 0.002))
hole_disc = bpy.context.active_object
hole_disc.name = "HoleDisc"
hmat = bpy.data.materials.new("HoleDark")
hmat.use_nodes = True
hbsdf = next(n for n in hmat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
hbsdf.inputs["Base Color"].default_value = (0.12, 0.11, 0.10, 1.0)
hbsdf.inputs["Roughness"].default_value = 1.0
hole_disc.data.materials.append(hmat)

# ---------- 6. 相机 ----------
cam_data = bpy.data.cameras.new("SimCam")
cam_data.lens = LENS
cam_data.sensor_width = SENSOR_W
cam_data.sensor_fit = "HORIZONTAL"
cam_obj = bpy.data.objects.new("SimCam", cam_data)
bpy.context.collection.objects.link(cam_obj)
cam_obj.location = (CAM_X, CAM_Y, CAM_H)
cam_obj.rotation_euler = (0.0, 0.0, 0.0)
bpy.context.scene.camera = cam_obj

# ---------- 7. 灯光 ----------
bpy.ops.object.light_add(type="SUN", location=(1.5, -1.0, 3.0))
bpy.context.active_object.data.energy = 5.0
bpy.ops.object.light_add(type="SUN", location=(-2.0, 1.5, 2.0))
bpy.context.active_object.data.energy = 1.5

# ---------- 8. 渲染 ----------
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
try:
    scene.eevee.use_gtao = True
except Exception:
    pass
scene.render.resolution_x = W
scene.render.resolution_y = H
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.render.filepath = os.path.join(OUT, "rgb.png")
scene.render.image_settings.file_format = "PNG"
bpy.ops.render.render(write_still=True)

# ---------- 9. 相机内外参 ----------
fx = W * LENS / SENSOR_W
K = [[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]]
w2c = cam_obj.matrix_world.inverted()
R = [[w2c[i][j] for j in range(3)] for i in range(3)]
t = [w2c[0][3], w2c[1][3], w2c[2][3]]
R = [[R[0][0], R[0][1], R[0][2]],
     [-R[1][0], -R[1][1], -R[1][2]],
     [-R[2][0], -R[2][1], -R[2][2]]]
t = [t[0], -t[1], -t[2]]
with open(os.path.join(OUT, "meta.json"), "w") as f:
    json.dump({"K": K, "R": R, "t": t, "W": W, "H": H}, f, indent=1)

# ---------- 10. 点云 ----------
deps = bpy.context.evaluated_depsgraph_get()
eval_obj = deck.evaluated_get(deps)
m = eval_obj.to_mesh()
M = deck.matrix_world
pts = []
for v in m.vertices:
    w = M @ v.co
    pts.append((w.x + rng.gauss(0, 0.002),
                w.y + rng.gauss(0, 0.002),
                w.z + rng.gauss(0, 0.002)))
eval_obj.to_mesh_clear()
with open(os.path.join(OUT, "cloud.ply"), "w") as f:
    f.write("ply\nformat ascii 1.0\n")
    f.write(f"element vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\n")
    f.write("end_header\n")
    for x, y, z in pts:
        f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

# ---------- 11. 真值 ----------
with open(os.path.join(OUT, "truth.json"), "w") as f:
    json.dump({"crack_length_m": round(CRACK_LEN, 4),
               "spall_area_sqm": round(math.pi * SPALL_R ** 2, 6),
               "hole_area_sqm": round(math.pi * HOLE_R ** 2, 6),
               "hole_depth_m": round(HOLE_DEPTH, 4)}, f, indent=1)

print("DONE:", OUT, f"points={len(pts)}")
