"""blender_sim01.py v3 - 波浪裂缝版（治sim2real）
===============================================
v3改动（相对v2）：
  - 裂缝条不再是直尺：48段，中线=6mm正弦+±5mm随机抖，半宽6~12mm随机
  - 材质不再纯黑：深灰噪点（0.07~0.20），有层次，真裂缝即如此
  - 几何沟槽加宽到±20mm，接住波浪条
  - 输出改到 sample03（sample02保留B2产物）
  - 相机仍1.2m；meta导出已含diag(1,-1,-1)翻转（B2抓到的bug）

用法（Blender已关掉，待会儿照做）：
  1. 打开 Blender → 顶部切到 Scripting → Open 打开本文件
  2. 点 Run Script，等控制台打印 DONE（EEVEE，约1-3分钟）
  3. 检查 D:/开始做项目/fusion_sim/sample03/ 里有4个文件：
     rgb.png / cloud.ply / meta.json / truth.json
  4. 回来喊我，我用本地一轮best.pt跑B2验证

输出真值：裂缝真长 1.0m；剥落真面积 pi*0.12^2 = 0.045239 m²
相机：(0.1,-0.05,1.2) 垂直下视，视野约1.2x0.9m，裂缝+剥落都在框内
约定：X_cam = R @ X_world + t（late_fusion.load_camera直接读）
"""
import bpy
import json
import math
import os
import random
from mathutils import Matrix

OUT = r"D:\开始做项目\fusion_sim\sample03"
W, H = 1280, 960
LENS = 35.0       # mm
SENSOR_W = 36.0   # mm
CAM_POS = (0.1, -0.05, 1.2)   # v2：1.2m近拍
CRACK_X0, CRACK_X1 = -0.5, 0.5  # 真长1.0m
CRACK_HALF_W = 0.020          # 几何沟槽半宽20mm（接住波浪条）
SPALL_C = (0.35, -0.25)       # v2：移到视野内
SPALL_R = 0.12
random.seed(0)

os.makedirs(OUT, exist_ok=True)

# ---------- 1. 清空默认场景 ----------
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

# ---------- 2. 桥面板网格 ----------
bpy.ops.mesh.primitive_grid_add(x_subdivisions=200, y_subdivisions=150, size=4.0)
deck = bpy.context.active_object
deck.name = "BridgeDeck"
deck.scale = (1.0, 0.75, 1.0)  # 4m x 3m
bpy.ops.object.transform_apply(scale=True)
mesh = deck.data
for v in mesh.vertices:
    x, y = v.co.x, v.co.y
    # 裂缝沟槽：加宽加深，渲染+点云都可见
    if abs(y) < CRACK_HALF_W and CRACK_X0 <= x <= CRACK_X1:
        v.co.z -= 0.010
    # 剥落圆坑
    d = math.hypot(x - SPALL_C[0], y - SPALL_C[1])
    if d < SPALL_R:
        v.co.z -= 0.02 * (1.0 - d / SPALL_R)

# ---------- 3. 混凝土材质（程序噪点，不再纯灰） ----------
mat = bpy.data.materials.new("Concrete")
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links
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
# 灰度两档：0.52 / 0.68，混凝土斑驳感
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

# ---------- 4. 裂缝条（v3：波浪中线+随机宽度+深灰噪点，治sim2real） ----------
NSEG = 48
_cxs = [CRACK_X0 + (CRACK_X1 - CRACK_X0) * i / NSEG for i in range(NSEG + 1)]
_crack_rng = random.Random(7)  # 独立seed，不扰动点云噪声流
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
# 深灰噪点材质（非纯黑，有层次；用Generated坐标，无需UV）
cmat = bpy.data.materials.new("CrackDark")
cmat.use_nodes = True
_cn = cmat.node_tree.nodes
_cl = cmat.node_tree.links
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

# ---------- 5. 剥落深色圆盘 ----------
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

# ---------- 6. 相机（1.2m垂直下视） ----------
cam_data = bpy.data.cameras.new("SimCam")
cam_data.lens = LENS
cam_data.sensor_width = SENSOR_W
cam_data.sensor_fit = "HORIZONTAL"
cam_obj = bpy.data.objects.new("SimCam", cam_data)
bpy.context.collection.objects.link(cam_obj)
cam_obj.location = CAM_POS
cam_obj.rotation_euler = (0.0, 0.0, 0.0)  # 本地 -Z 即世界 -Z，正对桥面
bpy.context.scene.camera = cam_obj

# ---------- 7. 灯光（主+补，避免裂缝死黑） ----------
bpy.ops.object.light_add(type="SUN", location=(1.5, -1.0, 3.0))
sun = bpy.context.active_object
sun.data.energy = 5.0
bpy.ops.object.light_add(type="SUN", location=(-2.0, 1.5, 2.0))
fill = bpy.context.active_object
fill.data.energy = 1.5

# ---------- 8. 渲染 ----------
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
try:
    scene.eevee.use_gtao = True  # 环境光遮蔽，沟槽有层次
except Exception:
    pass
scene.render.resolution_x = W
scene.render.resolution_y = H
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.render.filepath = os.path.join(OUT, "rgb.png")
scene.render.image_settings.file_format = "PNG"
bpy.ops.render.render(write_still=True)

# ---------- 9. 相机内外参（约定 X_cam = R @ X_world + t） ----------
fx = W * LENS / SENSOR_W
fy = fx
K = [[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]]
w2c = cam_obj.matrix_world.inverted()
R = [[w2c[i][j] for j in range(3)] for i in range(3)]
t = [w2c[0][3], w2c[1][3], w2c[2][3]]
# Blender相机看本地-Z（前方点z为负），late_fusion约定前方z为正且v向下：
# X_c = F @ X_b, F=diag(1,-1,-1)，即 R_f = F@R, t_f = F@t
R = [[R[0][0], R[0][1], R[0][2]],
     [-R[1][0], -R[1][1], -R[1][2]],
     [-R[2][0], -R[2][1], -R[2][2]]]
t = [t[0], -t[1], -t[2]]
with open(os.path.join(OUT, "meta.json"), "w") as f:
    json.dump({"K": K, "R": R, "t": t, "W": W, "H": H}, f, indent=1)

# ---------- 10. 点云：网格顶点世界坐标 + 2mm 噪声 → ascii PLY ----------
deps = bpy.context.evaluated_depsgraph_get()
eval_obj = deck.evaluated_get(deps)
m = eval_obj.to_mesh()
M = deck.matrix_world
pts = []
for v in m.vertices:
    w = M @ v.co
    pts.append((w.x + random.gauss(0, 0.002),
                w.y + random.gauss(0, 0.002),
                w.z + random.gauss(0, 0.002)))
eval_obj.to_mesh_clear()
with open(os.path.join(OUT, "cloud.ply"), "w") as f:
    f.write("ply\nformat ascii 1.0\n")
    f.write(f"element vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\n")
    f.write("end_header\n")
    for x, y, z in pts:
        f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

# ---------- 11. 真值 ----------
with open(os.path.join(OUT, "truth.json"), "w") as f:
    json.dump({"crack_length_m": 1.0,
               "spall_area_sqm": round(math.pi * SPALL_R ** 2, 6)}, f, indent=1)

print("DONE:", OUT, f"points={len(pts)}")
