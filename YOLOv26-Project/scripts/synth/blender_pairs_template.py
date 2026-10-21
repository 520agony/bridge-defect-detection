"""
Blender 配对样本生成模板（裂缝示例，其余 4 类按注释 TODO 扩展）
================================================================
跑法（Blender 5.1 + TLSynth v3，需先在 GUI 里装好 TLSynth 插件）：
    blender --background --python blender_pairs_template.py -- \
        --bridge bridge_boxgirder.obj --out datasets/synth_pairs/pair_001 \
        --crack-len 0.5 --crack-width 0.008

本脚本只做：建模（桥+病害）+ 相机渲染 RGB + 导出相机内外参 meta.json。
TLSynth 激光扫描在 Blender GUI 里做（插件面板 -> 放扫描仪 -> 导出 PLY），
导出的 cloud.ply 与本脚本的 image.jpg / meta.json 落到同一 pair_xxx/ 目录。

注意：本模板在无 Blender 环境下仅做语法检查，未逐行实测，
第一次跑先用默认参数出一组，确认无报错再批量改。
"""

import argparse
import json
import math
import os
import sys

import bpy


# ---------------- 参数 ----------------
def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--bridge", required=True, help="桥梁 OBJ 路径")
    p.add_argument("--out", required=True, help="输出目录 pair_xxx")
    p.add_argument("--crack-len", type=float, default=0.5, help="裂缝真长（米）")
    p.add_argument("--crack-width", type=float, default=0.008, help="裂缝真宽（米）")
    p.add_argument("--cam-dist", type=float, default=5.0, help="相机距桥面距离（米）")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--lens", type=float, default=35.0, help="焦距 mm")
    return p.parse_args(argv)


# ---------------- 场景 ----------------
def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def import_obj(path):
    """Blender 4.x/5.x 用 wm.obj_import，老版本用 import_scene.obj"""
    try:
        bpy.ops.wm.obj_import(filepath=path)
    except AttributeError:
        bpy.ops.import_scene.obj(filepath=path)


def add_crack_ribbon(length, width, location=(0, 0, 0.001)):
    """裂缝：桥面表面贴一条下沉的暗色窄带（真值 = length × width）。
    TODO 孔洞/剥落：同位置改用 boolean 挖坑；露筋/泛碱：改金属/白色材质贴片。
    """
    bpy.ops.mesh.primitive_plane_add(size=1, location=location)
    ribbon = bpy.context.active_object
    ribbon.name = "defect_crack"
    ribbon.scale = (length, width, 1)
    bpy.ops.object.transform_apply(scale=True)
    # 中线下沉 5mm 形成沟槽
    mesh = ribbon.data
    for v in mesh.vertices:
        if abs(v.co.y) < width * 0.25:
            v.co.z -= 0.005
    mat = bpy.data.materials.new("crack_dark")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.02, 0.02, 0.02, 1.0)
    bsdf.inputs["Roughness"].default_value = 1.0
    mesh.materials.append(mat)
    return ribbon


def setup_camera(dist, W, H, lens_mm):
    cam_data = bpy.data.cameras.new("pair_cam")
    cam_data.lens = lens_mm
    cam_data.sensor_width = 36.0
    cam = bpy.data.objects.new("pair_cam", cam_data)
    bpy.context.collection.objects.link(cam)
    # 相机在 +Z 看向原点（桥面）
    cam.location = (0, 0, dist)
    cam.rotation_euler = (0, 0, 0)  # 默认朝 -Z，正好俯视
    scene = bpy.context.scene
    scene.camera = cam
    scene.render.resolution_x = W
    scene.render.resolution_y = H
    scene.render.resolution_percentage = 100
    scene.render.engine = "BLENDER_EEVEE"  # 无头快速；要质感改 CYCLES
    return cam


def export_camera_meta(cam, W, H, path):
    """导出 K,R,t（世界单位米；X_cam = R @ X_world + t）"""
    bpy.context.view_layer.update()
    M = cam.matrix_world
    R_wc = M.to_3x3()
    C = M.to_translation()
    R = R_wc.transposed()
    t = -(R @ C)
    fx = cam.data.lens * W / cam.data.sensor_width
    fy = fx  # 像素方形假设
    K = [[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]]
    meta = {
        "K": K, "R": [list(r) for r in R], "t": list(t),
        "W": W, "H": H,
        "lens_mm": cam.data.lens, "sensor_mm": cam.data.sensor_width,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    clear_scene()
    import_obj(args.bridge)
    crack = add_crack_ribbon(args.crack_len, args.crack_width)
    cam = setup_camera(args.cam_dist, args.width, args.height, args.lens)

    # 灯光：简单三点布光，保证桥面无纯黑
    bpy.ops.object.light_add(type="SUN", location=(5, -5, 10))

    scene = bpy.context.scene
    img_path = os.path.join(args.out, "image.jpg")
    scene.render.filepath = img_path
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = 92
    bpy.ops.render.render(write_still=True)

    export_camera_meta(cam, args.width, args.height,
                       os.path.join(args.out, "meta.json"))

    # 真值（建模时已知，直接写死，后续做误差对比用）
    gt = {"class": "crack", "length_m": args.crack_len,
          "width_m": args.crack_width, "unit": "meter"}
    with open(os.path.join(args.out, "gt.json"), "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2)

    print("PAIR OK:", args.out)
    print("下一步：在 Blender GUI 打开本场景 -> TLSynth 放扫描仪 -> 导出 PLY 为",
          os.path.join(args.out, "cloud.ply"))


if __name__ == "__main__":
    main()
