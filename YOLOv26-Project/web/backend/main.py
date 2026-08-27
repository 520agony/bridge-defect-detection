"""
桥梁病害检测 Web 后端 (FastAPI + SQLite)
=========================================
功能: 用户注册/登录、单张与批量检测、检测历史、统计分析、数据集信息

启动:
    ./.venv/Scripts/python.exe -m uvicorn web.backend.main:app --host 0.0.0.0 --port 8000
"""

import os
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import auth, db

# ============================================================
# 配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", PROJECT_ROOT / "experiments" / "yolov26_baseline_v2" / "weights" / "best.pt"))
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.25"))
IMGSZ = int(os.environ.get("IMGSZ", "640"))
UPLOAD_DIR = Path(__file__).resolve().parent / "data" / "images"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_model = None


def get_model():
    """懒加载 YOLO 模型"""
    global _model
    if _model is None:
        from ultralytics import YOLO
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"模型权重不存在: {MODEL_PATH}")
        _model = YOLO(str(MODEL_PATH))
    return _model


# ============================================================
# 数据集信息（7 类，静态统计）
# ============================================================
DATASET_CLASSES = [
    {"id": 0, "name": "crack",         "zh": "裂缝",       "boxes": 25590, "color": "#e6194b"},
    {"id": 1, "name": "spalling",      "zh": "剥落/破损",   "boxes": 51894, "color": "#f58231"},
    {"id": 2, "name": "honeycomb",     "zh": "蜂窝麻面",    "boxes": 7622,  "color": "#ffe119"},
    {"id": 3, "name": "hole",          "zh": "孔洞",       "boxes": 34345, "color": "#3cb44b"},
    {"id": 4, "name": "rebar_exposed", "zh": "露筋",       "boxes": 13374, "color": "#42d4f4"},
    {"id": 5, "name": "seepage",       "zh": "渗水/潮湿",   "boxes": 21519, "color": "#4363d8"},
    {"id": 6, "name": "corrosion",     "zh": "锈蚀/腐蚀",   "boxes": 10753, "color": "#f032e6"},
]
DATASET_SPLITS = {"train": 32996, "val": 6683, "test": 5111}


app = FastAPI(title="桥梁病害检测平台", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


# ============================================================
# 认证
# ============================================================
class RegisterBody(BaseModel):
    username: str
    password: str
    email: str


class LoginBody(BaseModel):
    username: str
    password: str


@app.post("/auth/register")
def register(body: RegisterBody):
    err = auth.validate_register(body.username, body.password, body.email)
    if err:
        raise HTTPException(status_code=400, detail=err)

    conn = db.get_db()
    exists = conn.execute("SELECT id FROM users WHERE username = ?", (body.username,)).fetchone()
    if exists:
        conn.close()
        raise HTTPException(status_code=409, detail="用户名已被占用")

    salt = uuid.uuid4().hex
    phash = auth.hash_password(body.password, salt)
    cur = conn.execute(
        "INSERT INTO users (username, email, password_hash, salt, created_at) VALUES (?,?,?,?,?)",
        (body.username, body.email, phash, salt, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    user_id = cur.lastrowid
    token = auth.create_token()
    conn.execute("INSERT INTO sessions (token, user_id, created_at) VALUES (?,?,?)",
                 (token, user_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()
    return {"token": token, "username": body.username, "email": body.email}


@app.post("/auth/login")
def login(body: LoginBody):
    conn = db.get_db()
    row = conn.execute("SELECT * FROM users WHERE username = ?", (body.username,)).fetchone()
    if row is None or not auth.verify_password(body.password, row["salt"], row["password_hash"]):
        conn.close()
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    token = auth.create_token()
    conn.execute("INSERT INTO sessions (token, user_id, created_at) VALUES (?,?,?)",
                 (token, row["id"], datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()
    return {"token": token, "username": row["username"], "email": row["email"]}


@app.post("/auth/logout")
def logout(authorization: str = Header(default=""), user: dict = Depends(auth.get_current_user)):
    """注销：删除服务端 session，使 token 真正失效"""
    token = authorization.removeprefix("Bearer ").strip() if authorization.startswith("Bearer ") else ""
    if token:
        conn = db.get_db()
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()
        conn.close()
    return {"status": "ok"}


@app.get("/auth/me")
def me(user: dict = Depends(auth.get_current_user)):
    return user


# ============================================================
# 检测
# ============================================================
def _run_detect(img: np.ndarray):
    """对一张图推理，返回 (detections 列表, inference_ms)"""
    model = get_model()
    h, w = img.shape[:2]
    results = model.predict(img, conf=CONF_THRESHOLD, imgsz=IMGSZ, verbose=False)
    r = results[0]
    detections = []
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "class_id": int(box.cls[0]),
            "class": r.names[int(box.cls[0])],
            "confidence": round(float(box.conf[0]), 4),
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        })
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections, round(float(r.speed["inference"]), 2)


def _save_detection(user_id: int, filename: str, img: np.ndarray, detections: list) -> dict:
    """保存检测记录（图片落盘 + 结果入库）"""
    stored_name = f"{uuid.uuid4().hex}_{filename}"
    stored_path = UPLOAD_DIR / stored_name
    cv2.imwrite(str(stored_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

    result_json = __import__("json").dumps(detections, ensure_ascii=False)
    conn = db.get_db()
    cur = conn.execute(
        "INSERT INTO detections (user_id, filename, image_path, result_json, count, created_at) "
        "VALUES (?,?,?,?,?,?)",
        (user_id, filename, stored_name, result_json, len(detections),
         datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    det_id = cur.lastrowid
    conn.commit()
    conn.close()
    return {"id": det_id, "filename": filename, "image": stored_name,
            "count": len(detections), "detections": detections}


@app.post("/detect")
async def detect(file: UploadFile = File(...), user: dict = Depends(auth.get_current_user)):
    img_bytes = await file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="无法解析图片")

    try:
        detections, ms = _run_detect(img)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    rec = _save_detection(user["id"], file.filename or "image.jpg", img, detections)
    h, w = img.shape[:2]
    return {"id": rec["id"], "detections": detections, "count": len(detections),
            "image_size": [w, h], "inference_ms": ms,
            "classes": {c["id"]: c["name"] for c in DATASET_CLASSES}}


@app.post("/detect/batch")
async def detect_batch(files: list[UploadFile] = File(...), user: dict = Depends(auth.get_current_user)):
    results = []
    for f in files:
        img_bytes = await f.read()
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            results.append({"filename": f.filename, "error": "无法解析"})
            continue
        try:
            detections, ms = _run_detect(img)
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e))
        rec = _save_detection(user["id"], f.filename or "image.jpg", img, detections)
        results.append({"id": rec["id"], "filename": rec["filename"],
                        "count": rec["count"], "detections": detections,
                        "image": rec["image"], "inference_ms": ms})
    return {"total": len(results), "results": results,
            "classes": {c["id"]: c["name"] for c in DATASET_CLASSES}}


# ============================================================
# 检测历史
# ============================================================
@app.get("/history")
def history(user: dict = Depends(auth.get_current_user)):
    conn = db.get_db()
    rows = conn.execute(
        "SELECT id, filename, image_path, count, created_at FROM detections "
        "WHERE user_id = ? ORDER BY id DESC LIMIT 200", (user["id"],)).fetchall()
    conn.close()
    return {"history": [dict(r) for r in rows]}


@app.get("/history/{det_id}")
def history_detail(det_id: int, user: dict = Depends(auth.get_current_user)):
    conn = db.get_db()
    row = conn.execute("SELECT * FROM detections WHERE id = ? AND user_id = ?",
                       (det_id, user["id"])).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="记录不存在")
    return {"id": row["id"], "filename": row["filename"], "image": row["image_path"],
            "count": row["count"], "created_at": row["created_at"],
            "detections": __import__("json").loads(row["result_json"])}


@app.delete("/history/{det_id}")
def history_delete(det_id: int, user: dict = Depends(auth.get_current_user)):
    conn = db.get_db()
    row = conn.execute("SELECT image_path FROM detections WHERE id = ? AND user_id = ?",
                       (det_id, user["id"])).fetchone()
    if row is None:
        conn.close()
        raise HTTPException(status_code=404, detail="记录不存在")
    conn.execute("DELETE FROM detections WHERE id = ?", (det_id,))
    conn.commit()
    conn.close()
    (UPLOAD_DIR / row["image_path"]).unlink(missing_ok=True)
    return {"status": "ok"}


# ============================================================
# 统计分析 + 数据集信息
# ============================================================
@app.get("/stats")
def stats(user: dict = Depends(auth.get_current_user)):
    conn = db.get_db()
    rows = conn.execute(
        "SELECT result_json, count, created_at FROM detections WHERE user_id = ?",
        (user["id"],)).fetchall()
    conn.close()

    total_images = len(rows)
    total_boxes = sum(r["count"] for r in rows)
    class_count = {c["id"]: 0 for c in DATASET_CLASSES}
    daily = {}
    for r in rows:
        dets = __import__("json").loads(r["result_json"])
        for d in dets:
            class_count[d["class_id"]] = class_count.get(d["class_id"], 0) + 1
        day = r["created_at"][:10]
        daily[day] = daily.get(day, 0) + 1

    return {
        "total_images": total_images,
        "total_boxes": total_boxes,
        "class_distribution": [{"class": c["name"], "zh": c["zh"], "color": c["color"],
                                "count": class_count[c["id"]]} for c in DATASET_CLASSES],
        "daily": [{"date": k, "count": v} for k, v in sorted(daily.items())],
    }


@app.get("/dataset")
def dataset_info():
    return {"classes": DATASET_CLASSES, "splits": DATASET_SPLITS,
            "total_images": sum(DATASET_SPLITS.values()),
            "total_boxes": sum(c["boxes"] for c in DATASET_CLASSES)}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None, "model_path": str(MODEL_PATH)}


# 已保存检测图的静态访问（uuid 文件名，不可猜测）
app.mount("/image", StaticFiles(directory=str(UPLOAD_DIR)), name="image")

# 前端静态页（挂在最后，让 API 路由优先匹配）
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
