# 桥梁病害检测 Web 平台

多用户检测系统：注册/登录 + 批量检测 + 结果分析 + 数据集管理 + 检测历史 + 项目说明。

> 复合病害提示（规则层）：大面积剥落内无模型露筋检出时，自动做锈色线状结构检测，
> 命中则追加虚线框 + “规则提示”徽章（`src/analysis/composite.py`）。规则提示**不是模型检出**，
> 不计入统计与 mAP，报告中明确标注。

## 启动后端

```bash
cd YOLOv26-Project
./.venv/Scripts/python.exe -m uvicorn web.backend.main:app --host 0.0.0.0 --port 8000
```

- 首次启动自动创建 SQLite 数据库 `web/backend/app.db`
- 模型**懒加载**：启动时不占显存，首次检测才加载 best.pt
- 环境变量：`MODEL_PATH`（权重路径）、`CONF_THRESHOLD`（默认 0.25）、`IMGSZ`（默认 640）、
  `FUSION_MODEL_PATH`（融合演示专用权重，默认第三轮 distill，缺失回退主权重）、
  `MAX_VIDEO_FRAMES`（视频帧上限，默认 1800）、`VIDEO_BATCH`（视频批推理帧数，默认 8）
- 并发说明：uvicorn 默认单 worker，长视频处理（1~2 分钟）期间其他请求排队；
  演示人多时可加 `--workers 2` 启动（多占约 1GB 显存）
- 启动自动清理 7 天前的输出视频 / 热力图（检测历史图片保留）

## 打开前端

直接用浏览器打开 `web/frontend/index.html`（前端通过 `http://127.0.0.1:8000` 调后端，已开 CORS）。
或起静态服务：

```bash
./.venv/Scripts/python.exe -m http.server 8080 --directory web/frontend
```

然后访问 http://localhost:8080 。

## 接口一览

| 方法 | 路径 | 说明 | 需登录 |
|------|------|------|--------|
| POST | `/auth/register` | 注册（用户名/密码/邮箱） | 否 |
| POST | `/auth/login` | 登录 | 否 |
| POST | `/auth/logout` | 登出 | 是 |
| GET  | `/auth/me` | 当前用户信息 | 是 |
| POST | `/detect` | 单张检测 | 是 |
| POST | `/detect/frame` | 单帧检测（摄像头实时用，不落盘不入库） | 是 |
| POST | `/detect/video` | 视频检测（逐帧标注，输出标注视频+统计） | 是 |
| GET  | `/fusion/samples` | 融合场景列表（自动扫描） | 是 |
| GET  | `/baselines` | 模型评测基线（总体+分项mAP50，供对比页） | 是 |
| GET  | `/lib` | 演示素材库（图片列表+视频列表） | 是 |
| GET  | `/lib_img/{filename}` | 素材库图片 | 否（随机文件名） |
| GET  | `/lib_video/{filename}` | 素材库视频 | 否（随机文件名） |
| POST | `/lib/detect` | 素材图片直接检测（入库，可导出报告） | 是 |
| POST | `/lib/detect_video` | 素材视频直接检测（返回标注视频+统计） | 是 |
| GET  | `/history` | 检测历史列表 | 是 |
| GET  | `/history/{id}` | 单条历史详情 | 是 |
| DELETE | `/history/{id}` | 删除历史 | 是 |
| GET  | `/report/{det_id}?format=docx\|pdf` | 导出检测报告（Word/PDF，含标注图+明细表） | 是 |
| GET  | `/stats` | 统计分析 | 是 |
| GET  | `/dataset` | 数据集概览 | 否 |
| GET  | `/health` | 健康检查 | 否 |
| GET  | `/image/{filename}` | 已保存检测图 | 否（uuid 文件名） |
| GET  | `/video/{filename}` | 检测输出视频 | 否（uuid 文件名） |

## 视频检测 / 摄像头实时检测

- **视频检测**：前端"视频检测"页上传视频（mp4/avi/mov/webm，≤300MB、≤1800 帧≈60s@30fps），后端逐帧推理（batch=8）输出标注视频。约 50ms/帧，1 分钟视频约 1.5~2 分钟处理完。
  可调环境变量：`MAX_VIDEO_FRAMES`（帧上限，默认 1800）、`VIDEO_BATCH`（批推理帧数，默认 8）。
- **摄像头实时**：前端"实时摄像头"页调用浏览器 `getUserMedia`，逐帧 POST `/detect/frame` 并在 canvas 上叠加 mask/标签，自适应节流（上一帧返回后才发下一帧）。约 5~15 FPS（取决于分辨率与网络）。
  ⚠️ 浏览器安全策略：**手机通过局域网 HTTP 访问时无法开摄像头**，需本机 `127.0.0.1` 或配置 HTTPS。
- 中文路径说明：后端图片/视频读写已全部改用 `imencode/imdecode + tofile/fromfile`，兼容含中文的项目路径。

## 安全说明

- 密码用 `pbkdf2_hmac(sha256)` + 随机盐 + 10 万次迭代存储，不存明文
- 会话用随机 token，前端存 localStorage，请求头 `Authorization: Bearer <token>`
- 本实现面向教学/演示，生产环境建议改用 HTTPS + 成熟的会话/JWT 方案
