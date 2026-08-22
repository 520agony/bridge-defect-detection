# 桥梁病害检测 Web 平台

多用户检测系统：注册/登录 + 批量检测 + 结果分析 + 数据集管理 + 检测历史 + 项目说明。

## 启动后端

```bash
cd YOLOv26-Project
./.venv/Scripts/python.exe -m uvicorn web.backend.main:app --host 0.0.0.0 --port 8000
```

- 首次启动自动创建 SQLite 数据库 `web/backend/app.db`
- 模型**懒加载**：启动时不占显存，首次检测才加载 best.pt
- 环境变量：`MODEL_PATH`（权重路径）、`CONF_THRESHOLD`（默认 0.25）、`IMGSZ`（默认 640）

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
| POST | `/detect/batch` | 批量检测 | 是 |
| GET  | `/history` | 检测历史列表 | 是 |
| GET  | `/history/{id}` | 单条历史详情 | 是 |
| DELETE | `/history/{id}` | 删除历史 | 是 |
| GET  | `/stats` | 统计分析 | 是 |
| GET  | `/dataset` | 数据集概览 | 否 |
| GET  | `/health` | 健康检查 | 否 |
| GET  | `/image/{filename}` | 已保存检测图 | 否（uuid 文件名） |

## 安全说明

- 密码用 `pbkdf2_hmac(sha256)` + 随机盐 + 10 万次迭代存储，不存明文
- 会话用随机 token，前端存 localStorage，请求头 `Authorization: Bearer <token>`
- 本实现面向教学/演示，生产环境建议改用 HTTPS + 成熟的会话/JWT 方案
