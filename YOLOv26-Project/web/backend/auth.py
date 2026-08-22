"""
认证模块（内置库实现，零依赖）
- 密码: pbkdf2_hmac(sha256) + 随机盐 + 10 万次迭代
- 会话: secrets.token_hex 随机 token，存 sessions 表
- 依赖: get_current_user 从 Authorization: Bearer <token> 解析当前用户
"""

import hashlib
import re
import secrets

from fastapi import Depends, Header, HTTPException

from . import db

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
USERNAME_RE = re.compile(r"^[\w一-鿿-]{3,20}$")  # 3~20 位，字母数字下划线中文横线
_ITERATIONS = 100_000


def hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), _ITERATIONS).hex()


def verify_password(password: str, salt: str, expected_hash: str) -> bool:
    return secrets.compare_digest(hash_password(password, salt), expected_hash)


def validate_register(username: str, password: str, email: str) -> str | None:
    """校验注册输入，返回错误信息（None 表示合法）"""
    if not username or not USERNAME_RE.match(username):
        return "用户名需 3~20 位，仅字母/数字/下划线/中文/横线"
    if not password or len(password) < 6:
        return "密码至少 6 位"
    if not email or not EMAIL_RE.match(email):
        return "请输入有效的邮箱地址"
    return None


def create_token() -> str:
    return secrets.token_hex(32)


def get_current_user(authorization: str = Header(default="")) -> dict:
    """FastAPI 依赖：解析 token，返回当前用户 dict"""
    token = authorization.removeprefix("Bearer ").strip() if authorization.startswith("Bearer ") else ""
    if not token:
        raise HTTPException(status_code=401, detail="未登录")

    conn = db.get_db()
    row = conn.execute(
        "SELECT u.id, u.username, u.email, u.created_at "
        "FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.token = ?",
        (token,),
    ).fetchone()
    conn.close()

    if row is None:
        raise HTTPException(status_code=401, detail="登录已失效，请重新登录")
    return dict(row)
