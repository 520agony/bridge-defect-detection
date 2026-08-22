"""
病害量化分析包
==============
检测模型的后续量化分析：像素级测量 → 病害等级评估 → 评估报告。

模块：
  - measure : 量化测量（裂缝骨架+距离变换、面积/尺寸代理）
  - grade   : 等级评估（7 类 + 像素/毫米标定）
  - report  : 报告生成（JSON + Markdown）
  - quantify: CLI 入口
"""

# 7 类中文名（与 web 后端 CLASS_INFO 一致）
CLASS_ZH = {
    "crack": "裂缝",
    "spalling": "剥落/破损",
    "honeycomb": "蜂窝麻面",
    "hole": "孔洞",
    "rebar_exposed": "露筋",
    "seepage": "渗水/潮湿",
    "corrosion": "锈蚀/腐蚀",
}
