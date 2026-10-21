"""
eval_wash_compare.py - wash独立测试集上公平重评（val口径泄漏，还原三轮真实差距）
================================================================================
用法（项目根，一次跑一个，8G卡batch=4）：
  python scripts/eval_wash_compare.py --weight ../第三轮/best.pt --tag distill
  python scripts/eval_wash_compare.py --weight ../第二轮/best.pt --tag m
  python scripts/eval_wash_compare.py --weight experiments/seg_v2_clean/weights/best.pt --tag s_local
输出：experiments/wash_eval/<tag>.json（含总体+逐类Box/Mask mAP）
"""
import argparse
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data", default=str(BASE / "configs" / "data_final_5class_wash.yaml"))
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    from ultralytics import YOLO
    from ultralytics.utils.metrics import ConfusionMatrix  # noqa (确保seg指标齐)

    m = YOLO(args.weight)
    r = m.val(data=args.data, split="test", imgsz=args.imgsz, batch=args.batch,
              device=0, verbose=True, plots=False, save_json=False)
    names = r.names if isinstance(r.names, dict) else {i: n for i, n in enumerate(r.names)}

    def per_cls50(metric):
        """逐类mAP50：all_ap是(nc,10)，取IoU=0.5列"""
        try:
            import numpy as _np
            a = _np.asarray(metric.all_ap)
            if a.ndim == 2 and a.shape[1] >= 1:
                return {names[i]: round(float(v), 4) for i, v in enumerate(a[:, 0])}
        except Exception:
            pass
        return {}

    out = {
        "tag": args.tag, "weight": args.weight, "n_test": 2474,
        "box": {"P": round(float(r.box.mp), 4), "R": round(float(r.box.mr), 4),
                "mAP50": round(float(r.box.map50), 4), "mAP50_95": round(float(r.box.map), 4),
                "per_class_mAP50": per_cls50(r.box)},
        "mask": {"P": round(float(r.seg.mp), 4), "R": round(float(r.seg.mr), 4),
                 "mAP50": round(float(r.seg.map50), 4), "mAP50_95": round(float(r.seg.map), 4),
                 "per_class_mAP50": per_cls50(r.seg)},
        "speed_ms": {k: round(float(v), 2) for k, v in r.speed.items()},
    }
    odir = BASE / "experiments" / "wash_eval"
    odir.mkdir(parents=True, exist_ok=True)
    (odir / f"{args.tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
