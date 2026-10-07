from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "config" / "default.yaml"


def _merge(a, b):
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path=None):
    """Đọc config/default.yaml, rồi ghi đè bằng file của bạn (chỉ cần ghi các khoá muốn đổi).

    path: một file, hoặc danh sách file ghép lần lượt (file sau thắng), vd [first_real.yaml, fusion_3cam.yaml].
    """
    cfg = yaml.safe_load(DEFAULT.read_text())
    for p in ([path] if isinstance(path, (str, Path)) else (path or [])):
        cfg = _merge(cfg, yaml.safe_load(Path(p).read_text()))
    for k in ("pose", "hand"):
        p = Path(cfg["models"][k])
        cfg["models"][k] = str(p if p.is_absolute() else ROOT / p)
    _resolve_mirror_limits(cfg)
    return cfg


def _resolve_mirror_limits(cfg):
    """`left: mirror` (hoặc `right: mirror`) trong safety.soft_limits_deg / robot.motor_limits_deg: tay đó lấy giới hạn
    của tay kia qua phép phản chiếu (mapping.arm.MIRROR_SIGNS). Tính SAU khi đã ghép mọi file config, nên đổi giới hạn
    tay phải ở file nào (first_real, real...) thì tay trái cũng đổi theo đúng như vậy."""
    from openarm_shadow.mapping.arm import mirror_limits_deg
    for sec, key in (("safety", "soft_limits_deg"), ("robot", "motor_limits_deg")):
        lims = (cfg.get(sec) or {}).get(key)
        if not isinstance(lims, dict):
            continue
        for side, other in (("left", "right"), ("right", "left")):
            if lims.get(side) == "mirror":
                if isinstance(lims.get(other), str) or lims.get(other) is None:
                    raise ValueError(f"{sec}.{key}: {side} = mirror nhưng {other} không có giới hạn cụ thể")
                lims[side] = mirror_limits_deg(lims[other])
