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
    """Đọc config/default.yaml, rồi ghi đè bằng file của bạn (chỉ cần ghi các khoá muốn đổi)."""
    cfg = yaml.safe_load(DEFAULT.read_text())
    if path:
        cfg = _merge(cfg, yaml.safe_load(Path(path).read_text()))
    for k in ("pose", "hand"):
        p = Path(cfg["models"][k])
        cfg["models"][k] = str(p if p.is_absolute() else ROOT / p)
    return cfg
