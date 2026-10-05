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

    path: một file, hoặc danh sách file ghép lần lượt (file sau thắng), vd [first_real.yaml, fusion_2cam.yaml].
    """
    cfg = yaml.safe_load(DEFAULT.read_text())
    for p in ([path] if isinstance(path, (str, Path)) else (path or [])):
        cfg = _merge(cfg, yaml.safe_load(Path(p).read_text()))
    return resolve_model_paths(cfg)


def resolve_model_paths(cfg):
    """Đường dẫn model MediaPipe tương đối -> tuyệt đối theo gốc repo (sửa tại chỗ, trả lại cfg)."""
    for k in ("pose", "hand"):
        p = Path(cfg["models"][k])
        cfg["models"][k] = str(p if p.is_absolute() else ROOT / p)
    return cfg
