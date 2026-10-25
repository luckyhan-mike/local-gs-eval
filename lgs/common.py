"""Config, image IO, chroma keying and small geometry helpers shared by all scripts."""
import json
import os
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_cfg(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = Path(os.environ.get("LGS_DATA_ROOT", cfg["data_root"]))
    cfg["data_root"] = str(root)
    for key in ("rife_dir", "anysplat_dir"):
        p = Path(cfg[key])
        cfg[key] = str(p if p.is_absolute() else root / p)
    ck = cfg.get("anysplat_ckpt", "lhjiang/anysplat")
    if (root / ck).exists():
        ck = str(root / ck)
    cfg["anysplat_ckpt"] = ck
    work = Path(os.environ.get("LGS_WORK_DIR", cfg.get("work_dir", "work")))
    cfg["work_dir"] = str(work if work.is_absolute() else REPO_ROOT / work)
    return cfg


def cam_path(cfg, kind, k):
    return str(Path(cfg["data_root"]) / cfg[f"{kind}_pattern"].format(k))


def setup_cams(cfg, name):
    ctx = sorted(cfg["setups"][name]["context"])
    targets = [k for k in range(ctx[0], ctx[-1] + 1) if k not in ctx]
    return ctx, targets


def neighbors(ctx, k):
    return max(c for c in ctx if c < k), min(c for c in ctx if c > k)


# ---------------------------------------------------------------- image IO
# imdecode/imencode keeps non-ASCII paths (北极星/魔方) working on every OS.
def imread_rgb(path):
    buf = np.fromfile(path, np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_u8(img):
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return img


def imwrite(path, img):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img = to_u8(img)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise IOError(f"cannot encode {path}")
    buf.tofile(path)


def read_mask(path):
    buf = np.fromfile(path, np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE) > 127


def write_mask(path, m):
    imwrite(path, (m.astype(np.uint8) * 255))


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- masks
def disk(r):
    r = max(int(r), 0)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def dilate(m, r):
    if r <= 0:
        return m.astype(bool)
    return cv2.dilate(m.astype(np.uint8), disk(r)) > 0


def erode(m, r):
    if r <= 0:
        return m.astype(bool)
    return cv2.erode(m.astype(np.uint8), disk(r)) > 0


def chroma_fg(img, ck):
    """Foreground = not green screen and not the black border left by warping."""
    hsv = cv2.cvtColor(to_u8(img), cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green = (h >= ck["h_lo"]) & (h <= ck["h_hi"]) & (s >= ck["s_min"]) & (v >= ck["v_min"])
    if "s_max" in ck:  # low-saturation backdrop (e.g. a grey wall instead of a green screen)
        green &= s <= ck["s_max"]
    fg = (~green) & (to_u8(img).max(-1) > 8)
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, disk(2))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, disk(5))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= ck["min_area_frac"] * fg.size
    return fill_holes(keep[lab])


def fill_holes(m):
    """Green regions enclosed by foreground (e.g. green cube stickers) belong to the foreground."""
    pad = np.pad(m.astype(np.uint8), 1)
    flood = pad.copy()
    cv2.floodFill(flood, np.zeros((pad.shape[0] + 2, pad.shape[1] + 2), np.uint8), (0, 0), 1)
    return (pad | (flood == 0))[1:-1, 1:-1].astype(bool)


def cube_mask(img, ck, min_area=300):
    """Saturated non-screen colors (Rubik's cube stickers); largest blob after closing, holes filled."""
    hsv = cv2.cvtColor(to_u8(img), cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    screen = (h >= ck["h_lo"]) & (h <= ck["h_hi"]) & (v < 170)
    colorful = (s >= 120) & (v >= 100) & ~screen
    m = cv2.morphologyEx(colorful.astype(np.uint8), cv2.MORPH_CLOSE, disk(7))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros(m.shape, bool)
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[best, cv2.CC_STAT_AREA] < min_area:
        return np.zeros(m.shape, bool)
    return fill_holes(lab == best)


def bbox(m, pad=0, shape=None):
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    h, w = shape if shape is not None else m.shape
    return (max(xs.min() - pad, 0), max(ys.min() - pad, 0),
            min(xs.max() + pad + 1, w), min(ys.max() + pad + 1, h))
