"""Per-pixel failure signals computed only from the flow network's own outputs (no ground truth)."""
import cv2
import numpy as np

from .common import chroma_fg, dilate


def signals(out, t):
    w0, w1, m, f = out["w0"], out["w1"], out["mask"], out["flow"]
    f0, f1 = f[..., :2], f[..., 2:4]
    # Two warps should agree wherever both cameras see the point and matching is right.
    r_warp = cv2.GaussianBlur(np.abs(w0 - w1).mean(-1), (0, 0), 1.5)
    # Linear camera motion implies (1-t)*F_t->0 + t*F_t->1 = 0; the residual is in pixels.
    r_col = np.linalg.norm((1 - t) * f0 + t * f1, axis=-1)
    # Total disparity I0->I1 seen at this pixel; its horizontal gradient marks depth edges.
    disp = f1[..., 0] - f0[..., 0]
    g_disp = np.abs(cv2.Sobel(disp, cv2.CV_32F, 1, 0, ksize=3)) / 8.0
    # RIFE fusion mask: entropy (undecided) and spatial gradient (abrupt switch).
    mc = np.clip(m, 1e-4, 1 - 1e-4)
    mask_ent = -(mc * np.log2(mc) + (1 - mc) * np.log2(1 - mc))
    gx = cv2.Sobel(m, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(m, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    mask_grad = np.sqrt(gx ** 2 + gy ** 2)
    return {
        "r_warp": r_warp.astype(np.float32),
        "r_col": r_col.astype(np.float32),
        "g_disp": g_disp.astype(np.float32),
        "mask_ent": mask_ent.astype(np.float32),
        "mask_grad": mask_grad.astype(np.float32),
        "mask": m.astype(np.float32),
    }


def risk_mask(sig, out, rcfg, ck):
    """Region the hybrid system would hand to local GS: flow inconsistency near the subject."""
    raw = (sig["r_warp"] > rcfg["warp_thresh"]) | (sig["r_col"] > rcfg["col_thresh"])
    near_fg = dilate(chroma_fg(out["w0"], ck) | chroma_fg(out["w1"], ck), rcfg["fg_dilate_px"])
    return dilate(raw & near_fg, rcfg["dilate_px"]) & near_fg
