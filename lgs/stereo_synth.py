"""Round 16: pregs_exp P3 "H" stereo synthesis (no Gaussians) for the token region, as a baseline for pixgs.

Same inputs as pix2 (lgs/pixgs.py): recon pixels of sources a and b, their displacements filled by the P3 row rule and
the LR-ok masks. Per target t (pregs_exp p2_synth.render, fill "pp"):
  - 2x2 bilinear forward splat of both sources to x + t g_a / x + (1 - t) g_b, ONE z-buffer over both sides (closer =
    larger key), a contribution counts only within 1 px of the max key;
  - each side is weighted (1 - t) / t times (0.05 + its LR-ok share);
  - holes (no side visible) are push-pull filled.
Extra channel: the core flag of the source pixels, so the paste region (core alpha) comes out of the same splat.
"""
import numpy as np
import torch
import torch.nn.functional as F


def pushpull(img, m, levels=8):
    """img (C, H, W), m (H, W) float: fill pixels with m == 0 from coarser averages (pregs_exp P1)."""
    pyr = [(img * m, m)]
    for _ in range(levels):
        c, w = pyr[-1]
        pyr.append((F.avg_pool2d(c[None], 2, ceil_mode=True)[0], F.avg_pool2d(w[None, None], 2, ceil_mode=True)[0, 0]))
    c, w = pyr[-1]
    est = c / w.clamp(min=1e-6)
    for c, w in reversed(pyr[:-1]):
        up = F.interpolate(est[None], size=c.shape[1:], mode="bilinear", align_corners=False)[0]
        est = torch.where(w > 1e-6, c / w.clamp(min=1e-6), up)
    return est


def _side(img, g, ok, core, mask, key_sign, device):
    ys, xs = np.nonzero(mask & np.isfinite(g[0]))
    q = torch.as_tensor(np.stack([xs, ys]).astype(np.float32), device=device)
    gg = torch.as_tensor(np.stack([g[0, ys, xs], g[1, ys, xs]]).astype(np.float32), device=device)
    col = np.concatenate([img[ys, xs], np.ones((len(ys), 1)), ok[ys, xs, None], core[ys, xs, None]], 1)
    key = torch.as_tensor((key_sign * g[0, ys, xs]).astype(np.float32), device=device)
    return dict(q=q, g=gg, key4=key.repeat(4), col4=torch.as_tensor(col.astype(np.float32), device=device).repeat(4, 1))


def prepare(img_a, img_b, ga, gb, oka, okb, core_a, core_b, rec_a, rec_b, sign, device="cpu"):
    """img (H, W, 3) 0..1; g (2, H, W) filled displacement; ok / core / rec (H, W) bool. key of a = sign * g_a,x, of
    b = -sign * g_b,x (larger = closer)."""
    return {"a": _side(img_a, ga, oka, core_a, rec_a, sign, device),
            "b": _side(img_b, gb, okb, core_b, rec_b, -sign, device), "shape": ga.shape[1:], "device": device}


def _taps(S, w, h, wd):
    n = h * wd
    p = S["q"] + w * S["g"]
    x0f, y0f = torch.floor(p[0]), torch.floor(p[1])
    fx, fy = p[0] - x0f, p[1] - y0f
    x0, y0 = x0f.long(), y0f.long()
    base = y0 * wd + x0
    inx0, inx1 = (x0 >= 0) & (x0 < wd), (x0 >= -1) & (x0 < wd - 1)
    iny0, iny1 = (y0 >= 0) & (y0 < h), (y0 >= -1) & (y0 < h - 1)
    gx, gy = 1 - fx, 1 - fy
    idx = torch.cat([torch.where(inx0 & iny0, base, n), torch.where(inx1 & iny0, base + 1, n),
                     torch.where(inx0 & iny1, base + wd, n), torch.where(inx1 & iny1, base + wd + 1, n)])
    return idx, torch.cat([gx * gy, fx * gy, gx * fy, fx * fy])


def render(P, t, to_numpy=True):
    """-> rgb (H, W, 3), alpha (H, W): covered or hole filled inside the covered footprint, core alpha (H, W)."""
    h, w = P["shape"]
    n, dev = h * w, P["device"]
    zb = torch.full((n + 1,), -float("inf"), device=dev)
    sides = []
    for s, wt_t in (("a", t), ("b", 1 - t)):
        idx, wt = _taps(P[s], wt_t, h, w)
        zb.scatter_reduce_(0, idx, P[s]["key4"], reduce="amax")
        sides.append((s, idx, wt))
    acc = []
    for s, idx, wt in sides:
        wv = wt * (P[s]["key4"] >= zb[idx] - 1.0)
        a = torch.zeros(n + 1, 6, device=dev)
        a.index_add_(0, idx, P[s]["col4"] * wv[:, None])
        acc.append(a[:n].T)
    A, B = acc
    wa, wb = A[3], B[3]
    cova, covb = wa > 0.05, wb > 0.05
    ua = torch.where(cova, (1 - t) * (0.05 + A[4] / wa.clamp(min=1e-6)), torch.zeros_like(wa))
    ub = torch.where(covb, t * (0.05 + B[4] / wb.clamp(min=1e-6)), torch.zeros_like(wb))
    den = (ua + ub).clamp(min=1e-6)
    out = (A[:3] / wa.clamp(min=1e-6) * ua + B[:3] / wb.clamp(min=1e-6) * ub) / den
    # paste region: core of EITHER visible source (round-15 rule: max over the sides)
    core = torch.maximum(torch.where(cova, A[5] / wa.clamp(min=1e-6), torch.zeros_like(wa)),
                         torch.where(covb, B[5] / wb.clamp(min=1e-6), torch.zeros_like(wb)))
    cov = (cova | covb).reshape(h, w)
    img = out.reshape(3, h, w)
    img = torch.where(cov[None], img, pushpull(img, cov.float()))
    core = core.reshape(h, w)
    core = torch.where(cov, core, pushpull(core[None], cov.float())[0])
    # holes inside the footprint (cracks, disocclusions narrower than ~16 px) count as rendered, like P3's filled holes
    foot = -F.max_pool2d(-F.max_pool2d(cov.float()[None, None], 17, 1, 8), 17, 1, 8)[0, 0] > 0.5
    alpha = (cov | foot).float()
    core = -F.max_pool2d(-F.max_pool2d(core.clamp(0, 1)[None, None], 5, 1, 2), 5, 1, 2)[0, 0]   # close by 5 px
    out = (img.permute(1, 2, 0).clamp(0, 1), alpha, core * alpha)
    return tuple(x.cpu().numpy() for x in out) if to_numpy else out
