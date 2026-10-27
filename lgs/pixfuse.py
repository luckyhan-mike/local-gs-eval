"""Round 17: pix3 = fused pixel Gaussians of one input pair (a, b), jointly optimised on both real views.

Same space as pix2 (lgs/pixgs.py): cam a's aligned frame, position at view fraction s (a = 0, b = 1) is u + s * g with
g the a->b correspondence, depth order key = sign * g_x (larger = closer). Differences to pix2:
  1. mutual recon regions: a's recon / core region is completed with b's, forward-projected along the correspondences,
     and vice versa, so a surface marked unreliable in only one camera is taken from both;
  2. fusion: a pixel visible in both views (LR-consistent) gives ONE Gaussian (colour = mean of its two samples); b
     pixels are added only where they have no LR-consistent partner in a (seen by b only); a pixels hidden in b stay a
     single-view Gaussian. Single-view pixels carry the background-side displacement of the P3 row fill;
  3. joint optimisation (optional): rendered at s = 0 and s = 1 and compared with the real images of a and b inside
     their recon regions (L1). Optimised: colour, opacity, size, and the disparity offset of the two-view Gaussians
     only (the depth of a single-view Gaussian is not constrained by the views, so it keeps the background prior);
  4. one render per view, channels [rgb, core, visibility] (visibility 1 = both views, 0.5 = one view).
"""
import math
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lgs.pixgs import prep_side, prep_side_torch, close_px


def _splat_mask(mask, g, shape):
    """Forward-project the pixels of `mask` by g (nearest pixel) -> bool mask, cracks closed by 3 px."""
    h, w = shape
    ys, xs = np.nonzero(mask & np.isfinite(g[0]))
    x = np.round(xs + g[0, ys, xs]).astype(np.int64); y = np.round(ys + g[1, ys, xs]).astype(np.int64)
    v = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    out = np.zeros((h, w), np.uint8)
    out[y[v], x[v]] = 1
    return cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0


def _bilinear(img, x, y):
    h, w = img.shape[:2]
    x = np.clip(x, 0, w - 1.001); y = np.clip(y, 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
            + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)


def _edge_d(g, ys, xs, edge_px):
    dgx = np.full_like(g[0], np.inf)
    dgx[:, 1:-1] = np.abs(g[0, :, 2:] - g[0, :, :-2]) / 2
    d = np.nan_to_num(dgx[ys, xs], nan=np.inf, posinf=np.inf)
    return np.where(d <= edge_px, np.minimum(d, 1e3), 0.0)


def build(img_a, img_b, ga_raw, gb_raw, rec_a, rec_b, core_a, core_b, sign, edge_px=1.0, mutual=True, device="cpu"):
    """-> dict of numpy arrays (one row per Gaussian) + region masks for the optimisation."""
    shape = rec_a.shape
    if str(device).startswith("cuda"):   # same rule on the GPU (tests check numpy == torch)
        ga, oka = (x.cpu().numpy() for x in prep_side_torch(ga_raw, sign, device))
        gb, okb = (x.cpu().numpy() for x in prep_side_torch(gb_raw, -sign, device))
    else:
        ga, oka = prep_side(ga_raw, sign)
        gb, okb = prep_side(gb_raw, -sign)
    if mutual:
        rec_a, rec_b = rec_a | _splat_mask(rec_b, gb, shape), rec_b | _splat_mask(rec_a, ga, shape)
        core_a, core_b = core_a | _splat_mask(core_b, gb, shape), core_b | _splat_mask(core_a, ga, shape)
    # a side: every recon pixel of a
    ys, xs = np.nonzero(rec_a & np.isfinite(ga[0]))
    both_a = oka[ys, xs]
    col_a = img_a[ys, xs].copy()
    pb = _bilinear(img_b, xs + ga[0, ys, xs], ys + ga[1, ys, xs])
    col_a[both_a] = 0.5 * (col_a[both_a] + pb[both_a])
    A = dict(u=np.stack([xs, ys], 1).astype(np.float32), g=np.stack([ga[0, ys, xs], ga[1, ys, xs]], 1),
             col=col_a, core=core_a[ys, xs], both=both_a, d=_edge_d(ga, ys, xs, edge_px))
    # b side: only pixels without an LR-consistent partner in a's recon region
    yb, xb = np.nonzero(rec_b & np.isfinite(gb[0]))
    pa_x = np.round(xb + gb[0, yb, xb]).astype(np.int64); pa_y = np.round(yb + gb[1, yb, xb]).astype(np.int64)
    inside = (pa_x >= 0) & (pa_x < shape[1]) & (pa_y >= 0) & (pa_y < shape[0])
    partner = np.zeros(len(xb), bool)
    pa_xc, pa_yc = np.clip(pa_x, 0, shape[1] - 1), np.clip(pa_y, 0, shape[0] - 1)
    partner[inside] = (rec_a & oka)[pa_yc[inside], pa_xc[inside]]
    keep = ~(okb[yb, xb] & partner)
    yb, xb = yb[keep], xb[keep]
    Bs = dict(u=np.stack([xb + gb[0, yb, xb], yb + gb[1, yb, xb]], 1).astype(np.float32),
              g=-np.stack([gb[0, yb, xb], gb[1, yb, xb]], 1), col=img_b[yb, xb], core=core_b[yb, xb],
              both=okb[yb, xb] & False, d=_edge_d(gb, yb, xb, edge_px))
    G = {k: np.concatenate([A[k], Bs[k]]) for k in A}
    G["key"] = (sign * G["g"][:, 0]).astype(np.float32)
    G["sign"] = float(sign)
    G["n_a"], G["n_b"] = len(A["u"]), len(Bs["u"])
    G["rec_a"], G["rec_b"] = rec_a, rec_b
    return G


class Fused:
    """Torch parameters of a fused set; at(s) gives the render inputs at view fraction s."""

    def __init__(self, G, sigma, max_stretch, device):
        T = lambda x: torch.as_tensor(np.ascontiguousarray(x, dtype=np.float32), device=device)
        self.u, self.g, self.d = T(G["u"]), T(G["g"]), T(G["d"])
        self.key0 = T(G["key"])
        self.sign = float(G["sign"])
        self.both = torch.as_tensor(G["both"], device=device)
        self.core = T(G["core"]); self.vis = torch.where(self.both, 1.0, 0.5)
        self.col = T(G["col"]).clamp(1e-3, 1 - 1e-3).logit().requires_grad_(False)
        self.dg = torch.zeros(len(self.u), device=device)
        self.opa = torch.full((len(self.u),), math.log(0.99 / 0.01), device=device)
        self.lsig = torch.full((len(self.u),), math.log(sigma), device=device)
        self.cap = max_stretch / sigma
        self.k_lo, self.k_hi = float(self.key0.min()), float(self.key0.max())

    def params(self):
        return [self.col, self.dg, self.opa, self.lsig]

    def at(self, s, extra=True):
        gx = self.g[:, 0] + self.dg * self.both
        pos = torch.stack([self.u[:, 0] + s * gx, self.u[:, 1] + s * self.g[:, 1]], 1)
        sig = self.lsig.exp()
        stretch = (1.0 + abs(s) * self.d).clamp(1.0, self.cap)
        sig2 = torch.stack([sig * stretch, sig], 1)
        rgb = torch.sigmoid(self.col)
        col = torch.cat([rgb, self.core[:, None], self.vis[:, None]], 1) if extra else rgb
        key = self.key0 + self.sign * self.dg * self.both      # key = sign * g_x follows the optimised disparity
        return pos, col, key, sig2, torch.sigmoid(self.opa)


def render(pos, col, key, sig, opa, shape, k_lo, k_hi, eps2d, backend):
    """Differentiable render of one set: channels normalised by alpha. gsplat (pinhole at depths from the key, as
    lgs/pixgs.py) or the torch bilinear z-buffer splat (opacity ignored there)."""
    h, w = shape
    dev = pos.device
    if backend == "gsplat":
        from gsplat import rasterization
        f = 1000.0
        z = 10.0 - 5.0 * (key - k_lo) / max(k_hi - k_lo, 1e-6)
        means = torch.stack([(pos[:, 0] + 0.5) * z / f, (pos[:, 1] + 0.5) * z / f, z], 1)
        scales = torch.stack([sig[:, 0] * z / f, sig[:, 1] * z / f, 0.5 * sig[:, 1] * z / f], 1)
        quats = torch.zeros(len(pos), 4, device=dev); quats[:, 0] = 1
        K = torch.tensor([[f, 0, 0], [0, f, 0], [0, 0, 1.0]], device=dev)[None]
        nc = col.shape[1]
        c = torch.cat([col, torch.zeros(len(col), 8 - nc, device=dev)], 1) if nc < 8 else col
        out, alpha, _ = rasterization(means, quats, scales, opa, c, torch.eye(4, device=dev)[None], K, w, h,
                                      render_mode="RGB+ED", near_plane=1e-3, eps2d=eps2d, packed=False)
        a = alpha[0, ..., 0]
        return out[0, ..., :nc] / a.clamp(min=1e-6)[..., None], a
    x0, y0 = torch.floor(pos[:, 0]).detach(), torch.floor(pos[:, 1]).detach()
    fx, fy = pos[:, 0] - x0, pos[:, 1] - y0
    xs = torch.stack([x0, x0 + 1, x0, x0 + 1], 1).long(); ys = torch.stack([y0, y0, y0 + 1, y0 + 1], 1).long()
    wt = torch.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], 1)
    ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h) & (wt.detach() > 1e-3)
    idx = (ys * w + xs)[ok]
    kk = key.detach()[:, None].expand_as(wt)[ok]
    zb = torch.full((h * w,), -float("inf"), device=dev).scatter_reduce(0, idx, kk, reduce="amax")
    win = kk >= zb[idx] - 1.0
    idx, ww = idx[win], wt[ok][win]
    cc = col[:, None, :].expand(-1, 4, -1)[ok][win]
    acc = torch.zeros(h * w, col.shape[1], device=dev).index_add(0, idx, cc * ww[:, None])
    wsum = torch.zeros(h * w, device=dev).index_add(0, idx, ww)
    return (acc / wsum.clamp(min=1e-6)[:, None]).reshape(h, w, -1), wsum.clamp(max=1).reshape(h, w)


def optimise(S, img_a, img_b, shape, steps, lr, eps2d, backend, w_dg=1e-3):
    """Joint optimisation on the two real views (s = 0 -> image a, s = 1 -> image b) inside their recon regions."""
    if steps <= 0:
        return {"steps": 0, "time_s": 0.0}
    dev = S.u.device
    tgt = {0.0: torch.as_tensor(img_a, device=dev), 1.0: torch.as_tensor(img_b, device=dev)}
    m = {0.0: torch.as_tensor(S._rec[0], device=dev), 1.0: torch.as_tensor(S._rec[1], device=dev)}
    for p in S.params():
        p.requires_grad_(True)
    opt = torch.optim.Adam([{"params": [S.col], "lr": lr["color"]}, {"params": [S.dg], "lr": lr["disp"]},
                            {"params": [S.opa], "lr": lr["opacity"]}, {"params": [S.lsig], "lr": lr["scale"]}])
    log = []
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.enable_grad():      # callers may run under no_grad (e.g. the dense-view benchmark)
        for it in range(steps):
            opt.zero_grad(set_to_none=True)
            loss = 0.0
            for s in (0.0, 1.0):
                pos, col, key, sig, opa = S.at(s, extra=False)
                rgb, a = render(pos, col, key, sig, opa, shape, S.k_lo, S.k_hi, eps2d, backend)
                mm = m[s] & (a.detach() > 0.5)
                loss = loss + (rgb[mm] - tgt[s][mm]).abs().mean()
            loss = loss + w_dg * (S.dg ** 2).mean()
            loss.backward()
            opt.step()
            with torch.no_grad():
                S.lsig.clamp_(math.log(0.3), math.log(2.0))
                S.dg.clamp_(-4.0, 4.0)
            if it in (0, steps - 1) or (it + 1) % 25 == 0:
                log.append({"iter": it + 1, "loss": float(loss)})
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize()
    for p in S.params():
        p.requires_grad_(False)
    return {"steps": steps, "time_s": time.time() - t0, "log": log,
            "dg_abs_mean": float(S.dg[S.both].abs().mean()) if S.both.any() else 0.0}


def make(G, sigma, max_stretch, device):
    S = Fused(G, sigma, max_stretch, device)
    S._rec = (G["rec_a"], G["rec_b"])
    return S


def render_view(S, t, shape, eps2d, backend, to_numpy=True):
    """-> rgb (H, W, 3), alpha, core alpha, visibility map (1 both views, 0.5 one view, 0 none)."""
    with torch.no_grad():
        pos, col, key, sig, opa = S.at(t)
        ch, a = render(pos, col, key, sig, opa, shape, S.k_lo, S.k_hi, eps2d, backend)
        core = close_px(ch[..., 3].clamp(0, 1) * a, 5)
        vis = torch.where(a > 0.5, ch[..., 4], torch.zeros_like(a))
        out = (ch[..., :3].clamp(0, 1), a.clamp(0, 1), core, vis)
    return tuple(x.cpu().numpy() for x in out) if to_numpy else out
