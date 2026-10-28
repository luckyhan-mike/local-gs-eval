"""Round 17/18: pix3 = fused pixel Gaussians of one input pair (a, b), jointly optimised on both real views.

Same space as pix2 (lgs/pixgs.py): cam a's aligned frame, position at view fraction s (a = 0, b = 1) is u + s * g with
g the a->b correspondence, depth order key = sign * g_x (larger = closer).
  1. mutual recon regions: a's recon / core region is completed with b's, forward-projected along the correspondences,
     and vice versa, so a surface marked unreliable in only one camera is taken from both;
  2. fusion: a pixel visible in both views (LR-consistent) gives ONE Gaussian (colour = mean of its two samples); b
     pixels are added only where they have no LR-consistent partner in a (seen by b only); a pixels hidden in b stay a
     single-view Gaussian. Single-view pixels carry the background-side displacement of the P3 row fill, or (round 18,
     bg=) the boot background plane of the room (rt_token_exp R3);
  3. (round 18, conf=) two-view pixels whose colours disagree in the two views (photometric error of the match above
     conf) are dropped: no Gaussian, no paste, the flow result stays there (thin structures, wrong matches);
  4. joint optimisation (optional): rendered at s = 0 and s = 1, cropped to the recon region's box, compared with the
     real images of a and b inside their recon regions (L1). Optimised: colour, opacity, size, and the disparity offset
     of the two-view Gaussians only. Snapshots at chosen steps (one run gives 25 / 50 / 100 steps);
  5. one render per view, cropped to the Gaussians' box, channels [rgb, core, visibility].
Round 18: everything is built on the GPU (torch); only the evaluation writes numpy images.
"""
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from lgs.pixgs import prep_side_torch, close_px


def _T(x, device, dtype=torch.float32):
    return x.to(device=device, dtype=dtype) if torch.is_tensor(x) else torch.as_tensor(np.ascontiguousarray(x), device=device, dtype=dtype)


def _close3(m):
    x = F.max_pool2d(m.float()[None, None], 3, 1, 1)
    return (-F.max_pool2d(-x, 3, 1, 1))[0, 0] > 0.5


def _splat_mask(mask, g):
    """Forward-project the pixels of `mask` by g (nearest pixel) -> bool mask, cracks closed by 3 px (GPU)."""
    h, w = mask.shape
    ys, xs = torch.nonzero(mask & torch.isfinite(g[0]), as_tuple=True)
    x = torch.round(xs + g[0, ys, xs]).long(); y = torch.round(ys + g[1, ys, xs]).long()
    v = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    out = torch.zeros(h * w, dtype=torch.bool, device=mask.device)
    out[(y * w + x)[v]] = True
    return _close3(out.reshape(h, w))


def _sample(img, x, y):
    """Bilinear sample of img (H, W, C) at float pixel positions x, y (N,) -> (N, C)."""
    h, w = img.shape[:2]
    grid = torch.stack([(x + 0.5) / w * 2 - 1, (y + 0.5) / h * 2 - 1], -1)[None, None]
    return F.grid_sample(img.permute(2, 0, 1)[None], grid, mode="bilinear", padding_mode="border", align_corners=False)[0, :, 0].T


def _edge_d(g, ys, xs, edge_px):
    dgx = torch.full_like(g[0], float("inf"))
    dgx[:, 1:-1] = (g[0, :, 2:] - g[0, :, :-2]).abs() / 2
    d = torch.nan_to_num(dgx[ys, xs], nan=float("inf"), posinf=float("inf"))
    return torch.where(d <= edge_px, d.clamp(max=1e3), torch.zeros_like(d))


def _photo_err(img_s, img_o, g):
    """Mean |RGB| difference between each pixel of s and its match in o (H, W); inf where g is nan."""
    h, w = img_s.shape[:2]
    yy, xx = torch.meshgrid(torch.arange(h, device=g.device), torch.arange(w, device=g.device), indexing="ij")
    gz = torch.nan_to_num(g)
    o = _sample(img_o, (xx + gz[0]).reshape(-1).float(), (yy + gz[1]).reshape(-1).float()).reshape(h, w, -1)
    e = (img_s - o).abs().mean(-1)
    e = F.avg_pool2d(e[None, None], 3, 1, 1, count_include_pad=False)[0, 0]   # 3x3: one noisy pixel is not a mismatch
    return torch.where(torch.isfinite(g[0]), e, torch.full_like(e, float("inf")))


def build(img_a, img_b, ga_raw, gb_raw, rec_a, rec_b, core_a, core_b, sign, edge_px=1.0, mutual=True, device="cpu",
          bg=None, conf=None, bg_tol=6.0):
    """-> dict of tensors (one row per Gaussian) + region masks for the optimisation. bg = (plane_a, plane_b) boot
    background displacements (2, H, W) for the single-view pixels; conf = photometric threshold for two-view pixels."""
    dev = torch.device(device)
    img_a, img_b = _T(img_a, dev), _T(img_b, dev)
    rec_a, rec_b, core_a, core_b = (_T(m, dev, torch.bool) for m in (rec_a, rec_b, core_a, core_b))
    ga, oka = prep_side_torch(ga_raw, sign, dev)
    gb, okb = prep_side_torch(gb_raw, -sign, dev)
    stats = {}
    if bg is not None:   # single-view pixels whose row fill already lands near the wall (or has nothing): smooth plane
        pa, pb = _T(bg[0], dev), _T(bg[1], dev)
        ua = (~oka) & (~torch.isfinite(ga[0]) | ((ga[0] - pa[0]).abs() < bg_tol))
        ub = (~okb) & (~torch.isfinite(gb[0]) | ((gb[0] - pb[0]).abs() < bg_tol))
        stats["bg_px"] = int((ua & rec_a).sum() + (ub & rec_b).sum())
        stats["single_view_px"] = int(((~oka) & rec_a).sum() + ((~okb) & rec_b).sum())
        ga = torch.where(ua[None], pa, ga)
        gb = torch.where(ub[None], pb, gb)
    if mutual:
        rec_a, rec_b = rec_a | _splat_mask(rec_b, gb), rec_b | _splat_mask(rec_a, ga)
        core_a, core_b = core_a | _splat_mask(core_b, gb), core_b | _splat_mask(core_a, ga)
    if conf is not None:  # two-view pixels whose colours disagree: kept (they still occlude) but never pasted
        bad_a = oka & (_photo_err(img_a, img_b, ga) > conf)
        bad_b = okb & (_photo_err(img_b, img_a, gb) > conf)
        stats["conf_drop_a"] = float((bad_a & rec_a).sum() / rec_a.sum().clamp(min=1))
        stats["conf_drop_b"] = float((bad_b & rec_b).sum() / rec_b.sum().clamp(min=1))
        core_a, core_b = core_a & ~bad_a, core_b & ~bad_b
    h, w = rec_a.shape
    # a side: every recon pixel of a
    ys, xs = torch.nonzero(rec_a & torch.isfinite(ga[0]), as_tuple=True)
    both_a = oka[ys, xs]
    xf, yf = xs.float(), ys.float()
    col_a = img_a[ys, xs]
    pbc = _sample(img_b, xf + ga[0, ys, xs], yf + ga[1, ys, xs])
    col_a = torch.where(both_a[:, None], 0.5 * (col_a + pbc), col_a)
    A = dict(u=torch.stack([xf, yf], 1), g=torch.stack([ga[0, ys, xs], ga[1, ys, xs]], 1), col=col_a,
             core=core_a[ys, xs].float(), both=both_a, d=_edge_d(ga, ys, xs, edge_px))
    # b side: only pixels without an LR-consistent partner in a's recon region
    yb, xb = torch.nonzero(rec_b & torch.isfinite(gb[0]), as_tuple=True)
    pa_x = torch.round(xb + gb[0, yb, xb]).long(); pa_y = torch.round(yb + gb[1, yb, xb]).long()
    inside = (pa_x >= 0) & (pa_x < w) & (pa_y >= 0) & (pa_y < h)
    partner = inside & (rec_a & oka)[pa_y.clamp(0, h - 1), pa_x.clamp(0, w - 1)]
    keep = ~(okb[yb, xb] & partner)
    yb, xb = yb[keep], xb[keep]
    Bs = dict(u=torch.stack([xb + gb[0, yb, xb], yb + gb[1, yb, xb]], 1), g=-torch.stack([gb[0, yb, xb], gb[1, yb, xb]], 1),
              col=img_b[yb, xb], core=core_b[yb, xb].float(), both=torch.zeros_like(okb[yb, xb]), d=_edge_d(gb, yb, xb, edge_px))
    G = {k: torch.cat([A[k], Bs[k]]) for k in A}
    G["key"] = sign * G["g"][:, 0]
    G["sign"] = float(sign)
    G["n_a"], G["n_b"] = len(A["u"]), len(Bs["u"])
    G["rec_a"], G["rec_b"] = rec_a, rec_b
    G["stats"] = stats
    return G


class Fused:
    """Torch parameters of a fused set; at(s) gives the render inputs at view fraction s."""

    def __init__(self, G, sigma, max_stretch, device):
        dev = torch.device(device)
        self.u, self.g, self.d = (_T(G[k], dev) for k in ("u", "g", "d"))
        self.key0 = _T(G["key"], dev)
        self.sign = float(G["sign"])
        self.both = _T(G["both"], dev, torch.bool)
        self.core = _T(G["core"], dev); self.vis = torch.where(self.both, 1.0, 0.5)
        self.col = _T(G["col"], dev).clamp(1e-3, 1 - 1e-3).logit()
        self.dg = torch.zeros(len(self.u), device=dev)
        self.opa = torch.full((len(self.u),), math.log(0.99 / 0.01), device=dev)
        self.lsig = torch.full((len(self.u),), math.log(sigma), device=dev)
        self.cap = max_stretch / sigma
        self.k_lo, self.k_hi = (float(self.key0.min()), float(self.key0.max())) if len(self.u) else (0.0, 1.0)

    def params(self):
        return [self.col, self.dg, self.opa, self.lsig]

    def snapshot(self):
        return [p.detach().clone() for p in self.params()]

    def load(self, snap):
        with torch.no_grad():
            for p, s in zip(self.params(), snap):
                p.copy_(s)

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


def _box(mask, pad, shape):
    """Bounding box (x0, y0, x1, y1) of a bool mask, padded and clipped; None if empty."""
    ys, xs = torch.nonzero(mask, as_tuple=True)
    if len(xs) == 0:
        return None
    h, w = shape
    return (max(int(xs.min()) - pad, 0), max(int(ys.min()) - pad, 0), min(int(xs.max()) + 1 + pad, w), min(int(ys.max()) + 1 + pad, h))


def render_crop(pos, col, key, sig, opa, shape, k_lo, k_hi, eps2d, backend, box):
    """render() restricted to box = (x0, y0, x1, y1): positions shifted, image of the box size."""
    x0, y0, x1, y1 = box
    p = pos - torch.tensor([x0, y0], device=pos.device, dtype=pos.dtype)
    return render(p, col, key, sig, opa, (y1 - y0, x1 - x0), k_lo, k_hi, eps2d, backend)


def optimise(S, img_a, img_b, shape, steps, lr, eps2d, backend, w_dg=1e-3, checkpoints=()):
    """Joint optimisation on the two real views (s = 0 -> image a, s = 1 -> image b) inside their recon regions, each
    view rendered only in the recon region's box. Returns the log and {step: snapshot} for the checkpoints."""
    if steps <= 0 or len(S.u) == 0:
        return {"steps": 0, "time_s": 0.0, "log": [], "dg_abs_mean": 0.0}, {}
    dev = S.u.device
    views = []
    for s, img, rec in ((0.0, img_a, S._rec[0]), (1.0, img_b, S._rec[1])):
        rec = _T(rec, dev, torch.bool)
        box = _box(rec, 8, shape)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        views.append((s, box, _T(img, dev)[y0:y1, x0:x1], rec[y0:y1, x0:x1]))
    for p in S.params():
        p.requires_grad_(True)
    opt = torch.optim.Adam([{"params": [S.col], "lr": lr["color"]}, {"params": [S.dg], "lr": lr["disp"]},
                            {"params": [S.opa], "lr": lr["opacity"]}, {"params": [S.lsig], "lr": lr["scale"]}])
    log, snaps, last = [], {}, None
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.enable_grad():      # callers may run under no_grad (e.g. the dense-view benchmark)
        for it in range(steps):
            opt.zero_grad(set_to_none=True)
            loss = 0.0
            for s, box, tgt, m in views:
                pos, col, key, sig, opa = S.at(s, extra=False)
                rgb, a = render_crop(pos, col, key, sig, opa, shape, S.k_lo, S.k_hi, eps2d, backend, box)
                mm = (m & (a.detach() > 0.5)).float()[..., None]
                loss = loss + ((rgb - tgt).abs() * mm).sum() / (3 * mm.sum()).clamp(min=1)
            loss = loss + w_dg * (S.dg ** 2).mean()
            loss.backward()
            opt.step()
            with torch.no_grad():
                S.lsig.clamp_(math.log(0.3), math.log(2.0))
                S.dg.clamp_(-4.0, 4.0)
            if it in (0, steps - 1) or (it + 1) % 25 == 0:
                last = loss.detach()
                log.append({"iter": it + 1, "loss": last})
            if it + 1 in checkpoints and it + 1 < steps:
                snaps[it + 1] = S.snapshot()
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize()
    dt = time.time() - t0
    for p in S.params():
        p.requires_grad_(False)
    for e in log:
        e["loss"] = float(e["loss"])
    return {"steps": steps, "time_s": dt, "log": log,
            "dg_abs_mean": float(S.dg[S.both].abs().mean()) if S.both.any() else 0.0}, snaps


def make(G, sigma, max_stretch, device):
    S = Fused(G, sigma, max_stretch, device)
    S._rec = (G["rec_a"], G["rec_b"])
    return S


def render_view(S, t, shape, eps2d, backend, to_numpy=True, pad=4):
    """-> rgb (H, W, 3), alpha, core alpha, visibility map (1 both views, 0.5 one view, 0 none). Rendered only inside the
    box of the moved Gaussians and placed into the full frame."""
    h, w = shape
    dev = S.u.device
    rgb = torch.zeros(h, w, 3, device=dev); a = torch.zeros(h, w, device=dev)
    core = torch.zeros(h, w, device=dev); vis = torch.zeros(h, w, device=dev)
    if len(S.u):
        with torch.no_grad():
            pos, col, key, sig, opa = S.at(t)
            ext = float(sig.max()) * 3 + pad
            x0 = max(int(math.floor(float(pos[:, 0].min()) - ext)), 0); x1 = min(int(math.ceil(float(pos[:, 0].max()) + ext)) + 1, w)
            y0 = max(int(math.floor(float(pos[:, 1].min()) - ext)), 0); y1 = min(int(math.ceil(float(pos[:, 1].max()) + ext)) + 1, h)
            if x1 > x0 and y1 > y0:
                ch, ac = render_crop(pos, col, key, sig, opa, shape, S.k_lo, S.k_hi, eps2d, backend, (x0, y0, x1, y1))
                rgb[y0:y1, x0:x1] = ch[..., :3].clamp(0, 1); a[y0:y1, x0:x1] = ac.clamp(0, 1)
                core[y0:y1, x0:x1] = close_px(ch[..., 3].clamp(0, 1) * ac, 5)
                vis[y0:y1, x0:x1] = torch.where(ac > 0.5, ch[..., 4], torch.zeros_like(ac))
    out = (rgb, a, core, vis)
    return tuple(x.cpu().numpy() for x in out) if to_numpy else out


def gate_t(rgb, flow, core, thr, blur_px, dilate_px):
    """GPU version of lgs/pixgs.py gate_by_flow: keep the core where the render differs from the flow result."""
    d = (rgb - flow).abs().mean(-1)[None, None]
    if blur_px > 0:
        r = int(math.ceil(3 * blur_px))
        k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=d.device, dtype=d.dtype) / blur_px) ** 2)
        k = k / k.sum()
        d = F.conv2d(F.pad(d, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
        d = F.conv2d(F.pad(d, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
    g = (d > thr).float()
    if dilate_px > 0:
        g = F.max_pool2d(g, 2 * dilate_px + 1, 1, dilate_px)
    return core * g[0, 0]
