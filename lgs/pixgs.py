"""Round 15: pixel-wise Gaussians in the aligned (disparity) space, only for the token region.

Every recon pixel x of source a (pair a-b) becomes one small 2D Gaussian with the colour of that pixel; its position in
a virtual view at t (same t and same aligned frame as the flow branch) is x + t * g_a(x), where g_a = q_b - q_a is the
stereo correspondence in aligned pixels (tools/export_fs_disp.py, FoundationStereo, no calibration). Pixels of b move
by (1 - t) * g_b. Depth order comes from the horizontal displacement itself (the wall, behind the zero-parallax plane,
has the smallest key), so no camera model is needed. No network, no window: cost grows with the token area only, and
the Gaussians stay at the native resolution of the aligned images.

Rendering: gsplat (alpha compositing of the 2D Gaussians placed on a pinhole at depths set by the key) when available,
else a z-buffered bilinear splat in torch (tests / CPU). Sources a and b are rendered separately and merged per pixel:
the clearly nearer one wins, equal depths are blended with weights (1 - t) and t.
"""
import numpy as np
import torch

NAN16 = -32768


def load_disp(path, shape):
    """-> {"ga": (2, H, W), "gb": (2, H, W)} float32, nan where no estimate."""
    z = np.load(path)
    x0, y0, x1, y1 = (int(v) for v in z["box"])
    out = {}
    for name in ("ga", "gb"):
        q = z[name]
        g = np.full((2,) + tuple(shape), np.nan, np.float32)
        g[:, y0:y1, x0:x1] = np.where(q == NAN16, np.nan, q.astype(np.float32) / 4.0)
        out[name] = g
    return out


def depth_sign(gx, fg, valid):
    """+1 / -1 so that key = sign * g_x is larger for the foreground (people) than for the wall."""
    f, b = valid & fg, valid & ~fg
    if f.sum() < 50 or b.sum() < 50:
        return 1.0
    return 1.0 if np.median(gx[f]) >= np.median(gx[b]) else -1.0


def fill_background(g, need, sign, max_px):
    """Fill missing correspondences (mostly pixels of b hidden in a) along the row from the nearer valid neighbour on
    the background side (smaller key), at most max_px away. Only pixels in `need` are filled."""
    g = g.copy()
    h, w = need.shape
    valid = np.isfinite(g[0])
    hole = need & ~valid
    if not hole.any():
        return g
    idx = np.where(valid, np.arange(w)[None, :], -1)
    left = np.maximum.accumulate(idx, axis=1)
    idx_r = np.where(valid, np.arange(w)[None, :], w)
    right = np.minimum.accumulate(idx_r[:, ::-1], axis=1)[:, ::-1]
    ys, xs = np.nonzero(hole)
    l, r = left[ys, xs], right[ys, xs]
    ok_l = (l >= 0) & (xs - l <= max_px)
    ok_r = (r < w) & (r - xs <= max_px)
    kl = np.where(ok_l, sign * g[0, ys, np.clip(l, 0, w - 1)], np.inf)
    kr = np.where(ok_r, sign * g[0, ys, np.clip(r, 0, w - 1)], np.inf)
    use_l = kl <= kr
    src = np.where(use_l, l, r)
    ok = ok_l | ok_r
    for c in range(2):
        g[c, ys[ok], xs[ok]] = g[c, ys[ok], src[ok]]
    return g


def make_set(img, g, mask, t, sign, sigma, max_stretch):
    """Gaussians of one source for a view at fraction t of its displacement: (pos (N, 2), colour (N, 3), key (N,),
    sigma (N, 2))."""
    ys, xs = np.nonzero(mask & np.isfinite(g[0]))
    pos = np.stack([xs + t * g[0, ys, xs], ys + t * g[1, ys, xs]], 1).astype(np.float32)
    dgx = np.zeros_like(g[0])
    dgx[:, 1:-1] = np.abs(g[0, :, 2:] - g[0, :, :-2]) / 2
    stretch = np.clip(1.0 + t * np.nan_to_num(dgx[ys, xs]), 1.0, max_stretch / sigma)
    sig = np.stack([sigma * stretch, np.full_like(stretch, sigma)], 1).astype(np.float32)
    return pos, img[ys, xs].astype(np.float32), (sign * g[0, ys, xs]).astype(np.float32), sig


def _render_gsplat(pos, col, key, sig, shape, k_lo, k_hi, eps2d, device):
    from gsplat import rasterization
    h, w = shape
    f = 1000.0
    p = torch.as_tensor(pos, device=device)
    z = 10.0 - 5.0 * (torch.as_tensor(key, device=device) - k_lo) / max(k_hi - k_lo, 1e-6)  # nearer = smaller depth
    means = torch.stack([(p[:, 0] + 0.5) * z / f, (p[:, 1] + 0.5) * z / f, z], 1)
    s = torch.as_tensor(sig, device=device)
    scales = torch.stack([s[:, 0] * z / f, s[:, 1] * z / f, 0.5 * s[:, 1] * z / f], 1)
    quats = torch.zeros(len(p), 4, device=device); quats[:, 0] = 1
    K = torch.tensor([[f, 0, 0], [0, f, 0], [0, 0, 1.0]], device=device)[None]
    out, alpha, _ = rasterization(means, quats, scales, torch.ones(len(p), device=device) * 0.99,
                                  torch.as_tensor(col, device=device), torch.eye(4, device=device)[None], K, w, h,
                                  render_mode="RGB+ED", near_plane=1e-3, eps2d=eps2d, packed=False)
    rgb, depth = out[0, ..., :3], out[0, ..., 3]
    return rgb, alpha[0, ..., 0], depth


def _render_torch(pos, col, key, sig, shape, k_lo, k_hi, eps2d, device, tol=1.0):
    """z-buffered bilinear splat: each Gaussian gives its 2x2 neighbours bilinear weights; only splats within tol
    (key units) of the nearest key at a pixel count."""
    h, w = shape
    p = torch.as_tensor(pos, device=device)
    c = torch.as_tensor(col, device=device)
    k = torch.as_tensor(key, device=device)
    x0, y0 = torch.floor(p[:, 0]), torch.floor(p[:, 1])
    fx, fy = p[:, 0] - x0, p[:, 1] - y0
    xs = torch.stack([x0, x0 + 1, x0, x0 + 1], 1).long(); ys = torch.stack([y0, y0, y0 + 1, y0 + 1], 1).long()
    wt = torch.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], 1)
    ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h) & (wt > 1e-3)
    idx = (ys * w + xs)[ok]
    kk = k[:, None].expand_as(wt)[ok]
    zb = torch.full((h * w,), -float("inf"), device=device).scatter_reduce(0, idx, kk, reduce="amax")
    win = kk >= zb[idx] - tol
    idx, ww = idx[win], wt[ok][win]
    cc = c[:, None, :].expand(-1, 4, -1)[ok][win]
    acc = torch.zeros(h * w, 3, device=device).index_add_(0, idx, cc * ww[:, None])
    wsum = torch.zeros(h * w, device=device).index_add_(0, idx, ww)
    rgb = (acc / wsum.clamp(min=1e-6)[:, None]).reshape(h, w, 3)
    alpha = wsum.clamp(max=1).reshape(h, w)
    depth = 10.0 - 5.0 * (zb.reshape(h, w) - k_lo) / max(k_hi - k_lo, 1e-6)
    return rgb, alpha, depth


def render_view(sets, t, shape, pcfg, device="cpu", backend=None):
    """sets = [(pos, col, key, sig) for a, (...) for b] -> rgb (H, W, 3) 0..1, alpha (H, W), merged as described above."""
    if backend is None:
        try:
            import gsplat  # noqa: F401
            backend = "gsplat" if torch.cuda.is_available() else "torch"
        except ImportError:
            backend = "torch"
    keys = np.concatenate([s[2] for s in sets if len(s[2])]) if any(len(s[2]) for s in sets) else np.zeros(1)
    k_lo, k_hi = float(keys.min()), float(keys.max())
    fn = _render_gsplat if backend == "gsplat" else _render_torch
    outs = []
    for pos, col, key, sig in sets:
        if len(pos) == 0:
            outs.append((torch.zeros(*shape, 3, device=device), torch.zeros(shape, device=device),
                         torch.full(shape, 1e9, device=device)))
            continue
        outs.append(fn(pos, col, key, sig, shape, k_lo, k_hi, float(pcfg.get("eps2d", 0.3)), device))
    (ca, aa, za), (cb, ab, zb) = outs
    dz = 5.0 * float(pcfg.get("depth_tol_px", 3.0)) / max(k_hi - k_lo, 1e-6)
    za = torch.where(aa > 0.05, za, torch.full_like(za, 1e9))   # gsplat "ED" is already normalised by alpha
    zb = torch.where(ab > 0.05, zb, torch.full_like(zb, 1e9))
    wa, wb = aa * (1 - t), ab * t
    wa = torch.where(za < zb - dz, aa, torch.where(zb < za - dz, torch.zeros_like(aa), wa))
    wb = torch.where(zb < za - dz, ab, torch.where(za < zb - dz, torch.zeros_like(ab), wb))
    s = (wa + wb).clamp(min=1e-6)
    rgb = (wa[..., None] * ca + wb[..., None] * cb) / s[..., None]
    alpha = torch.maximum(aa, ab)
    return rgb.clamp(0, 1).cpu().numpy(), alpha.clamp(0, 1).cpu().numpy(), backend


# ---------------------------------------------------------------- round 16: pix2 (P3 fixes carried over to pixgs)
def fill_rows(g, bad, key_sign):
    """pregs_exp P1/P3 fill of displacements along the row: pixels in `bad` take the nearest valid left / right
    neighbour; linear interpolation when the two neighbours agree (keys within 3 px), else the BACKGROUND one (smaller
    key = key_sign * g_x). Nan where the row has nothing."""
    h, w = bad.shape
    ar = np.broadcast_to(np.arange(w)[None], (h, w))
    left = np.maximum.accumulate(np.where(~bad, ar, -1), axis=1)
    right = np.minimum.accumulate(np.where(~bad, ar, w)[:, ::-1], axis=1)[:, ::-1]
    rows = np.broadcast_to(np.arange(h)[:, None], (h, w))
    has_l, has_r = left >= 0, right < w
    lc, rc = np.clip(left, 0, w - 1), np.clip(right, 0, w - 1)
    key = key_sign * np.nan_to_num(g[0])
    kl, kr = key[rows, lc], key[rows, rc]
    wr = (ar - left) / np.maximum(right - left, 1)
    agree = has_l & has_r & (np.abs(kl - kr) <= 3)
    out = g.copy()
    for c in range(2):
        vl, vr = g[c][rows, lc], g[c][rows, rc]
        bg = np.where(kl <= kr, vl, vr)
        v = np.where(agree, (1 - wr) * vl + wr * vr,
                     np.where(has_l & has_r, bg, np.where(has_l, vl, np.where(has_r, vr, np.nan))))
        out[c] = np.where(bad, v, g[c])
    return out.astype(np.float32)


def lost_mask(g, key_sign):
    """Pixels of this source hidden in the other view: forward-projected to x + g (nearest pixel), they lose the
    z-buffer (closer = larger key) by more than 0.5 px. Together with missing pixels this is the "LR failed" set of
    pregs_exp P3 (one-direction mode)."""
    h, w = g.shape[1:]
    ys, xs = np.nonzero(np.isfinite(g[0]))
    x = np.round(xs + g[0, ys, xs]).astype(np.int64)
    y = np.round(ys + g[1, ys, xs]).astype(np.int64)
    v = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    key = (key_sign * g[0, ys, xs]).astype(np.float32)
    idx = np.where(v, y * w + x, h * w)
    zb = np.full(h * w + 1, -np.inf, np.float32)
    np.maximum.at(zb, idx, np.where(v, key, -np.inf).astype(np.float32))
    lost = np.zeros((h, w), bool)
    lost[ys, xs] = ~v | (key < zb[idx] - 0.5)
    return lost


def prep_side(g_raw, key_sign):
    """-> displacement with LR-failed / missing pixels filled by the P3 row rule, and the LR-ok mask."""
    bad = ~np.isfinite(g_raw[0]) | lost_mask(g_raw, key_sign)
    return fill_rows(g_raw, bad, key_sign), ~bad


def prep_side_torch(g_raw, key_sign, device):
    """prep_side on the device (cummax / cummin along rows, scatter-amax z-buffer); same rule, used by the dense-view
    benchmark. g_raw (2, H, W) numpy -> filled g, ok (torch)."""
    g = torch.as_tensor(np.ascontiguousarray(g_raw, dtype=np.float32), device=device)
    h, w = g.shape[1:]
    fin = torch.isfinite(g[0])
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    gz = torch.nan_to_num(g)
    x = torch.round(xx + gz[0]).long(); y = torch.round(yy + gz[1]).long()
    v = fin & (x >= 0) & (x < w) & (y >= 0) & (y < h)
    key = key_sign * gz[0]
    idx = torch.where(v, y * w + x, torch.full_like(x, h * w)).reshape(-1)
    zb = torch.full((h * w + 1,), -float("inf"), device=device)
    zb.scatter_reduce_(0, idx, torch.where(v, key, torch.full_like(key, -float("inf"))).reshape(-1), reduce="amax")
    lost = fin & (~v | (key < zb[idx].reshape(h, w) - 0.5))
    bad = ~fin | lost
    ar = xx
    left = torch.cummax(torch.where(~bad, ar, torch.full_like(ar, -1)), 1).values
    right = torch.cummin(torch.where(~bad, ar, torch.full_like(ar, w)).flip(1), 1).values.flip(1)
    has_l, has_r = left >= 0, right < w
    lc, rc = left.clamp(0, w - 1), right.clamp(0, w - 1)
    kl, kr = key.gather(1, lc), key.gather(1, rc)
    wr = (ar - left).float() / (right - left).clamp(min=1).float()
    agree = has_l & has_r & ((kl - kr).abs() <= 3)
    out = []
    for c in range(2):
        vl, vr = g[c].gather(1, lc), g[c].gather(1, rc)
        bg = torch.where(kl <= kr, vl, vr)
        nan = torch.full_like(vl, float("nan"))
        val = torch.where(agree, (1 - wr) * vl + wr * vr,
                          torch.where(has_l & has_r, bg, torch.where(has_l, vl, torch.where(has_r, vr, nan))))
        out.append(torch.where(bad, val, g[c]))
    return torch.stack(out), ~bad


def make_set2(img, g, ok, core, mask, t, sign, sigma, max_stretch, edge_px):
    """pix2 Gaussians of one source: channels [r, g, b, core, ok]. The horizontal stretch that closes cracks is only
    applied where the displacement is smooth (|dg_x/dx| <= edge_px): at depth edges cracks stay open (the other source or
    the flow fallback fills them) instead of smearing the foreground colour over the background."""
    ys, xs = np.nonzero(mask & np.isfinite(g[0]))
    pos = np.stack([xs + t * g[0, ys, xs], ys + t * g[1, ys, xs]], 1).astype(np.float32)
    dgx = np.full_like(g[0], np.inf)
    dgx[:, 1:-1] = np.abs(g[0, :, 2:] - g[0, :, :-2]) / 2
    d = np.nan_to_num(dgx[ys, xs], nan=np.inf, posinf=np.inf)
    stretch = np.where(d <= edge_px, np.clip(1.0 + t * np.minimum(d, 1e3), 1.0, max_stretch / sigma), 1.0)
    sig = np.stack([sigma * stretch, np.full_like(stretch, sigma)], 1).astype(np.float32)
    col = np.concatenate([img[ys, xs], core[ys, xs, None], ok[ys, xs, None]], 1).astype(np.float32)
    return pos, col, (sign * g[0, ys, xs]).astype(np.float32), sig


def _render_gsplat_d(pos, col, key, sig, shape, k_lo, k_hi, eps2d, device):
    """As _render_gsplat with any number of channels; channels are normalised by alpha."""
    from gsplat import rasterization
    h, w = shape
    f = 1000.0
    p = torch.as_tensor(pos, device=device)
    z = 10.0 - 5.0 * (torch.as_tensor(key, device=device) - k_lo) / max(k_hi - k_lo, 1e-6)
    means = torch.stack([(p[:, 0] + 0.5) * z / f, (p[:, 1] + 0.5) * z / f, z], 1)
    s = torch.as_tensor(sig, device=device)
    scales = torch.stack([s[:, 0] * z / f, s[:, 1] * z / f, 0.5 * s[:, 1] * z / f], 1)
    quats = torch.zeros(len(p), 4, device=device); quats[:, 0] = 1
    K = torch.tensor([[f, 0, 0], [0, f, 0], [0, 0, 1.0]], device=device)[None]
    c = torch.as_tensor(col, device=device)
    nc = c.shape[1]
    c = torch.cat([c, torch.zeros(len(c), 8 - nc, device=device)], 1) if nc < 8 else c   # 8 + depth = 9 channels: a
    out, alpha, _ = rasterization(means, quats, scales, torch.ones(len(p), device=device) * 0.99,   # size gsplat compiles
                                  c, torch.eye(4, device=device)[None], K, w, h,
                                  render_mode="RGB+ED", near_plane=1e-3, eps2d=eps2d, packed=False)
    a = alpha[0, ..., 0]
    return out[0, ..., :nc] / a.clamp(min=1e-6)[..., None], a, out[0, ..., -1]


def _render_torch_d(pos, col, key, sig, shape, k_lo, k_hi, eps2d, device, tol=1.0):
    """As _render_torch with any number of channels."""
    h, w = shape
    p = torch.as_tensor(pos, device=device)
    c = torch.as_tensor(col, device=device)
    k = torch.as_tensor(key, device=device)
    x0, y0 = torch.floor(p[:, 0]), torch.floor(p[:, 1])
    fx, fy = p[:, 0] - x0, p[:, 1] - y0
    xs = torch.stack([x0, x0 + 1, x0, x0 + 1], 1).long(); ys = torch.stack([y0, y0, y0 + 1, y0 + 1], 1).long()
    wt = torch.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], 1)
    ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h) & (wt > 1e-3)
    idx = (ys * w + xs)[ok]
    kk = k[:, None].expand_as(wt)[ok]
    zb = torch.full((h * w,), -float("inf"), device=device).scatter_reduce(0, idx, kk, reduce="amax")
    win = kk >= zb[idx] - tol
    idx, ww = idx[win], wt[ok][win]
    cc = c[:, None, :].expand(-1, 4, -1)[ok][win]
    acc = torch.zeros(h * w, c.shape[1], device=device).index_add_(0, idx, cc * ww[:, None])
    wsum = torch.zeros(h * w, device=device).index_add_(0, idx, ww)
    chans = (acc / wsum.clamp(min=1e-6)[:, None]).reshape(h, w, -1)
    alpha = wsum.clamp(max=1).reshape(h, w)
    depth = 10.0 - 5.0 * (zb.reshape(h, w) - k_lo) / max(k_hi - k_lo, 1e-6)
    return chans, alpha, depth


class PixSet2:
    """make_set2 built ONCE per pair on the device; at(t) only moves the Gaussians (dense views: one build, many t)."""

    def __init__(self, img, g, ok, core, mask, sign, sigma, max_stretch, edge_px, device):
        ys, xs = np.nonzero(mask & np.isfinite(g[0]))
        dgx = np.full_like(g[0], np.inf)
        dgx[:, 1:-1] = np.abs(g[0, :, 2:] - g[0, :, :-2]) / 2
        d = np.nan_to_num(dgx[ys, xs], nan=np.inf, posinf=np.inf)
        T = lambda x: torch.as_tensor(np.ascontiguousarray(x, dtype=np.float32), device=device)
        self.q = T(np.stack([xs, ys], 1))
        self.g = T(np.stack([g[0, ys, xs], g[1, ys, xs]], 1))
        self.col = T(np.concatenate([img[ys, xs], core[ys, xs, None], ok[ys, xs, None]], 1))
        self.key = T(sign * g[0, ys, xs])
        self.d = T(np.where(d <= edge_px, np.minimum(d, 1e3), 0.0))
        self.sigma, self.cap = sigma, max_stretch / sigma

    def at(self, t):
        stretch = (1.0 + t * self.d).clamp(1.0, self.cap)
        sig = torch.stack([self.sigma * stretch, torch.full_like(stretch, self.sigma)], 1)
        return self.q + t * self.g, self.col, self.key, sig


def render_view2(sets, t, shape, pcfg, device="cpu", backend=None, to_numpy=True):
    """One render per source (channels rgb, core, ok); merged as render_view, but at equal depths each side is weighted
    by (0.05 + its LR-ok share) as in pregs_exp P2/P3, so filled (LR-failed) displacements yield to the other side.
    -> rgb (H, W, 3), alpha (H, W), core alpha (H, W), backend."""
    if backend is None:
        try:
            import gsplat  # noqa: F401
            backend = "gsplat" if torch.cuda.is_available() else "torch"
        except ImportError:
            backend = "torch"
    ne = [s for s in sets if len(s[2])]
    k_lo = min((float(s[2].min()) for s in ne), default=0.0)
    k_hi = max((float(s[2].max()) for s in ne), default=0.0)
    fn = _render_gsplat_d if backend == "gsplat" else _render_torch_d
    outs = []
    for pos, col, key, sig in sets:
        if len(pos) == 0:
            outs.append((torch.zeros(*shape, 5, device=device), torch.zeros(shape, device=device),
                         torch.full(shape, 1e9, device=device)))
            continue
        outs.append(fn(pos, col, key, sig, shape, k_lo, k_hi, float(pcfg.get("eps2d", 0.3)), device))
    (ca, aa, za), (cb, ab, zb) = outs
    dz = 5.0 * float(pcfg.get("depth_tol_px", 3.0)) / max(k_hi - k_lo, 1e-6)
    za = torch.where(aa > 0.05, za, torch.full_like(za, 1e9))
    zb = torch.where(ab > 0.05, zb, torch.full_like(zb, 1e9))
    wa = aa * (1 - t) * (0.05 + ca[..., 4].clamp(0, 1))
    wb = ab * t * (0.05 + cb[..., 4].clamp(0, 1))
    wa2 = torch.where(za < zb - dz, aa, torch.where(zb < za - dz, torch.zeros_like(aa), wa))
    wb2 = torch.where(zb < za - dz, ab, torch.where(za < zb - dz, torch.zeros_like(ab), wb))
    s = (wa2 + wb2).clamp(min=1e-6)
    ch = (wa2[..., None] * ca + wb2[..., None] * cb) / s[..., None]
    alpha = torch.maximum(aa, ab)
    # paste region: core of EITHER source (as round 15's max over the two core renders), closed by 5 px
    core = torch.maximum(ca[..., 3].clamp(0, 1) * aa, cb[..., 3].clamp(0, 1) * ab)
    out = (ch[..., :3].clamp(0, 1), alpha.clamp(0, 1), close_px(core, 5))
    return tuple(x.cpu().numpy() for x in out) + (backend,) if to_numpy else out + (backend,)


def close_px(m, k):
    """Morphological closing of a soft (H, W) map with a k x k square (torch max-pool)."""
    import torch.nn.functional as F
    x = F.max_pool2d(m[None, None], k, 1, k // 2)
    return -F.max_pool2d(-x, k, 1, k // 2)[0, 0]


def gate_by_flow(rgb, flow, core_alpha, thr, blur_px, dilate_px):
    """GT-free paste gate: keep the core only where the render differs clearly from the flow result (mean |RGB|
    difference, blurred, > thr, then dilated). Where both agree the flow pixels stay, so a correct flow is not replaced."""
    import cv2
    d = np.abs(rgb - flow).mean(-1).astype(np.float32)
    if blur_px > 0:
        d = cv2.GaussianBlur(d, (0, 0), blur_px)
    g = (d > thr).astype(np.uint8)
    if dilate_px > 0:
        g = cv2.dilate(g, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1,) * 2))
    return core_alpha * g
