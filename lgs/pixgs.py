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
