"""Round 9: geometry from a 3D model, texture from the input images.

Round 8 showed the token detector is close to saturated (perfect detection adds 0.2 dB on cam3) and
that the GS patch itself is the bottleneck: AnySplat's colours are soft (18-21 dB in the paste area)
while its geometry is good (ring displacement ~0 px). Here the 3D model only supplies the depth of
the virtual view; colours are sampled from the two neighbouring input images (raw, full resolution)
where the point is visible, weighted by (1 - t, t).

Two geometry sources share everything else:
  gs    expected depth of all Gaussians rendered at the virtual camera; input-view visibility from the
        Gaussians' rendered depth at each input camera
  vggt  AnySplat's VGGT depth maps of the pair's input views, forward-splatted with a z-buffer into the
        virtual camera; input-view visibility from the same VGGT depth maps
Pixel convention: integer pixel coordinates are pixel centres for both the raw images and the
size x size crop grids (as crop_K_to_raw assumes); the 0.5 px offset of the crop grid is ~1 raw px.
"""
import cv2
import numpy as np

from .aligned import project


def backproject(K, c2w, px, depth):
    """Pixels (N,2) with z-depth (N,) -> world points (N,3)."""
    rays = np.c_[px, np.ones(len(px))] @ np.linalg.inv(K).T
    return (rays * depth[:, None]) @ c2w[:3, :3].T + c2w[:3, 3]


def grid_points(K, c2w, depth):
    """World points of every pixel of a depth map (H,W) -> (H*W,3), plus the valid mask (H*W,)."""
    h, w = depth.shape
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    d = depth.reshape(-1).astype(np.float64)
    return backproject(K, c2w, np.c_[xs.reshape(-1), ys.reshape(-1)], d), np.isfinite(d) & (d > 0)


def zbuffer_depth(points, K, c2w, width, height, scale=0.5, fill_iters=6):
    """Forward-splat world points into a pinhole camera; nearest z per pixel.

    Splats at `scale` x the target resolution (so the points are dense enough to leave few cracks),
    fills remaining holes from the farther neighbour (disocclusions belong to the background), then
    upsamples with nearest neighbour. Returns (depth (H,W), valid (H,W))."""
    Ks = np.array(K, np.float64).copy()
    Ks[:2] *= scale
    ws, hs = int(np.ceil(width * scale)), int(np.ceil(height * scale))
    uv, z = project(Ks, c2w, points)
    iu, iv = np.round(uv[:, 0]).astype(np.int64), np.round(uv[:, 1]).astype(np.int64)
    ok = (z > 1e-6) & np.isfinite(z) & (iu >= 0) & (iu < ws) & (iv >= 0) & (iv < hs)
    d = np.full((hs, ws), np.inf)
    np.minimum.at(d, (iv[ok], iu[ok]), z[ok])
    valid = np.isfinite(d)
    d[~valid] = 0
    kern = np.ones((3, 3), np.uint8)
    for _ in range(fill_iters):
        if valid.all():
            break
        grown = cv2.dilate(d.astype(np.float32), kern)  # max over the neighbourhood = farther surface
        nv = cv2.dilate(valid.astype(np.uint8), kern) > 0
        d = np.where(valid, d, grown)
        valid = nv & (d > 0)
    d = cv2.resize(d.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
    valid = cv2.resize(valid.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
    return d, valid


def bilinear(img, x, y):
    """img (H,W,C) sampled at float pixel positions (N,) (cv2.remap caps maps at 32767 rows)."""
    h, w = img.shape[:2]
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    x0 = np.minimum(np.floor(x).astype(np.int64), w - 2)
    y0 = np.minimum(np.floor(y).astype(np.int64), h - 2)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    return ((img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx) * (1 - fy) +
            (img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx) * fy).astype(np.float32)


def sample_sources(X, valid, sources, tol):
    """Colour of world points X (N,3) from input images where each point is visible.

    sources: list of dicts with img (raw float HxWx3), K (raw pixel intrinsics), c2w, depth (size x size
    depth map of that input camera on its crop grid), window (x0, y0, side), weight.
    Returns rgb (N,3), alpha (N,) = 1 where at least one source sees the point, and the per-point
    number of sources that see it."""
    acc = np.zeros((len(X), 3), np.float64)
    wsum = np.zeros(len(X), np.float64)
    nvis = np.zeros(len(X), np.int32)
    for s in sources:
        uv, z = project(s["K"], s["c2w"], X)
        h, w = s["img"].shape[:2]
        x0, y0, side = s["window"]
        size = s["depth"].shape[0]
        cu = (uv[:, 0] - x0) * size / side
        cv = (uv[:, 1] - y0) * size / side
        inside = (valid & (z > 1e-6) & (uv[:, 0] >= 0) & (uv[:, 0] <= w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h - 1)
                  & (cu >= 0) & (cu <= size - 1) & (cv >= 0) & (cv <= size - 1))
        dref = np.zeros(len(X))
        ii = np.nonzero(inside)[0]
        dref[ii] = s["depth"][np.round(cv[ii]).astype(int), np.round(cu[ii]).astype(int)]
        vis = inside & (dref > 0) & (z <= dref * (1 + tol))
        col = np.zeros((len(X), 3), np.float32)
        if vis.any():
            vi = np.nonzero(vis)[0]
            col[vi] = bilinear(s["img"], uv[vi, 0], uv[vi, 1])
        acc += col * (vis * s["weight"])[:, None]
        wsum += vis * s["weight"]
        nvis += vis
    rgb = np.where(wsum[:, None] > 0, acc / np.maximum(wsum, 1e-12)[:, None], 0)
    return rgb.astype(np.float32), (wsum > 0).astype(np.float32), nvis


def reproject_view(K, c2w, depth, valid, sources, tol):
    """Virtual pinhole view: depth (H,W) + validity -> colours sampled from the input images."""
    h, w = depth.shape
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    X = backproject(K, c2w, np.c_[xs.reshape(-1), ys.reshape(-1)], depth.reshape(-1).astype(np.float64))
    rgb, alpha, nvis = sample_sources(X, valid.reshape(-1), sources, tol)
    return rgb.reshape(h, w, 3), alpha.reshape(h, w), nvis.reshape(h, w)
