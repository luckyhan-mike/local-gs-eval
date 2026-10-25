"""Round 10: test-time optimisation (TTO) of the feed-forward Gaussians on the input views.

Round 9 showed the Gaussians' geometry is not accurate enough to sample sharp texture from the input
images, while their colours are view-consistent but soft (AnySplat predicts them at 448 px). Here the
Gaussians are refined against the input views themselves at full crop resolution (~900 px), so the
texture can sharpen while geometry stays close to the feed-forward initialisation.

Parameterisation (keeps the first render bit-identical to the feed-forward one):
  means       position offset, learning rate scaled by the context baseline length
  covariance  cov = L0 (I + A)(I + A)^T L0^T, L0 = Cholesky factor of AnySplat's covariance, A lower
              triangular and initialised to 0 (no assumption on the quaternion convention)
  opacity     logit
  colour      SH DC term (higher orders optional: few input views overfit view dependence)
Only Gaussians in `train_mask` (the token reconstruction region in the default local scope) are
optimised; all others stay fixed but are still rendered. No densification or pruning, so the
per-variant keep masks computed before TTO stay valid. Loss: (1 - w) L1 + w (1 - SSIM).

Round 11 (speed): with ocfg["crop"] each input view is rendered and supervised only inside one box around
the projected trainable Gaussians (plus a margin), instead of the full ~900 px crop; the principal point
is shifted so the box is an exact sub-window of the full view. The wall time to every checkpoint is
logged (CUDA synchronised).
"""
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


def ssim(x, y, win=11, sigma=1.5):
    """Mean SSIM of (B,3,H,W) images in [0,1] with a Gaussian window."""
    g = torch.exp(-((torch.arange(win, device=x.device, dtype=x.dtype) - win // 2) ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())
    w = (g[:, None] * g[None, :]).expand(x.shape[1], 1, win, win).contiguous()

    def f(z):
        return F.conv2d(z, w, padding=win // 2, groups=x.shape[1])

    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx ** 2, f(y * y) - my ** 2, f(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))).mean()


def cov_factor(cov, rel_eps=1e-6):
    """Lower-triangular L0 with L0 L0^T ~= cov (tiny relative jitter for flat surfels)."""
    tr = torch.diagonal(cov, dim1=-2, dim2=-1).sum(-1).clamp(min=1e-20)
    eye = torch.eye(3, device=cov.device, dtype=cov.dtype)
    L, info = torch.linalg.cholesky_ex(cov + (rel_eps * tr)[:, None, None] * eye)
    bad = info != 0
    if bad.any():  # fall back to the eigen-decomposition factor (not triangular, still L L^T = cov)
        e, v = torch.linalg.eigh(cov[bad])
        L[bad] = v * e.clamp(min=0).sqrt()[:, None, :]
    return L


def cov_from(L0, A):
    M = L0 @ (torch.eye(3, device=L0.device, dtype=L0.dtype) + torch.tril(A))
    return M @ M.transpose(-1, -2)


def crop_boxes(means, viewmats, Ks, res, margin, q=0.002):
    """Common (w, h) and per-view top-left (x, y) of a box holding the projections of `means` in every
    view (quantiles q..1-q against outliers) plus `margin` px; boxes are clamped inside the res x res view."""
    pts = torch.cat([means, torch.ones_like(means[:, :1])], 1)
    lo, hi = [], []
    for vm, K in zip(viewmats, Ks):
        cam = (vm @ pts.T).T[:, :3]
        ok = cam[:, 2] > 1e-6
        if int(ok.sum()) == 0:
            lo.append(torch.zeros(2, device=means.device))
            hi.append(torch.full((2,), float(res), device=means.device))
            continue
        uv = (K @ cam[ok].T).T
        uv = uv[:, :2] / uv[:, 2:3]
        lo.append(torch.quantile(uv, q, dim=0) - margin)
        hi.append(torch.quantile(uv, 1 - q, dim=0) + margin)
    lo = torch.stack(lo).clamp(0, res).floor()
    hi = torch.stack(hi).clamp(0, res).ceil()
    w, h = [int(min(res, max(16, float((hi - lo)[:, i].max())))) for i in (0, 1)]
    xy = [(int(min(max(0, float(a[0])), res - w)), int(min(max(0, float(a[1])), res - h))) for a in lo]
    return w, h, xy


def optimize(runner, state, images, res, checkpoints, train_mask, ocfg, eps2d=0.3, mode="classic"):
    """Refine `state["gaussians"]` on the context views. images: context crops (V,res,res,3) float [0,1].

    Returns ({iteration: gaussians namespace}, log) where each namespace has the attributes the
    renderer reads (means, covariances, opacities, rotations, scales, harmonics; batch dim 1); log
    entries carry the loss, the input PSNR on the supervised (cropped) area and, at checkpoints, the
    wall time since the start of the call."""
    cuda = torch.cuda.is_available() and str(runner.device).startswith("cuda")
    if cuda:
        torch.cuda.synchronize()
    t_start = time.time()
    g = state["gaussians"]
    means0 = g.means[0].float().detach()
    cov0 = g.covariances[0].float().detach()
    op0 = g.opacities[0].float().reshape(-1).detach()
    feat0 = g.harmonics[0].permute(0, 2, 1).float().contiguous().detach()  # N,K,3
    idx = torch.nonzero(torch.as_tensor(np.asarray(train_mask), device=means0.device)).reshape(-1)
    L0 = cov_factor(cov0[idx])
    p = {"means": means0[idx].clone(), "A": torch.zeros(len(idx), 3, 3, device=means0.device),
         "logit": torch.logit(op0[idx].clamp(1e-4, 1 - 1e-4)), "dc": feat0[idx, :1].clone()}
    if ocfg.get("sh_rest") and feat0.shape[1] > 1:
        p["rest"] = feat0[idx, 1:].clone()
    for v in p.values():
        v.requires_grad_(True)
    lr = dict(ocfg.get("lr") or {})
    c = np.asarray(state["ctx_c2w"])[:, :3, 3]
    extent = float(np.linalg.norm(c[-1] - c[0])) or 1.0
    groups = [{"params": [p["means"]], "lr": lr.get("means", 1.6e-4) * extent},
              {"params": [p["A"]], "lr": lr.get("cov", 5e-3)},
              {"params": [p["logit"]], "lr": lr.get("opacity", 2.5e-2)},
              {"params": [p["dc"]], "lr": lr.get("dc", 2.5e-3)}]
    if "rest" in p:
        groups.append({"params": [p["rest"]], "lr": lr.get("rest", 1.25e-4)})
    opt = torch.optim.Adam(groups, eps=1e-15)
    gt = torch.as_tensor(np.asarray(images), device=means0.device, dtype=torch.float32)
    Ks = torch.as_tensor(np.array(state["ctx_intr"], np.float64), device=means0.device, dtype=torch.float32).clone()
    Ks[:, :2] *= res
    viewmats = torch.linalg.inv(torch.as_tensor(np.asarray(state["ctx_c2w"]), device=means0.device,
                                                dtype=torch.float32))
    nv = len(gt)
    bg = runner.model.decoder.background_color.float().to(means0.device)[None].expand(nv, -1)
    sh_degree = int(round(np.sqrt(feat0.shape[1]))) - 1
    W = H = int(res)
    ccfg = ocfg.get("crop") or {}
    box = None
    if ccfg.get("enabled") and len(idx):
        W, H, xy = crop_boxes(means0[idx], viewmats, Ks, int(res), float(ccfg.get("margin_px", 32)),
                              float(ccfg.get("quantile", 0.002)))
        Ks[:, 0, 2] -= torch.as_tensor([x for x, _ in xy], device=Ks.device, dtype=Ks.dtype)
        Ks[:, 1, 2] -= torch.as_tensor([y for _, y in xy], device=Ks.device, dtype=Ks.dtype)
        gt = torch.stack([gt[i, y:y + H, x:x + W] for i, (x, y) in enumerate(xy)])
        box = {"w": W, "h": H, "xy": xy, "area_frac": W * H / float(res) ** 2}
    quats0 = g.rotations[0].float().detach()
    scales0 = g.scales[0].float().detach()
    w_ssim = float(ocfg.get("ssim_weight", 0.2))

    def assemble():
        means = means0.index_copy(0, idx, p["means"])
        cov = cov0.index_copy(0, idx, cov_from(L0, p["A"]))
        op = op0.index_copy(0, idx, torch.sigmoid(p["logit"]))
        fi = torch.cat([p["dc"], p["rest"] if "rest" in p else feat0[idx, 1:]], 1)
        feat = feat0.index_copy(0, idx, fi)
        return means, cov, op, feat

    snaps, log = {}, []
    last = max(checkpoints)
    for it in range(1, last + 1):
        means, cov, op, feat = assemble()
        out, _, _ = runner.rasterization(means, quats0, scales0, op, feat, viewmats, Ks, W, H,
                                         sh_degree=sh_degree, packed=False, near_plane=1e-10, backgrounds=bg,
                                         radius_clip=0.1, covars=cov, rasterize_mode=mode, eps2d=float(eps2d))
        pred = out[..., :3].clamp(0, 1)
        l1 = (pred - gt).abs().mean()
        loss = (1 - w_ssim) * l1 + w_ssim * (1 - ssim(pred.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it in checkpoints and cuda:
            torch.cuda.synchronize()
        t_it = time.time() - t_start
        if it == 1 or it in checkpoints or it % 100 == 0:
            mse = ((pred.detach() - gt) ** 2).mean(dim=(1, 2, 3)).clamp(min=1e-10)
            e = {"iter": it, "loss": float(loss.detach()), "ctx_psnr": [float(x) for x in -10 * torch.log10(mse)]}
            if it in checkpoints:
                e["time_s"] = t_it
            if it == 1 and box is not None:
                e["box"] = box
            log.append(e)
        if it in checkpoints:
            with torch.no_grad():
                means, cov, op, feat = assemble()
                snaps[it] = SimpleNamespace(means=means[None].detach(), covariances=cov[None].detach(),
                                            opacities=op[None].detach(), rotations=quats0[None],
                                            scales=scales0[None], harmonics=feat.permute(0, 2, 1)[None].detach())
    return snaps, log
