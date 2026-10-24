"""AnySplat feed-forward GS: Gaussians from context views, rendered at held-out camera poses.

Target poses come from one joint VGGT camera-head pass over all cameras of the rig. Target images
are used only for that pose estimate (equivalent to calibrating a fixed rig once); the Gaussians
are predicted from the context views alone.

Rendering goes through gsplat directly so the screen-space / 3D filters can be varied:
  classic + eps2d      original 3DGS dilation (AnySplat default eps2d=0.3 at the input resolution)
  antialiased + eps2d  Mip-Splatting 2D Mip filter (dilation with opacity energy compensation)
  filter3d             Mip-Splatting 3D smoothing filter, sampling rate from the context cameras
"""
import sys
import time

import cv2
import numpy as np
import torch


def sim3_from_cams(c2w_src, c2w_dst):
    """Similarity (s, R, t) mapping camera poses in frame src onto the same cameras in frame dst."""
    rs, cs = c2w_src[:, :3, :3], c2w_src[:, :3, 3]
    rd, cd = c2w_dst[:, :3, :3], c2w_dst[:, :3, 3]
    m = sum(rd[i] @ rs[i].T for i in range(len(rs)))
    u, _, vt = np.linalg.svd(m)
    r = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
    cs0, cd0 = cs.mean(0), cd.mean(0)
    s = np.linalg.norm(cd - cd0, axis=1).sum() / max(np.linalg.norm((cs - cs0) @ r.T, axis=1).sum(), 1e-12)
    t = cd0 - s * r @ cs0
    return s, r, t


def apply_sim3(c2w, s, r, t):
    out = c2w.copy()
    out[..., :3, :3] = r @ c2w[..., :3, :3]
    out[..., :3, 3] = (s * (r @ c2w[..., :3, 3].T)).T + t
    return out


def rot_angle_deg(ra, rb):
    c = (np.trace(ra.T @ rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def mip_filter3d(means, cov, opacity, ctx_c2w, ctx_intr_norm, size, var=0.2, compensate=False):
    """Mip-Splatting 3D smoothing: cov + var*(d/f)^2 I with d/f from the densest context sampling.

    Mip-Splatting also scales opacity by sqrt(det cov / det cov'), which assumes the Gaussians were
    trained with the filter. AnySplat outputs many flat surfels (det cov ~ 0), and that factor drives
    their opacity to zero post hoc (round 2: transparent surfaces, striped background), so it is off
    by default. Returns filtered covariances, opacities and the per-Gaussian filter std.
    """
    nu = torch.zeros(len(means), device=means.device)
    for c2w, intr in zip(ctx_c2w, ctx_intr_norm):
        w2c = torch.linalg.inv(torch.as_tensor(c2w, device=means.device, dtype=torch.float32))
        k = torch.as_tensor(intr, device=means.device, dtype=torch.float32)
        pc = means @ w2c[:3, :3].T + w2c[:3, 3]
        z = pc[:, 2]
        fx, fy = k[0, 0] * size, k[1, 1] * size
        u = fx * pc[:, 0] / z.clamp(min=1e-6) + k[0, 2] * size
        v = fy * pc[:, 1] / z.clamp(min=1e-6) + k[1, 2] * size
        vis = (z > 1e-3) & (u > -0.15 * size) & (u < 1.15 * size) & (v > -0.15 * size) & (v < 1.15 * size)
        nu = torch.where(vis, torch.maximum(nu, fx / z.clamp(min=1e-6)), nu)
    if (nu > 0).any():
        nu = torch.where(nu > 0, nu, nu[nu > 0].min())
    else:
        nu = torch.ones_like(nu)
    s2 = var / nu ** 2
    eye = torch.eye(3, device=means.device)
    cov2 = cov + s2[:, None, None] * eye
    if compensate:
        det0 = torch.linalg.det(cov).clamp(min=0)
        det1 = torch.linalg.det(cov2).clamp(min=1e-30)
        opacity = opacity * torch.sqrt(det0 / det1)
    return cov2, opacity, torch.sqrt(s2)


class AnySplatRunner:
    def __init__(self, anysplat_dir, ckpt, device="cuda"):
        sys.path.insert(0, anysplat_dir)
        from src.model.model.anysplat import AnySplat  # noqa: E402
        from src.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402
        from gsplat import rasterization  # noqa: E402

        self.pose_dec = pose_encoding_to_extri_intri
        self.rasterization = rasterization
        self.device = device
        self.model = AnySplat.from_pretrained(ckpt).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @staticmethod
    def _batch(imgs, device):
        x = np.stack(imgs).astype(np.float32) / 255.0  # V,S,S,3
        return torch.from_numpy(x).permute(0, 3, 1, 2)[None].to(device)

    @torch.no_grad()
    def joint_poses(self, x_all):
        enc = self.model.encoder
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            tokens, _ = enc.aggregator(x_all.to(torch.bfloat16),
                                       intermediate_layer_idx=enc.cfg.intermediate_layer_idx)
        with torch.amp.autocast("cuda", enabled=False):
            tokens = [tk.float() for tk in tokens]
            pose_enc = enc.camera_head(tokens)[-1]
            w2c, intr = self.pose_dec(pose_enc, x_all.shape[-2:])
        pad = torch.tensor([0, 0, 0, 1.0], device=w2c.device).view(1, 1, 1, 4).expand(*w2c.shape[:2], 1, 4)
        c2w = torch.cat([w2c.float(), pad], 2).inverse()
        h, w = x_all.shape[-2:]
        intr = intr.float().clone()
        intr[:, :, 0] /= w
        intr[:, :, 1] /= h
        return c2w[0].cpu().numpy(), intr[0].cpu().numpy()

    @torch.no_grad()
    def encode(self, ctx_imgs, tgt_imgs, size):
        """ctx_imgs/tgt_imgs: lists of SxSx3 uint8 crops. Gaussians + target poses in the GS frame."""
        x_ctx = self._batch(ctx_imgs, self.device)
        x_all = self._batch(list(ctx_imgs) + list(tgt_imgs), self.device)
        nc = len(ctx_imgs)
        torch.cuda.synchronize()
        t0 = time.time()
        self.model.encoder.distill = False
        enc_out = self.model.encoder(x_ctx, global_step=0, visualization_dump=None)
        torch.cuda.synchronize()
        t_enc = time.time() - t0
        ctx_c2w = enc_out.pred_context_pose["extrinsic"][0].float().cpu().numpy()
        ctx_intr = enc_out.pred_context_pose["intrinsic"][0].float().cpu().numpy()

        all_c2w, all_intr = self.joint_poses(x_all)
        s, r, t = sim3_from_cams(all_c2w[:nc], ctx_c2w)
        aligned = apply_sim3(all_c2w, s, r, t)
        base = np.linalg.norm(ctx_c2w[-1, :3, 3] - ctx_c2w[0, :3, 3]) + 1e-12
        g = enc_out.gaussians
        state = {
            "gaussians": g, "size": size,
            "ctx_c2w": ctx_c2w, "ctx_intr": ctx_intr,
            "tgt_c2w": aligned[nc:], "tgt_intr": all_intr[nc:],
            "all_c2w_joint": all_c2w,
        }
        diag = {
            "num_gaussians": int(g.means.shape[1]),
            "time_encode_s": t_enc,
            "sim3_scale": float(s),
            "ctx_center_err_rel_baseline": [float(np.linalg.norm(aligned[i, :3, 3] - ctx_c2w[i, :3, 3]) / base)
                                            for i in range(nc)],
            "ctx_rot_err_deg": [rot_angle_deg(aligned[i, :3, :3], ctx_c2w[i, :3, :3]) for i in range(nc)],
        }
        ctx_rgb, _ = self.render(state, ctx_c2w, ctx_intr, size)
        diag["ctx_self_psnr"] = [float(-10 * np.log10(max(np.mean((ctx_rgb[i] - ctx_imgs[i] / 255.0) ** 2), 1e-10)))
                                 for i in range(nc)]
        return state, diag

    @torch.no_grad()
    def render(self, state, c2w, intr_norm, res, mode="classic", eps2d=0.3, filter3d=False, filter3d_var=0.2,
               filter3d_comp=False):
        """Render views at res x res. Returns (V,res,res,3) colors, (V,res,res) alpha."""
        g, dev = state["gaussians"], self.device
        means = g.means[0].float()
        cov = g.covariances[0].float()
        op = g.opacities[0].float().reshape(-1)
        if filter3d:
            cov, op, _ = mip_filter3d(means, cov, op, state["ctx_c2w"], state["ctx_intr"],
                                      state["size"], filter3d_var, filter3d_comp)
        feat = g.harmonics[0].permute(0, 2, 1).float().contiguous()  # N,K,3
        sh_degree = int(round(np.sqrt(feat.shape[1]))) - 1
        bg = self.model.decoder.background_color.float().to(dev)[None]
        colors, alphas = [], []
        for j in range(len(c2w)):
            viewmat = torch.linalg.inv(torch.as_tensor(c2w[j], dtype=torch.float32, device=dev))[None]
            k = torch.as_tensor(intr_norm[j], dtype=torch.float32, device=dev).clone()
            k[0] *= res
            k[1] *= res
            rgb, alpha, _ = self.rasterization(
                means, g.rotations[0].float(), g.scales[0].float(), op, feat, viewmat, k[None], res, res,
                sh_degree=sh_degree, render_mode="RGB", packed=False, near_plane=1e-10,
                backgrounds=bg, radius_clip=0.1, covars=cov, rasterize_mode=mode, eps2d=float(eps2d))
            colors.append(rgb[0].clamp(0, 1).cpu().numpy())
            alphas.append(alpha[0, ..., 0].cpu().numpy())
        return np.stack(colors), np.stack(alphas)


def distort_maps(K, dist, raw_shape):
    """remap() maps that turn an undistorted pinhole full-frame image into the distorted raw image."""
    h, w = raw_shape
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    pts = np.stack([xs.ravel(), ys.ravel()], 1)[:, None]
    und = cv2.undistortPoints(pts, np.asarray(K, np.float64), np.asarray(dist, np.float64),
                              P=np.asarray(K, np.float64)).reshape(h, w, 2)
    return und[..., 0].astype(np.float32), und[..., 1].astype(np.float32)


def paste_and_warp(render, alpha, window, raw_shape, homography, aligned_shape, dmap=None):
    """Put a square crop render back into the raw frame, then warp into the aligned frame.

    dmap: optional distort_maps() output when the render used calibrated (undistorted) intrinsics.
    """
    x0, y0, side = window
    h, w = raw_shape
    canvas = np.zeros((h, w, 3), np.float32)
    acanvas = np.zeros((h, w), np.float32)
    interp = cv2.INTER_CUBIC if render.shape[0] < side else cv2.INTER_AREA
    canvas[y0:y0 + side, x0:x0 + side] = cv2.resize(render, (side, side), interpolation=interp)
    acanvas[y0:y0 + side, x0:x0 + side] = cv2.resize(alpha, (side, side), interpolation=cv2.INTER_LINEAR)
    win = np.zeros((h, w), np.uint8)
    win[y0:y0 + side, x0:x0 + side] = 1
    if dmap is not None:
        canvas = cv2.remap(canvas, *dmap, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        acanvas = cv2.remap(acanvas, *dmap, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        win = cv2.remap(win, *dmap, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    ha, wa = aligned_shape
    rgb = cv2.warpPerspective(canvas, homography, (wa, ha), flags=cv2.INTER_LINEAR)
    a = cv2.warpPerspective(acanvas, homography, (wa, ha), flags=cv2.INTER_LINEAR)
    valid = cv2.warpPerspective(win, homography, (wa, ha), flags=cv2.INTER_NEAREST) > 0
    return np.clip(rgb, 0, 1), a, valid
