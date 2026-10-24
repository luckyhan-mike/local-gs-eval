"""AnySplat feed-forward GS: Gaussians from context views, rendering at held-out camera poses.

Target poses come from one joint VGGT camera-head pass over all cameras of the rig. Target images
are used only for that pose estimate (equivalent to calibrating a fixed rig once); the Gaussians
are predicted from the context views alone.
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


class AnySplatRunner:
    def __init__(self, anysplat_dir, ckpt, device="cuda"):
        sys.path.insert(0, anysplat_dir)
        from src.model.model.anysplat import AnySplat  # noqa: E402
        from src.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402

        self.pose_dec = pose_encoding_to_extri_intri
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
        with torch.amp.autocast("cuda", enabled=False):
            tokens, _ = enc.aggregator(x_all.to(torch.bfloat16),
                                       intermediate_layer_idx=enc.cfg.intermediate_layer_idx)
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
    def render(self, gaussians, c2w, intr_norm, size):
        v = len(c2w)
        out = self.model.decoder.forward(
            gaussians,
            torch.from_numpy(c2w).float()[None].to(self.device),
            torch.from_numpy(intr_norm).float()[None].to(self.device),
            torch.ones(1, v, device=self.device) * 0.01,
            torch.ones(1, v, device=self.device) * 100,
            (size, size),
        )
        color = out.color[0].permute(0, 2, 3, 1).float().cpu().numpy()  # V,S,S,3
        alpha = out.alpha[0].float().cpu().numpy().reshape(v, size, size)
        return color, alpha

    def run(self, ctx_imgs, tgt_imgs, size):
        """ctx_imgs/tgt_imgs: lists of SxSx3 uint8 crops. Returns renders, poses and diagnostics."""
        dev = self.device
        x_ctx = self._batch(ctx_imgs, dev)
        x_all = self._batch(list(ctx_imgs) + list(tgt_imgs), dev)
        nc = len(ctx_imgs)

        torch.cuda.synchronize()
        t0 = time.time()
        self.model.encoder.distill = False
        enc_out = self.model.encoder(x_ctx, global_step=0, visualization_dump=None)
        torch.cuda.synchronize()
        t_enc = time.time() - t0
        gaussians = enc_out.gaussians
        ctx_c2w_enc = enc_out.pred_context_pose["extrinsic"][0].float().cpu().numpy()
        ctx_intr_enc = enc_out.pred_context_pose["intrinsic"][0].float().cpu().numpy()

        all_c2w, all_intr = self.joint_poses(x_all)
        s, r, t = sim3_from_cams(all_c2w[:nc], ctx_c2w_enc)
        aligned = apply_sim3(all_c2w, s, r, t)
        base = np.linalg.norm(ctx_c2w_enc[-1, :3, 3] - ctx_c2w_enc[0, :3, 3]) + 1e-12
        center_err = [float(np.linalg.norm(aligned[i, :3, 3] - ctx_c2w_enc[i, :3, 3]) / base) for i in range(nc)]
        rot_err = [rot_angle_deg(aligned[i, :3, :3], ctx_c2w_enc[i, :3, :3]) for i in range(nc)]

        torch.cuda.synchronize()
        t0 = time.time()
        tgt_rgb, tgt_alpha = self.render(gaussians, aligned[nc:], all_intr[nc:], size)
        torch.cuda.synchronize()
        t_render = (time.time() - t0) / max(len(tgt_imgs), 1)
        ctx_rgb, _ = self.render(gaussians, ctx_c2w_enc, ctx_intr_enc, size)
        ctx_psnr = []
        for i in range(nc):
            mse = np.mean((ctx_rgb[i] - ctx_imgs[i].astype(np.float32) / 255.0) ** 2)
            ctx_psnr.append(float(-10 * np.log10(max(mse, 1e-10))))

        return {
            "tgt_rgb": tgt_rgb,
            "tgt_alpha": tgt_alpha,
            "all_c2w_joint": all_c2w,
            "num_gaussians": int(gaussians.means.shape[1]),
            "time_encode_s": t_enc,
            "time_render_per_view_s": t_render,
            "sim3_scale": float(s),
            "ctx_center_err_rel_baseline": center_err,
            "ctx_rot_err_deg": rot_err,
            "ctx_self_psnr": ctx_psnr,
        }


def paste_and_warp(render, alpha, window, raw_shape, homography, aligned_shape):
    """Put an SxS crop render back into the raw frame, then warp into the aligned frame."""
    x0, y0, side = window
    h, w = raw_shape
    canvas = np.zeros((h, w, 3), np.float32)
    acanvas = np.zeros((h, w), np.float32)
    canvas[y0:y0 + side, x0:x0 + side] = cv2.resize(render, (side, side), interpolation=cv2.INTER_CUBIC)
    acanvas[y0:y0 + side, x0:x0 + side] = cv2.resize(alpha, (side, side), interpolation=cv2.INTER_LINEAR)
    win = np.zeros((h, w), np.uint8)
    win[y0:y0 + side, x0:x0 + side] = 1
    ha, wa = aligned_shape
    rgb = cv2.warpPerspective(canvas, homography, (wa, ha), flags=cv2.INTER_LINEAR)
    a = cv2.warpPerspective(acanvas, homography, (wa, ha), flags=cv2.INTER_LINEAR)
    valid = cv2.warpPerspective(win, homography, (wa, ha), flags=cv2.INTER_NEAREST) > 0
    return np.clip(rgb, 0, 1), a, valid
