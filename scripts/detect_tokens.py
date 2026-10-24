#!/usr/bin/env python3
"""Step 1a: mark unreliable regions (tokens) on the real input cameras, without ground truth.

Temporary detector until experiment 1 settles, for each adjacent context pair (a, b):
  * source-to-source flow F_a->b, F_b->a from OpenCV DIS (coarse-to-fine). RIFE cannot provide it:
    at t=0 its fusion mask is 1, so F_t->1 is unsupervised there and comes out far too short.
  * RIFE at t=0.5 (the synthesis network itself); its virtual-view failure signals |W0-W1| and the
    bidirectional collinearity residual (AUC ~0.9 in rounds 1-2) are carried back onto both source
    images along RIFE's own correspondences m + F_m->0 and m + F_m->1.
On every context camera:
  fail   DIS forward-backward inconsistency |F_ab(x) + F_ba(x + F_ab(x))| > fb_thresh (occluded
         pixels have no correspondence, so the check fails there too), or photometric residual
         |I_a - I_b(x + F_ab)| > photo_thresh, or a RIFE failure pixel maps here
  depth  horizontal gradient of the horizontal DIS flow (disparity) > depth_thresh
Union over both neighbours, restricted to the dilated foreground. Two masks are kept per camera:
  core   tokens dilated by core_dilate_px              -> region pasted back from GS
  recon  core dilated by recon_margin_px               -> region whose Gaussians are kept
All masks are computed in the aligned (flow) domain and also mapped to the raw frame (GS domain).
Outputs work/tokens/<setup>/cam{k}_{fail,depth,core,recon}{,_raw}.png and tokens.json
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lgs.common import (cam_path, chroma_fg, dilate, imread_rgb, imwrite, load_cfg, read_mask,  # noqa: E402
                        save_json, setup_cams, write_mask)
from lgs.rife import RIFE  # noqa: E402
from lgs.risk import signals  # noqa: E402


def sample(img, flow):
    """img(x + flow(x)) with bilinear sampling; border replicated."""
    h, w = flow.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, xs + flow[..., 0], ys + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def dis_flow(img_a, img_b, preset):
    """Dense DIS optical flow a -> b on a's grid (uint8 RGB inputs)."""
    p = {"ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST, "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
         "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM}[preset]
    dis = cv2.DISOpticalFlow_create(p)
    ga, gb = (cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in (img_a, img_b))
    return dis.calc(ga, gb, None)


def splat(mask_mid, flow, shape):
    """Mark source pixels m + flow(m) for every flagged virtual-view pixel m."""
    ys, xs = np.nonzero(mask_mid)
    u = np.round(xs + flow[ys, xs, 0]).astype(int)
    v = np.round(ys + flow[ys, xs, 1]).astype(int)
    ok = (u >= 0) & (u < shape[1]) & (v >= 0) & (v < shape[0])
    out = np.zeros(shape, np.uint8)
    out[v[ok], u[ok]] = 1
    return cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0


def pair_occlusion(img_a, img_b, f_ab, f_ba, tcfg):
    """Occlusion / match failure on a's grid: the DIS correspondence is inconsistent forward-backward
    (relative to its length) AND the colours disagree. Requiring both keeps textureless areas (hair,
    plain shirt), where DIS is ambiguous but the colours still match, out of the token set."""
    fb = np.linalg.norm(f_ab + sample(f_ba, f_ab), axis=-1)
    photo = cv2.GaussianBlur(np.abs(img_a - sample(img_b, f_ab)).mean(-1), (0, 0), 1.5)
    rel = tcfg["fb_thresh"] + tcfg["fb_rel"] * np.linalg.norm(f_ab, axis=-1)
    return (fb > rel) & (photo > tcfg["photo_thresh"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--setups", nargs="+", default=None)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    tcfg, ck = cfg["tokens"], cfg["chroma"]
    work = Path(cfg["work_dir"])
    hs = np.load(work / "prepare" / "H_raw2aligned.npy")
    rife = RIFE(cfg["rife_dir"])
    report = {}
    for setup in args.setups or list(cfg["setups"]):
        ctx, _ = setup_cams(cfg, setup)
        out = work / "tokens" / setup
        u8 = {k: imread_rgb(cam_path(cfg, "aligned", k)) for k in ctx}
        img = {k: u8[k].astype(np.float32) / 255.0 for k in ctx}
        raw_shape = imread_rgb(cam_path(cfg, "raw", ctx[0])).shape[:2]
        shape = u8[ctx[0]].shape[:2]
        occl = {k: np.zeros(shape, bool) for k in ctx}
        rife_fail = {k: np.zeros(shape, bool) for k in ctx}
        depth_src = {k: np.zeros(shape, bool) for k in ctx}
        rcfg = cfg["risk"]
        for a, b in zip(ctx[:-1], ctx[1:]):
            f_ab = dis_flow(u8[a], u8[b], tcfg["dis_preset"])
            f_ba = dis_flow(u8[b], u8[a], tcfg["dis_preset"])
            occl[a] |= pair_occlusion(img[a], img[b], f_ab, f_ba, tcfg)
            occl[b] |= pair_occlusion(img[b], img[a], f_ba, f_ab, tcfg)
            mid = rife.synth(u8[a], u8[b], 0.5, tcfg["rife_scale"])
            s = signals(mid, 0.5)
            near_mid = dilate(chroma_fg(mid["pred"], ck), tcfg["fg_dilate_px"])
            bad = ((s["r_warp"] > rcfg["warp_thresh"]) | (s["r_col"] > rcfg["col_thresh"])) & near_mid
            edge = (s["g_disp"] > tcfg["depth_thresh"]) & near_mid  # depth jump in RIFE's own disparity
            for src, ch in ((a, slice(0, 2)), (b, slice(2, 4))):
                rife_fail[src] |= splat(bad, mid["flow"][..., ch], shape)
                depth_src[src] |= splat(edge, mid["flow"][..., ch], shape)
            print(f"[{setup}] pair {a}-{b}: median |F_ab| (DIS) = {np.median(np.linalg.norm(f_ab, axis=-1)):.1f}px, "
                  f"RIFE t=0.5 failure = {bad.mean():.3f}, depth edge = {edge.mean():.3f} of frame")
        report[setup] = {}
        for k in ctx:
            fg = read_mask(work / "prepare" / f"fg_aligned_{k:02d}.png")
            near = dilate(fg | chroma_fg(u8[k], ck), tcfg["fg_dilate_px"])
            depth = depth_src[k].copy()
            fail = occl[k] | rife_fail[k]
            fail &= near
            depth &= near
            core = dilate(fail | depth, tcfg["core_dilate_px"]) & near
            recon = dilate(core, tcfg["recon_margin_px"])
            hinv = np.linalg.inv(hs[k])
            masks = {"fail": fail, "depth": depth, "core": core, "recon": recon}
            for name, m in masks.items():
                write_mask(out / f"cam{k}_{name}.png", m)
                raw = cv2.warpPerspective(m.astype(np.uint8), hinv, raw_shape[::-1], flags=cv2.INTER_NEAREST) > 0
                write_mask(out / f"cam{k}_{name}_raw.png", raw)
            ov = img[k].copy()
            ov[recon] = 0.7 * ov[recon] + 0.3 * np.array([0, 0.4, 1.0], np.float32)
            ov[fail] = 0.4 * ov[fail] + 0.6 * np.array([1, 0, 0], np.float32)
            ov[depth & ~fail] = 0.4 * ov[depth & ~fail] + 0.6 * np.array([1, 1, 0], np.float32)
            imwrite(out / f"cam{k}_overlay.png", ov[::2, ::2])
            nfg = max(fg.sum(), 1)
            report[setup][k] = {name: float((m & fg).sum() / nfg) for name, m in masks.items()}
            report[setup][k]["fail_and_depth"] = float((fail & depth & fg).sum() / nfg)
            report[setup][k]["occlusion_dis"] = float((occl[k] & near & fg).sum() / nfg)
            report[setup][k]["rife_fail"] = float((rife_fail[k] & near & fg).sum() / nfg)
            print(f"[{setup}] cam{k}: occl(DIS)={report[setup][k]['occlusion_dis']:.3f} rife_fail={report[setup][k]['rife_fail']:.3f} fail={report[setup][k]['fail']:.3f} depth={report[setup][k]['depth']:.3f} "
                  f"both={report[setup][k]['fail_and_depth']:.3f} core={report[setup][k]['core']:.3f} "
                  f"recon={report[setup][k]['recon']:.3f} (fraction of foreground)")
    save_json(work / "tokens" / "tokens.json", report)


if __name__ == "__main__":
    main()
