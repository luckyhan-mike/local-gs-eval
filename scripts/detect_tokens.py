#!/usr/bin/env python3
"""Step 1a: mark unreliable regions (tokens) on the real input cameras, without ground truth.

Temporary detector until experiment 1 settles. For each adjacent context pair (a, b), computed once:
  * RIFE at t=0.5 (the synthesis network itself): its virtual-view failure signals |W0-W1| and the
    bidirectional collinearity residual (AUC ~0.9 in rounds 1-2) and its disparity edges, carried
    back onto both source images along RIFE's own correspondences m + F_m->0 and m + F_m->1.
  * optional DIS source-to-source flow for an occlusion check (forward-backward AND photometric
    inconsistency). RIFE cannot provide source-to-source flow: at t=0 its fusion mask is 1, so
    F_t->1 is unsupervised there and comes out far too short.
Tokens are kept per context pair (round 7): a virtual view between a and b is synthesised from a and b
only, so it may only use the failures detected in that pair. In round 6 the union over all pairs let
cam4 tokens found by pair 0-4 replace a correct flow result between 4 and 8.
Every entry of `tokens.variants` recombines these signals with its own thresholds:
  fail   RIFE failure (r_warp > rife_warp or r_col > rife_col) [+ DIS occlusion if use_dis];
         connected components smaller than min_cc_px dropped
  depth  RIFE disparity jump > depth_thresh; depth: any | near_fail (within near_fail_px of a fail
         pixel, i.e. a depth jump where the flow actually breaks) | off
  core   (fail | depth) dilated by core_dilate_px      -> region pasted back from GS
  recon  core dilated by recon_margin_px                -> region whose Gaussians are kept
  external   (round 14) instead of the RIFE signals, a per-pair raw-frame mask computed elsewhere (e.g. the
         round-7 Cn_wide region from tools/export_cnw_tokens.py) is the fail map; core / recon as below
  pair_gate  if the pair's core covers less than this fraction of the foreground (mean over its two
         cameras) the pair does not trigger GS at all: its views stay pure flow (small baselines)
Round 5 showed GS only helps where the flow really fails (perfect detector: +4.4 dB fg on cam3) and
hurts where the flow is right, so the variants trade recall for precision.
Outputs work/tokens/<setup>/<variant>/pair{a}-{b}/cam{k}_{fail,depth,core,recon}{,_raw}.png (used for
paste-back), the union over pairs in work/tokens/<setup>/<variant>/ (overlays, encoding window), tokens.json
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


def drop_small(m, min_px):
    if min_px <= 0 or not m.any():
        return m
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_px
    return keep[lab]


def external_mask(cfg, pattern, a, b, k, h_raw2al, shape):
    """Round 14: a GS-region mask computed outside this repo (e.g. tools/export_cnw_tokens.py), stored per pair
    on the raw frame as <pattern>/pair{a}-{b}/cam{k}.png; pattern may contain {shot} (scene_shot of this work dir)
    and is relative to the repository root. Warped into the aligned frame like the raw images."""
    root = Path(pattern.format(shot=cfg.get("shot", "").replace("/", "_")))
    root = root if root.is_absolute() else Path(__file__).resolve().parents[1] / root
    m = read_mask(root / f"pair{a}-{b}" / f"cam{k}.png")
    return cv2.warpPerspective(m.astype(np.uint8), h_raw2al, shape[::-1], flags=cv2.INTER_NEAREST) > 0


def variant_params(tcfg, name):
    p = {"use_dis": False, "rife_warp": 0.20, "rife_col": 6.0, "depth": "near_fail", "near_fail_px": 12,
         "min_cc_px": 150, "depth_thresh": tcfg["depth_thresh"], "pair_gate": 0.0}
    p.update(tcfg["variants"][name] or {})
    return p


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
    variants = {v: variant_params(tcfg, v) for v in tcfg["variants"]}
    need_dis = any(p["use_dis"] for p in variants.values())
    report = {v: {} for v in variants}
    pair_report = {v: {} for v in variants}
    for setup in args.setups or list(cfg["setups"]):
        ctx, _ = setup_cams(cfg, setup)
        u8 = {k: imread_rgb(cam_path(cfg, "aligned", k)) for k in ctx}
        img = {k: u8[k].astype(np.float32) / 255.0 for k in ctx}
        raw_shape = imread_rgb(cam_path(cfg, "raw", ctx[0])).shape[:2]
        shape = u8[ctx[0]].shape[:2]
        mids = []  # per pair: (a, b, signals, flow, near-foreground mask, DIS occlusion {a, b})
        for a, b in zip(ctx[:-1], ctx[1:]):
            occl = {a: np.zeros(shape, bool), b: np.zeros(shape, bool)}
            if need_dis:
                f_ab = dis_flow(u8[a], u8[b], tcfg["dis_preset"])
                f_ba = dis_flow(u8[b], u8[a], tcfg["dis_preset"])
                occl[a] = pair_occlusion(img[a], img[b], f_ab, f_ba, tcfg)
                occl[b] = pair_occlusion(img[b], img[a], f_ba, f_ab, tcfg)
            mid = rife.synth(u8[a], u8[b], 0.5, tcfg["rife_scale"])
            s = signals(mid, 0.5)
            near_mid = dilate(chroma_fg(mid["pred"], ck), tcfg["fg_dilate_px"])
            mids.append((a, b, {n: s[n] for n in ("r_warp", "r_col", "g_disp")}, mid["flow"], near_mid, occl))
            print(f"[{setup}] pair {a}-{b}: RIFE t=0.5 |W0-W1|>0.2 on {((s['r_warp'] > 0.2) & near_mid).mean():.4f} "
                  f"of frame, median |F_t0| = {np.median(np.linalg.norm(mid['flow'][..., :2], axis=-1)[near_mid]):.1f}px")
        fgs = {k: read_mask(work / "prepare" / f"fg_aligned_{k:02d}.png") for k in ctx}
        near = {k: dilate(fgs[k] | chroma_fg(u8[k], ck), tcfg["fg_dilate_px"]) for k in ctx}

        def save(out, k, masks):
            hinv = np.linalg.inv(hs[k])
            for name, m in masks.items():
                write_mask(out / f"cam{k}_{name}.png", m)
                raw = cv2.warpPerspective(m.astype(np.uint8), hinv, raw_shape[::-1], flags=cv2.INTER_NEAREST) > 0
                write_mask(out / f"cam{k}_{name}_raw.png", raw)

        def frac(m, k):
            return float((m & fgs[k]).sum() / max(fgs[k].sum(), 1))

        for v, p in variants.items():
            out = work / "tokens" / setup / v
            union = {k: {n: np.zeros(shape, bool) for n in ("fail", "depth", "core", "recon")} for k in ctx}
            pairs = {}
            for a, b, s, flow, near_mid, occl in mids:
                bad = ((s["r_warp"] > p["rife_warp"]) | (s["r_col"] > p["rife_col"])) & near_mid
                edge = (s["g_disp"] > p["depth_thresh"]) & near_mid
                pm = {}
                for src, ch in ((a, slice(0, 2)), (b, slice(2, 4))):
                    if p.get("external"):  # round 14: region from another detector, used as given (no fg limit)
                        fail = external_mask(cfg, p["external"], a, b, src, hs[src], shape)
                        depth = np.zeros(shape, bool)
                        core = dilate(fail, tcfg["core_dilate_px"])
                        pm[src] = {"fail": fail, "depth": depth, "core": core,
                                   "recon": dilate(core, tcfg["recon_margin_px"])}
                        continue
                    fail = splat(bad, flow[..., ch], shape) | (occl[src] if p["use_dis"] else False)
                    fail = drop_small(fail & near[src], p["min_cc_px"])
                    depth = splat(edge, flow[..., ch], shape) & near[src]
                    if p["depth"] == "near_fail":
                        depth &= dilate(fail, p["near_fail_px"])
                    elif p["depth"] in ("off", False, None):  # YAML reads a bare off as False
                        depth[:] = False
                    core = dilate(fail | depth, tcfg["core_dilate_px"]) & near[src]
                    pm[src] = {"fail": fail, "depth": depth, "core": core,
                               "recon": dilate(core, tcfg["recon_margin_px"])}
                core_frac = float(np.mean([frac(pm[k]["core"], k) for k in (a, b)]))
                gated = core_frac < p["pair_gate"]
                pairs[f"{a}-{b}"] = {"core": core_frac, "gated": bool(gated),
                                     **{f"core_cam{k}": frac(pm[k]["core"], k) for k in (a, b)}}
                for k in (a, b):
                    if gated:
                        pm[k] = {n: np.zeros(shape, bool) for n in pm[k]}
                    save(out / f"pair{a}-{b}", k, pm[k])
                    for n in union[k]:
                        union[k][n] |= pm[k][n]
                print(f"[{setup}/{v}] pair {a}-{b}: core={core_frac:.3f} of foreground"
                      + (f" < gate {p['pair_gate']}: pure flow" if gated else ""))
            report[v][setup] = {}
            pair_report[v][setup] = pairs
            for k in ctx:
                masks = union[k]
                save(out, k, masks)
                fail, depth, recon = masks["fail"], masks["depth"], masks["recon"]
                ov = img[k].copy()
                ov[recon] = 0.7 * ov[recon] + 0.3 * np.array([0, 0.4, 1.0], np.float32)
                ov[fail] = 0.4 * ov[fail] + 0.6 * np.array([1, 0, 0], np.float32)
                ov[depth & ~fail] = 0.4 * ov[depth & ~fail] + 0.6 * np.array([1, 1, 0], np.float32)
                imwrite(out / f"cam{k}_overlay.png", ov[::2, ::2])
                r = {name: frac(m, k) for name, m in masks.items()}
                report[v][setup][k] = r
                print(f"[{setup}/{v}] cam{k} (union over pairs): fail={r['fail']:.3f} depth={r['depth']:.3f} "
                      f"core={r['core']:.3f} recon={r['recon']:.3f} (fraction of foreground)")
    save_json(work / "tokens" / "tokens.json", {"variants": variants, "fractions": report, "pairs": pair_report})


if __name__ == "__main__":
    main()
