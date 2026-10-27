#!/usr/bin/env python3
"""Step 2b: pixel-wise Gaussians / stereo synthesis in the aligned space for the token region.

Round 15 (gs.pixgs.variants, lgs/pixgs.py make_set / render_view): for every token variant and every target k between
context cameras a and b, Gaussians from the recon pixels of a and b (tokens/<setup>/<variant>/pair{a}-{b}/cam*_recon.png,
aligned frame), moved by the stereo correspondences (gs.pixgs.disp) to the target's t (the same poses/<setup>.json t as
the flow branch). Output gs/<setup>/gs_tok<variant>_pix<size>_pix_al_local/ -> "<variant>_pix" in the summary.

Round 16 (gs.pixgs.runs, applied to every token in gs.pixgs.run_tokens): label <token>_<run name>, output
gs/<setup>/gs_tok<label><size>_<method>_al_local/
  method pix2  lgs/pixgs.py prep_side / make_set2 / render_view2: P3 row fill of LR-failed displacements, no stretch at
               depth edges, LR-ok weighted merge, one render per source (core alpha as an extra channel), holes stay
               holes (alpha < 0.5 -> evaluate.py keeps the flow pixels there)
  method p3h   lgs/stereo_synth.py: pregs_exp P3 "H" stereo forward synthesis, holes push-pull filled
  disp         correspondence source (disp_fs = FoundationStereo-L, disp_fss = FS-S, disp_flw = RIFE flow of t = 0.5)
Runs named in gs.pixgs.gate.runs also get a gated copy <label>_g: the same render, core alpha kept only where the render
differs from the flow result (lgs/pixgs.py gate_by_flow; no ground truth). Needs the flow step to have run.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lgs.common import cam_path, imread_rgb, imwrite, load_cfg, load_json, read_mask, save_json, setup_cams, neighbors, write_mask  # noqa: E402
from lgs.pixgs import (depth_sign, fill_background, gate_by_flow, load_disp, make_set, make_set2, prep_side,  # noqa: E402
                       render_view, render_view2)
from lgs import stereo_synth  # noqa: E402


def sync(device):
    if device == "cuda":
        import torch
        torch.cuda.synchronize()


def disp_root(spec, cfg, repo):
    d = Path(spec.format(shot=cfg.get("shot", "").replace("/", "_")))
    return d if d.is_absolute() else repo / d


def stereo_time(droot):
    dmeta = load_json(droot / "meta.json") if (droot / "meta.json").exists() else {}
    return float(sum(x.get("time_stereo_s", 0) for x in dmeta.values() if isinstance(x, dict))), dmeta.get("config")


def base_meta(n_total, t_render, stats, size, v, pcfg, backend, shape, droot, extra):
    t_enc, enc_note = stereo_time(droot)
    m = {"num_gaussians": n_total, "time_encode_s": t_enc,
         "time_encode_note": f"correspondences per pair, measured on the Windows RTX 3090 at export ({enc_note})",
         "ctx_self_psnr": [float("nan")], "ctx_rot_err_deg": [0.0], "ctx_center_err_rel_baseline": [0.0],
         "window": [0, 0, shape[1]], "size": size, "crop": "token", "token_variant": v, "local": True,
         "render": {"name": "pix", "res": "aligned", "mode": backend, "eps2d": pcfg.get("eps2d", 0.3), "filter3d": False},
         "render_res": None, "eps2d_used": float(pcfg.get("eps2d", 0.3)),
         "time_render_per_view_s": float(np.mean(t_render)), "pose_source": "aligned_int", "pose_diag": None,
         "tto_iters": 0, "pixgs": stats, "pixgs_cfg": pcfg}
    m.update(extra)
    return m


def write_view(out, k, rgb, alpha, acore, shape):
    imwrite(out / f"cam{k}.png", rgb)
    imwrite(out / f"cam{k}_alpha.png", alpha)
    imwrite(out / f"cam{k}_alpha_core.png", acore)
    write_mask(out / f"cam{k}_valid.png", np.ones(shape, bool))


def round15(cfg, work, setup, ctx, targets, t_of, img, fg, pcfg, size, device, repo):
    droot = disp_root(pcfg["disp"], cfg, repo)
    disp = {(a, b): load_disp(droot / f"pair{a}-{b}.npz", img[ctx[0]].shape[:2]) for a, b in zip(ctx[:-1], ctx[1:])}
    shape = img[ctx[0]].shape[:2]
    for v in pcfg.get("variants", []):
        tdir = work / "tokens" / setup / v
        out = work / "gs" / setup / f"gs_tok{v}_pix{size}_pix_al_local"
        if not tdir.is_dir():
            print(f"[{setup}] 没有 token 变体 {v}，跳过")
            continue
        if pcfg.get("skip_existing") and (out / "meta.json").exists():
            print(f"[{setup}] {out.name} 已存在（第十五轮结果），跳过")
            continue
        n_total, t_render, stats = 0, [], {}
        for k in targets:
            a, b = neighbors(ctx, k)
            t = float(t_of[k])
            d = disp[(a, b)]
            pdir = tdir / f"pair{a}-{b}"
            rec = {s: read_mask(pdir / f"cam{s}_recon.png") for s in (a, b)}
            core = {s: read_mask(pdir / f"cam{s}_core.png") for s in (a, b)}
            ga, gb = d["ga"], d["gb"]
            sign = depth_sign(ga[0], fg[a], np.isfinite(ga[0]))
            t0 = time.time()
            ga_f = fill_background(ga, rec[a], sign, pcfg.get("fill_px", 96))
            gb_f = fill_background(gb, rec[b], -sign, pcfg.get("fill_px", 96))   # key of b = -sign * g_b,x
            sig, mx = float(pcfg.get("sigma_px", 0.5)), float(pcfg.get("max_stretch_px", 3.0))
            full = [make_set(img[a], ga_f, rec[a], t, sign, sig, mx), make_set(img[b], gb_f, rec[b], 1 - t, -sign, sig, mx)]
            cores = [make_set(img[a], ga_f, core[a], t, sign, sig, mx), make_set(img[b], gb_f, core[b], 1 - t, -sign, sig, mx)]
            rgb, alpha, backend = render_view(full, t, shape, pcfg, device)
            _, acore, _ = render_view(cores, t, shape, pcfg, device, backend)
            t_render.append(time.time() - t0)
            n = sum(len(s[0]) for s in full)
            n_total += n
            stats[str(k)] = {"pair": f"{a}-{b}", "t": t, "gaussians": n,
                             "missing_a": float((rec[a] & ~np.isfinite(ga_f[0])).sum() / max(rec[a].sum(), 1)),
                             "missing_b": float((rec[b] & ~np.isfinite(gb_f[0])).sum() / max(rec[b].sum(), 1)),
                             "depth_sign": sign}
            write_view(out, k, rgb, alpha, acore, shape)
        save_json(out / "meta.json", base_meta(n_total, t_render, stats, size, v, pcfg, backend, shape, droot, {}))
        print(f"[{setup}] {out.name}: {n_total} Gaussians over {len(targets)} views, "
              f"{np.mean(t_render) * 1000:.0f} ms/view ({backend}); "
              + ", ".join(f"cam{k} t={s['t']:.2f} n={s['gaussians']} miss a/b={s['missing_a']:.2f}/{s['missing_b']:.2f}"
                          for k, s in stats.items()))


def round16(cfg, work, setup, ctx, targets, t_of, img, fg, pcfg, size, device, repo):
    shape = img[ctx[0]].shape[:2]
    gcfg = pcfg.get("gate") or {}
    gated = set(gcfg.get("runs", []))
    ref = cfg["eval"]["ref_flow"]
    flow = {}
    for run in pcfg.get("runs", []):
        droot = disp_root(run["disp"], cfg, repo)
        disp = {(a, b): load_disp(droot / f"pair{a}-{b}.npz", shape) for a, b in zip(ctx[:-1], ctx[1:])}
        for tok in pcfg.get("run_tokens", []):
            tdir = work / "tokens" / setup / tok
            if not tdir.is_dir():
                print(f"[{setup}] 没有 token 变体 {tok}，跳过")
                continue
            label, method = f"{tok}_{run['name']}", run["method"]
            outs = {label: work / "gs" / setup / f"gs_tok{label}{size}_{method}_al_local"}
            if run["name"] in gated:
                outs[f"{label}_g"] = work / "gs" / setup / f"gs_tok{label}_g{size}_{method}_al_local"
            n_total, t_render, t_prep, stats, backend = 0, [], [], {}, "torch"
            for k in targets:
                a, b = neighbors(ctx, k)
                t = float(t_of[k])
                d = disp[(a, b)]
                pdir = tdir / f"pair{a}-{b}"
                rec = {s: read_mask(pdir / f"cam{s}_recon.png") for s in (a, b)}
                core = {s: read_mask(pdir / f"cam{s}_core.png") for s in (a, b)}
                sign = depth_sign(d["ga"][0], fg[a], np.isfinite(d["ga"][0]))
                t0 = time.time()
                ga, oka = prep_side(d["ga"], sign)     # once per pair in deployment (not per view)
                gb, okb = prep_side(d["gb"], -sign)
                t_prep.append(time.time() - t0)
                sync(device); t0 = time.time()
                if method == "pix2":
                    sig, mx = float(pcfg.get("sigma_px", 0.5)), float(pcfg.get("max_stretch_px", 3.0))
                    e = float(pcfg.get("edge_px", 1.0))
                    sets = [make_set2(img[a], ga, oka, core[a], rec[a], t, sign, sig, mx, e),
                            make_set2(img[b], gb, okb, core[b], rec[b], 1 - t, -sign, sig, mx, e)]
                    rgb, alpha, acore, backend = render_view2(sets, t, shape, pcfg, device)
                    n = sum(len(s[0]) for s in sets)
                else:
                    P = stereo_synth.prepare(img[a], img[b], ga, gb, oka, okb, core[a], core[b], rec[a], rec[b], sign, device)
                    rgb, alpha, acore = stereo_synth.render(P, t)
                    backend, n = "splat", int(len(P["a"]["key4"]) // 4 + len(P["b"]["key4"]) // 4)
                sync(device); t_render.append(time.time() - t0)
                n_total += n
                stats[str(k)] = {"pair": f"{a}-{b}", "t": t, "gaussians": n, "depth_sign": sign,
                                 "lr_fail_a": float((rec[a] & ~oka).sum() / max(rec[a].sum(), 1)),
                                 "lr_fail_b": float((rec[b] & ~okb).sum() / max(rec[b].sum(), 1)),
                                 "missing_a": float((rec[a] & ~np.isfinite(ga[0])).sum() / max(rec[a].sum(), 1)),
                                 "missing_b": float((rec[b] & ~np.isfinite(gb[0])).sum() / max(rec[b].sum(), 1)),
                                 "core_px": int((acore > 0.5).sum()), "hole_in_core": float(((acore > 0.5) & (alpha < 0.5)).sum())}
                write_view(outs[label], k, rgb, alpha, acore, shape)
                if f"{label}_g" in outs:
                    if k not in flow:
                        flow[k] = imread_rgb(work / "flow" / setup / ref / f"cam{k}.png").astype(np.float32) / 255.0
                    ag = gate_by_flow(rgb, flow[k], acore, float(gcfg.get("thr", 0.05)), float(gcfg.get("blur_px", 3)),
                                      int(gcfg.get("dilate_px", 8)))
                    stats[str(k)]["gate_keep"] = float((ag > 0.5).sum() / max((acore > 0.5).sum(), 1))
                    write_view(outs[f"{label}_g"], k, rgb, alpha, ag, shape)
            for lab, out in outs.items():
                save_json(out / "meta.json", base_meta(
                    n_total, t_render, stats, size, tok, pcfg, backend, shape, droot,
                    {"round16": {"label": lab, "method": method, "disp": run["disp"], "gated": lab.endswith("_g"),
                                 "time_prep_per_pair_s": float(np.mean(t_prep)), "gate": gcfg if lab.endswith("_g") else None}}))
            print(f"[{setup}] {label} ({method}, {run['disp']}): {n_total} primitives over {len(targets)} views, "
                  f"{np.mean(t_render) * 1000:.0f} ms/view ({backend}), prep {np.mean(t_prep) * 1000:.0f} ms; "
                  + ", ".join(f"cam{k} t={s['t']:.2f} lr-fail a/b={s['lr_fail_a']:.2f}/{s['lr_fail_b']:.2f}"
                              + (f" gate={s['gate_keep']:.2f}" if "gate_keep" in s else "") for k, s in stats.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--setups", nargs="+", default=None)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    pcfg = cfg["gs"].get("pixgs") or {}
    if not pcfg.get("enabled"):
        print("gs.pixgs 未启用，跳过")
        return
    work = Path(cfg["work_dir"])
    repo = Path(__file__).resolve().parents[1]
    size = cfg["gs"]["sizes"][0]
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    for setup in args.setups or list(cfg["setups"]):
        ctx, targets = setup_cams(cfg, setup)
        t_of = {int(k): v for k, v in load_json(work / "poses" / f"{setup}.json")["t_pose"].items()}
        img = {k: imread_rgb(cam_path(cfg, "aligned", k)).astype(np.float32) / 255.0 for k in ctx}
        fg = {k: read_mask(work / "prepare" / f"fg_aligned_{k:02d}.png") for k in ctx}
        round15(cfg, work, setup, ctx, targets, t_of, img, fg, pcfg, size, device, repo)
        round16(cfg, work, setup, ctx, targets, t_of, img, fg, pcfg, size, device, repo)


if __name__ == "__main__":
    main()
