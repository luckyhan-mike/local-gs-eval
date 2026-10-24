#!/usr/bin/env python3
"""Offline rig calibration: chessboard corners (board shot) + scene feature matches (scene shot), one BA.

The rig is rigid and was not moved between the board shot and the scene shot, so each camera has a
single pose for both. World frame = chessboard plane (z = 0, unit = one square), which also fixes
the gauge. Unknowns: per-camera rotation/translation, focal (shared or per camera), shared radial
distortion (k1, k2); principal point fixed at the image center; scene points triangulated.
Stage 1 = OpenCV calibrateCamera on the board alone (all cameras share intrinsics); stage 2 = BA.
Outputs work/calib/rig.json and work/poses/<setup>_calib.json (t from calibrated camera centers).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lgs.common import (cam_path, imread_rgb, load_cfg, neighbors, save_json,  # noqa: E402
                        setup_cams)


def detect_board(gray, inner):
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    ok, c = cv2.findChessboardCornersSB(gray, tuple(inner), flags)
    if not ok:
        return None
    c = c.reshape(-1, 2).astype(np.float64)
    if c[0, 1] > c[-1, 1]:  # odd x even patterns can come back rotated by 180 degrees
        c = c[::-1].copy()
    return c


def scene_matches(grays, pairs, max_per_pair):
    sift = cv2.SIFT_create(nfeatures=20000)
    feats = [sift.detectAndCompute(g, None) for g in grays]
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    out = []
    for i, j in pairs:
        (ki, di), (kj, dj) = feats[i], feats[j]
        good = [m for m, n in (p for p in flann.knnMatch(di, dj, k=2) if len(p) == 2) if m.distance < 0.7 * n.distance]
        if len(good) < 30:
            continue
        pi = np.float64([ki[m.queryIdx].pt for m in good])
        pj = np.float64([kj[m.trainIdx].pt for m in good])
        _, inl = cv2.findFundamentalMat(pi, pj, cv2.FM_RANSAC, 1.0, 0.999)
        if inl is None:
            continue
        inl = inl.ravel() > 0
        pi, pj = pi[inl], pj[inl]
        if len(pi) > max_per_pair:
            sel = np.random.default_rng(i * 100 + j).choice(len(pi), max_per_pair, replace=False)
            pi, pj = pi[sel], pj[sel]
        out.append((i, j, pi, pj))
    return out


def project(points, rvec, tvec, f, cx, cy, k1, k2):
    pc = Rotation.from_rotvec(rvec).apply(points) + tvec
    x, y = pc[:, 0] / pc[:, 2], pc[:, 1] / pc[:, 2]
    r2 = x * x + y * y
    d = 1 + k1 * r2 + k2 * r2 * r2
    return np.stack([f * x * d + cx, f * y * d + cy], 1), pc[:, 2]


def triangulate(K, dist, rv, tv, i, j, pi, pj):
    ui = cv2.undistortPoints(pi[:, None], K, dist).reshape(-1, 2)
    uj = cv2.undistortPoints(pj[:, None], K, dist).reshape(-1, 2)
    Pi = np.hstack([cv2.Rodrigues(rv[i])[0], tv[i][:, None]])
    Pj = np.hstack([cv2.Rodrigues(rv[j])[0], tv[j][:, None]])
    X = cv2.triangulatePoints(Pi, Pj, ui.T, uj.T)
    return (X[:3] / X[3]).T


def bundle_adjust(board_obs, objp, scene, K0, dist0, rv0, tv0, size, per_cam_focal, f_scale):
    n = len(rv0)
    w, h = size
    cx, cy = (w - 1) / 2, (h - 1) / 2
    nf = n if per_cam_focal else 1
    pts0 = np.concatenate([s[4] for s in scene]) if scene else np.zeros((0, 3))
    npts = len(pts0)
    x0 = np.concatenate([np.hstack([rv0, tv0]).ravel(), np.full(nf, K0[0, 0]), dist0[:2], pts0.ravel()])
    obs = []  # (cam, point index or -1-board_index, u, v)
    for k, c in board_obs.items():
        for b, (u, v) in enumerate(c):
            obs.append((k, -1 - b, u, v))
    off = 0
    for i, j, pi, pj, _ in scene:
        for q in range(len(pi)):
            obs.append((i, off + q, *pi[q]))
            obs.append((j, off + q, *pj[q]))
        off += len(pi)
    obs = np.array(obs, dtype=np.float64)
    cams, pidx, uv = obs[:, 0].astype(int), obs[:, 1].astype(int), obs[:, 2:]
    is_board = pidx < 0

    def unpack(x):
        pose = x[:6 * n].reshape(n, 6)
        f = x[6 * n:6 * n + nf]
        k1, k2 = x[6 * n + nf:6 * n + nf + 2]
        pts = x[6 * n + nf + 2:].reshape(-1, 3)
        return pose, f, k1, k2, pts

    def residuals(x):
        pose, f, k1, k2, pts = unpack(x)
        res = np.zeros((len(obs), 2))
        for k in range(n):
            sel = cams == k
            if not sel.any():
                continue
            p3 = np.where(is_board[sel, None], objp[np.clip(-1 - pidx[sel], 0, None)],
                          pts[np.clip(pidx[sel], 0, None)] if npts else 0)
            proj, _ = project(p3, pose[k, :3], pose[k, 3:], f[k if per_cam_focal else 0], cx, cy, k1, k2)
            res[sel] = proj - uv[sel]
        return res.ravel()

    m = len(obs) * 2
    A = lil_matrix((m, len(x0)), dtype=np.uint8)
    rows = np.arange(len(obs))
    for d in range(2):
        r = 2 * rows + d
        for q in range(6):
            A[r, 6 * cams + q] = 1
        A[r, 6 * n + (cams if per_cam_focal else 0)] = 1
        A[r, 6 * n + nf] = 1
        A[r, 6 * n + nf + 1] = 1
        sp = ~is_board
        for q in range(3):
            A[r[sp], 6 * n + nf + 2 + 3 * pidx[sp] + q] = 1
    # Explicit parameter scales: focal ~1e3 px next to rotations ~1e-2 rad otherwise trips the relative
    # x-tolerance after a few iterations. Linear loss first to converge, then Huber against outliers.
    scale = np.concatenate([np.tile([1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2], n), np.full(nf, 1.0),
                            [1e-3, 1e-3], np.full(3 * npts, 1e-2)])
    x = x0
    for loss in ("linear", "huber"):
        sol = least_squares(residuals, x, jac_sparsity=A, method="trf", x_scale=scale, loss=loss,
                            f_scale=f_scale, xtol=1e-12, ftol=1e-12, gtol=1e-10, max_nfev=600, verbose=0)
        x = sol.x
        print(f"  BA ({loss}): nfev={sol.nfev} status={sol.status} "
              f"median reproj={np.median(np.linalg.norm(sol.fun.reshape(-1, 2), axis=1)):.4f}px")
    pose, f, k1, k2, _ = unpack(sol.x)
    res = sol.fun.reshape(-1, 2)
    err = np.linalg.norm(res, axis=1)
    return pose, f, (k1, k2), err, is_board, cams


def c2w_from(rvec, tvec):
    r = cv2.Rodrigues(np.asarray(rvec, np.float64))[0]
    c2w = np.eye(4)
    c2w[:3, :3] = r.T
    c2w[:3, 3] = -r.T @ np.asarray(tvec, np.float64).ravel()
    return c2w


def rig_t(c2w, cfg):
    """t of each held-out camera = projection of its center on the line between its context neighbours."""
    out = {}
    for setup in cfg["setups"]:
        ctx, targets = setup_cams(cfg, setup)
        t = {}
        for k in targets:
            a, b = neighbors(ctx, k)
            ca, cb, ck = c2w[a][:3, 3], c2w[b][:3, 3], c2w[k][:3, 3]
            d = cb - ca
            t[k] = float((ck - ca) @ d / (d @ d))
        out[setup] = {"context": ctx, "targets": targets, "t_pose": t,
                      "t_uniform": {k: (k - neighbors(ctx, k)[0]) / (neighbors(ctx, k)[1] - neighbors(ctx, k)[0])
                                    for k in targets}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    ccfg = cfg["calib"]
    work = Path(cfg["work_dir"])
    n = cfg["num_cams"]
    inner = ccfg["board_inner"]
    objp = np.zeros((inner[0] * inner[1], 3))
    objp[:, :2] = np.mgrid[0:inner[0], 0:inner[1]].T.reshape(-1, 2) * ccfg["square_size"]

    board = {}
    for k in range(n):
        g = cv2.cvtColor(imread_rgb(str(Path(cfg["data_root"]) / ccfg["board_pattern"].format(k))), cv2.COLOR_RGB2GRAY)
        c = detect_board(g, inner)
        if c is None:
            raise RuntimeError(f"cam {k}: 未检测到 {inner} 内角点棋盘格")
        board[k] = c
        size = g.shape[::-1]
    print(f"board corners found in all {n} cameras ({len(objp)} each)")

    # stage 1: all cameras share one intrinsic matrix, principal point at the image center
    w, h = size
    K0 = np.array([[1.2 * w, 0, (w - 1) / 2], [0, 1.2 * w, (h - 1) / 2], [0, 0, 1]])
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO |
             cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3)
    rms, K1, dist1, rvecs, tvecs = cv2.calibrateCamera(
        [objp.astype(np.float32)] * n, [board[k].astype(np.float32) for k in range(n)], size, K0, None, flags=flags)
    rv1 = np.array([r.ravel() for r in rvecs])
    tv1 = np.array([t.ravel() for t in tvecs])
    print(f"stage 1 (board only, shared K): rms={rms:.3f}px  f={K1[0, 0]:.1f}  k1={dist1.ravel()[0]:.4f} "
          f"k2={dist1.ravel()[1]:.4f}")

    # stage 2: add scene matches, BA over everything
    scene_raw = [cv2.cvtColor(imread_rgb(cam_path(cfg, "raw", k)), cv2.COLOR_RGB2GRAY) for k in range(n)]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if j - i <= ccfg["pair_max_gap"]]
    matches = scene_matches(scene_raw, pairs, ccfg["max_matches_per_pair"])
    scene = []
    for i, j, pi, pj in matches:
        X = triangulate(K1, dist1, rv1, tv1, i, j, pi, pj)
        ok = np.ones(len(X), bool)
        for c, p in ((i, pi), (j, pj)):
            proj, z = project(X, rv1[c], tv1[c], K1[0, 0], K1[0, 2], K1[1, 2], *dist1.ravel()[:2])
            ok &= (z > 0) & (np.linalg.norm(proj - p, axis=1) < ccfg["triangulate_max_err_px"])
        if ok.sum() >= 20:
            scene.append((i, j, pi[ok], pj[ok], X[ok]))
    nobs = sum(len(s[2]) for s in scene)
    print(f"scene matches: {len(scene)} pairs, {nobs} two-view points")
    pose, f, (k1, k2), err, is_board, cams = bundle_adjust(
        board, objp, scene, K1, dist1.ravel(), rv1, tv1, size, ccfg["per_cam_focal"], ccfg["huber_px"])

    report = {"board_rms_px": float(np.sqrt(np.mean(err[is_board] ** 2))),
              "board_median_px": float(np.median(err[is_board])),
              "scene_median_px": float(np.median(err[~is_board])) if (~is_board).any() else None,
              "scene_p90_px": float(np.percentile(err[~is_board], 90)) if (~is_board).any() else None,
              "stage1_rms_px": float(rms), "stage1_f": float(K1[0, 0]),
              "per_cam_board_median_px": {k: float(np.median(err[is_board & (cams == k)])) for k in range(n)}}
    print(f"stage 2 (BA): board rms={report['board_rms_px']:.3f}px  scene median={report['scene_median_px']}"
          f"  p90={report['scene_p90_px']}  f={np.round(f, 1).tolist()}  k1={k1:.4f} k2={k2:.4f}")

    cams_out = {}
    for k in range(n):
        fk = float(f[k if ccfg["per_cam_focal"] else 0])
        K = [[fk, 0, (w - 1) / 2], [0, fk, (h - 1) / 2], [0, 0, 1]]
        cams_out[k] = {"K": K, "dist": [float(k1), float(k2), 0, 0, 0],
                       "c2w": c2w_from(pose[k, :3], pose[k, 3:]).tolist(),
                       "stage1": {"K": K1.tolist(), "dist": dist1.ravel().tolist(),
                                  "c2w": c2w_from(rv1[k], tv1[k]).tolist()}}
    c2w = {k: np.array(cams_out[k]["c2w"]) for k in range(n)}
    centers = np.array([c2w[k][:3, 3] for k in range(n)])
    gaps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    report["adjacent_center_gaps_squares"] = gaps.tolist()
    print("adjacent camera spacing (board squares):", np.round(gaps, 3).tolist())
    save_json(work / "calib" / "rig.json", {"image_size": [w, h], "world": "chessboard, unit = 1 square",
                                            "cameras": cams_out, "report": report})
    for setup, d in rig_t(c2w, cfg).items():
        save_json(work / "poses" / f"{setup}_calib.json", d)
        print(f"[{setup}] t_pose (calibrated):", {k: round(v, 3) for k, v in d["t_pose"].items()})


if __name__ == "__main__":
    main()
