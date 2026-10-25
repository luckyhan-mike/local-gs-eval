import ast
import unittest

import numpy as np

from lgs.anysplat_wrap import apply_sim3, sim3_from_cams
from lgs.compat import Annotations
from lgs.metrics import auc


class NumericsTests(unittest.TestCase):
    def test_auc_ties_and_order(self):
        labels = np.array([False, False, True, True])
        self.assertAlmostEqual(auc(np.ones(4), labels), 0.5)
        scores = np.array([0., 1., 1., 2.])
        self.assertAlmostEqual(auc(scores, labels), 0.875)
        order = np.array([3, 1, 0, 2])
        self.assertAlmostEqual(auc(scores[order], labels[order]), 0.875)
        self.assertAlmostEqual(auc(np.arange(4), labels), 1.)

    def test_sim3_known_transform(self):
        src = np.repeat(np.eye(4)[None], 3, axis=0)
        src[:, 0, 3] = [0, 1, 2]
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        dst = apply_sim3(src, 2.5, rotation, np.array([3., 4., 5.]))
        transform = sim3_from_cams(src, dst)
        np.testing.assert_allclose(apply_sim3(src, *transform), dst, atol=1e-10)

    def test_mip_filter3d_matches_formula(self):
        import torch
        from lgs.anysplat_wrap import mip_filter3d
        size, fx_n, d = 448, 1.2, 3.0
        means = torch.tensor([[0., 0., d]])
        cov = torch.eye(3)[None] * 1e-4
        op = torch.tensor([0.8])
        c2w = np.eye(4)[None]
        intr = np.array([[[fx_n, 0, 0.5], [0, fx_n, 0.5], [0, 0, 1]]])
        cov2, op2, std = mip_filter3d(means, cov, op, c2w, intr, size, var=0.2, compensate=True)
        s2 = 0.2 * (d / (fx_n * size)) ** 2
        np.testing.assert_allclose(cov2[0].numpy(), np.eye(3) * (1e-4 + s2), rtol=1e-5)
        np.testing.assert_allclose(op2.item(), 0.8 * ((1e-4 / (1e-4 + s2)) ** 1.5), rtol=1e-4)
        _, op3, _ = mip_filter3d(means, cov, op, c2w, intr, size, var=0.2)
        self.assertAlmostEqual(op3.item(), 0.8, places=6)  # default: no energy compensation

    def test_bundle_adjust_recovers_synthetic_rig(self):
        import cv2
        from scripts.calibrate_rig import bundle_adjust, project
        rng = np.random.default_rng(1)
        w, h, f = 1920, 1080, 1800.0
        objp = np.zeros((88, 3))
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2) - [5, 3.5]
        n = 5
        rv, tv = [], []
        for k in range(n):
            x = (k - 2) * 2.0
            rot = cv2.Rodrigues(np.array([0.0, np.arctan2(x, 30.0), 0.0]))[0]  # look at the board centre
            c = np.array([x, 0.5, -30.0])
            rv.append(cv2.Rodrigues(rot)[0].ravel())
            tv.append(-rot @ c)
        rv, tv = np.array(rv), np.array(tv)
        cx, cy = (w - 1) / 2, (h - 1) / 2
        k1, k2 = -0.08, 0.2
        board = {k: project(objp, rv[k], tv[k], f, cx, cy, k1, k2)[0] for k in range(n)}
        pts = rng.uniform([-6, -6, -3], [6, 6, 3], (300, 3))
        scene = []
        for i in range(n):
            for j in range(i + 1, n):
                sel = rng.choice(300, 60, replace=False)
                pi = project(pts[sel], rv[i], tv[i], f, cx, cy, k1, k2)[0]
                pj = project(pts[sel], rv[j], tv[j], f, cx, cy, k1, k2)[0]
                scene.append((i, j, pi, pj, pts[sel] + rng.normal(0, 0.05, (60, 3))))
        K0 = np.array([[f * 1.03, 0, cx], [0, f * 1.03, cy], [0, 0, 1]])
        pose, fs, (e1, e2), err, _, _ = bundle_adjust(board, objp, scene, K0, np.array([0.0, 0, 0, 0, 0]),
                                                      rv + rng.normal(0, 0.01, rv.shape),
                                                      tv + rng.normal(0, 0.2, tv.shape), (w, h), True, 1.5)
        self.assertLess(np.median(err), 1e-3)
        np.testing.assert_allclose(fs, f, rtol=1e-3)
        np.testing.assert_allclose(pose[:, 3:], tv, atol=1e-2)

    def test_distort_maps_match_opencv_projection(self):
        import cv2
        from lgs.anysplat_wrap import distort_maps
        K = np.array([[900.0, 0, 319.5], [0, 900.0, 239.5], [0, 0, 1]])
        dist = np.array([-0.1, 0.2, 0, 0, 0])
        und = np.zeros((480, 640), np.float32)
        und[100, 500] = 1.0  # single bright pixel in the pinhole image
        mx, my = distort_maps(K, dist, (480, 640))
        d = cv2.remap(und, mx, my, cv2.INTER_LINEAR)
        yy, xx = np.unravel_index(np.argmax(d), d.shape)
        ray = np.array([[(500 - 319.5) / 900, (100 - 239.5) / 900, 1.0]])
        exp = cv2.projectPoints(ray, np.zeros(3), np.zeros(3), K, dist)[0].ravel()
        self.assertLess(np.hypot(xx - exp[0], yy - exp[1]), 1.5)

    def test_token_keep_selects_rays_through_mask(self):
        import torch
        from lgs.anysplat_wrap import token_keep
        c2w = np.eye(4)[None]
        intr = np.array([[[1.0, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1]]])
        mask = np.zeros((100, 100), bool)
        mask[40:60, 40:60] = True
        means = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 9.0], [0.4, 0.0, 2.0], [0.0, 0.0, -1.0]])
        keep = token_keep(means, c2w, intr, [mask]).numpy()
        # both points on the central ray are kept (foreground and the background behind it)
        self.assertEqual(keep.tolist(), [True, True, False, False])

    def test_interp_pose_endpoints_and_midpoint(self):
        from scipy.spatial.transform import Rotation
        from lgs.anysplat_wrap import interp_pose
        a, b = np.eye(4), np.eye(4)
        b[:3, :3] = Rotation.from_euler("y", 20, degrees=True).as_matrix()
        b[:3, 3] = [2, 0, 0]
        ka, kb = np.eye(3), np.eye(3) * 2
        p0, k0 = interp_pose(a, b, ka, kb, 0.0)
        p1, _ = interp_pose(a, b, ka, kb, 1.0)
        pm, km = interp_pose(a, b, ka, kb, 0.5)
        np.testing.assert_allclose(p0, a, atol=1e-12)
        np.testing.assert_allclose(p1, b, atol=1e-12)
        self.assertAlmostEqual(Rotation.from_matrix(pm[:3, :3]).as_euler("xyz", degrees=True)[1], 10, places=6)
        np.testing.assert_allclose(pm[:3, 3], [1, 0, 0])
        np.testing.assert_allclose(km, np.eye(3) * 1.5)

    def _aligned_scene(self):
        import cv2
        from scipy.spatial.transform import Rotation
        from lgs.aligned import aligned_reference, project, ray_plane
        n, d = np.array([0.04, 0.01, 1.0]) / np.linalg.norm([0.04, 0.01, 1.0]), 30.0

        def cam(c, yaw, f):
            m = np.eye(4)
            m[:3, :3] = Rotation.from_euler("y", yaw, degrees=True).as_matrix()
            m[:3, 3] = c
            return np.array([[f, 0, 958.0], [0, f, 541.0], [0, 0, 1]]), m

        cams = {0: cam([-4, 0.1, 0.4], 9, 1790), 4: cam([0, 0, 0], 0, 1800), 8: cam([4, -0.1, 0.5], -8, 1820),
                2: cam([-2.1, 0.15, 0.3], 5.5, 1850)}
        S = np.array([[0.833, 0, 160], [0, 0.833, 90.0], [0, 0, 1]])
        K_al, c2w_al = aligned_reference(cams[4][0], cams[4][1], S)
        grid = np.stack(np.meshgrid(np.linspace(200, 1700, 10), np.linspace(150, 900, 8)), -1).reshape(-1, 2)
        hs = {}
        for k, (K, c2w) in cams.items():
            X = ray_plane(K, c2w, grid, n, d)
            hs[k] = cv2.findHomography(project(K, c2w, X)[0], project(K_al, c2w_al, X)[0], 0)[0]
        return cams, hs, n, d

    def test_aligned_domain_render_matches_real_camera_homography(self):
        from lgs.aligned import project
        from scripts.run_anysplat import aligned_setup, render_aligned
        cams, hs, _, _ = self._aligned_scene()
        win = (420, 0, 1080)
        Y = np.array([[0.4, 0.2, 24.0]])  # off the zero-parallax plane (z ~ 30)

        class MockRunner:
            def render_px(self, state, c2w, K, w, h, keep=None, depth=False, **kw):
                img = np.zeros((1, h, w, 3), np.float32)
                (u, v), = project(np.asarray(K[0]), np.asarray(c2w[0]), Y)[0]
                img[0, int(round(v)) - 1:int(round(v)) + 2, int(round(u)) - 1:int(round(u)) + 2] = 1
                return img, np.ones((1, h, w), np.float32), np.full((1, h, w), 29.0, np.float32)

        def crop_norm(K):
            x0, y0, side = win
            return np.array([[K[0, 0] / side, 0, (K[0, 2] - x0) / side], [0, K[1, 1] / side, (K[1, 2] - y0) / side],
                             [0, 0, 1]])

        ctx = [0, 4, 8]
        state = {"ctx_c2w": np.stack([cams[c][1] for c in ctx]), "ctx_intr": np.stack([crop_norm(cams[c][0]) for c in ctx]),
                 "size": 448, "tgt_c2w": np.stack([cams[2][1]])}
        al = aligned_setup(MockRunner(), state, ctx, [2], win, hs, {"aligned_ref": 4, "aligned_pad_px": 160},
                           {2: 0.5}, (1080, 1920))
        self.assertLess(al["plane_rms_px"], 1e-3)
        img, _, _ = render_aligned(MockRunner(), state, al, "aligned_vggt", None, {})[0]
        v, u = np.unravel_index(np.argmax(img[..., 0]), img.shape[:2])
        import cv2
        exp = cv2.perspectiveTransform(project(cams[2][0], cams[2][1], Y)[0][None], hs[2])[0, 0]
        self.assertLess(np.hypot(u - exp[0], v - exp[1]), 1.5)

    def test_local_keep_uses_only_the_targets_pair(self):
        import tempfile
        import torch
        from pathlib import Path
        from lgs.common import write_mask
        from scripts.run_anysplat import load_token_sets, local_keep
        ctx, targets = [0, 4, 8], [2, 6]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "rife"
            mark = np.zeros((40, 40), bool)
            mark[10:20, 10:20] = True
            for k in ctx:  # union: cam0 and cam4 marked; pair 0-4 marks both, pair 4-8 marks nothing
                for kind in ("core", "recon"):
                    write_mask(root / f"cam{k}_{kind}_raw.png", mark if k != 8 else np.zeros_like(mark))
            for (a, b), m in (((0, 4), mark), ((4, 8), np.zeros_like(mark))):
                for k in (a, b):
                    for kind in ("core", "recon"):
                        write_mask(root / f"pair{a}-{b}" / f"cam{k}_{kind}_raw.png", m)
            ts = load_token_sets(Path(d), ctx, {"variants": {"rife": {}}})["rife"]
        self.assertEqual(set(ts["pairs"]), {(0, 4), (4, 8)})

        class Gs:  # one Gaussian per context camera, seen only by that camera
            means = [torch.zeros(3, 3)]

        class MockRunner:
            def visible_keep(self, state, masks, tol, depth=None):
                return torch.tensor([m.any() for m in masks])

        kr, kc, counts = local_keep(MockRunner(), {"gaussians": Gs()}, ts, (0, 0, 40), 40, ctx, targets,
                                    {"keep_mode": "visible"}, None)
        np.testing.assert_array_equal(kr[0], [True, True, False])   # target 2: pair 0-4
        np.testing.assert_array_equal(kc[1], [False, False, False])  # target 6: pair 4-8 has no tokens
        self.assertEqual(counts["4-8"]["core"], 0)

    def test_reproject_textured_plane_and_occlusion(self):
        from lgs.aligned import ray_plane
        from lgs.reproject import grid_points, reproject_view, zbuffer_depth
        n, d = np.array([0, 0, 1.0]), 10.0

        def tex(X):  # smooth texture on the plane
            return np.stack([0.5 + 0.4 * np.sin(X[:, 0] * 1.3), 0.5 + 0.4 * np.cos(X[:, 1] * 0.9),
                             0.5 + 0.2 * np.sin(X[:, 0] + X[:, 1])], -1)

        K = np.array([[100.0, 0, 79.5], [0, 100.0, 59.5], [0, 0, 1]])
        cams = {}
        for name, cx in (("a", -1.0), ("b", 1.0), ("v", 0.0)):
            c2w = np.eye(4)
            c2w[0, 3] = cx
            cams[name] = c2w
        xs, ys = np.meshgrid(np.arange(160.0), np.arange(120.0))
        px = np.c_[xs.reshape(-1), ys.reshape(-1)]
        imgs = {k: tex(ray_plane(K, cams[k], px, n, d)).reshape(120, 160, 3).astype(np.float32) for k in "ab"}
        win, size = (20, 0, 120), 60  # square crop of the raw image, input grid 60 x 60
        Kc = np.array([[50.0, 0, (79.5 - 20) / 2], [0, 50.0, 59.5 / 2], [0, 0, 1]])
        dep = np.full((size, size), d)
        pts = np.concatenate([grid_points(Kc, cams[k], dep)[0] for k in "ab"])
        depth, valid = zbuffer_depth(pts, K, cams["v"], 160, 120, scale=0.5)
        self.assertGreater(valid[20:100, 40:120].mean(), 0.99)
        src = [dict(img=imgs[k], K=K, c2w=cams[k], depth=dep, window=win, weight=0.5) for k in "ab"]
        rgb, alpha, nvis = reproject_view(K, cams["v"], depth, valid, src, 0.05)
        exp = tex(ray_plane(K, cams["v"], px, n, d)).reshape(120, 160, 3)
        m = (alpha > 0) & valid
        m[:, :30] = m[:, 130:] = False
        self.assertGreater(m[20:100, 40:120].mean(), 0.99)
        self.assertLess(np.abs(rgb - exp)[m].mean(), 0.01)
        # an occluder in front of camera a's view of the point: only camera b is used there
        dep_a = dep.copy()
        dep_a[:, 25:35] = 5.0
        src[0]["depth"] = dep_a
        _, _, nvis2 = reproject_view(K, cams["v"], depth, valid, src, 0.05)
        self.assertTrue(((nvis2 == 1) & m).any())
        self.assertLess(((nvis2 == 1) & m).sum(), m.sum())

    def test_reproject_targets_matches_real_camera_on_plane(self):
        import cv2
        from lgs.aligned import crop_K_to_raw, ray_plane
        from scripts.run_anysplat import aligned_setup, reproject_targets
        cams, hs, n, d = self._aligned_scene()
        win, size = (420, 0, 1080), 96

        def tex(X):
            return np.stack([0.5 + 0.4 * np.sin(X[:, 0] * 0.9), 0.5 + 0.4 * np.cos(X[:, 1] * 0.7),
                             0.5 + 0.2 * np.sin(0.5 * (X[:, 0] - X[:, 1]))], -1)

        xs, ys = np.meshgrid(np.arange(1920.0), np.arange(1080.0))
        px = np.c_[xs.reshape(-1), ys.reshape(-1)]
        raws = {k: (np.clip(tex(ray_plane(cams[k][0], cams[k][1], px, n, d)), 0, 1).reshape(1080, 1920, 3) * 255
                    ).astype(np.uint8) for k in (0, 4, 2)}

        def crop_norm(K):
            x0, y0, side = win
            return np.array([[K[0, 0] / side, 0, (K[0, 2] - x0) / side], [0, K[1, 1] / side, (K[1, 2] - y0) / side],
                             [0, 0, 1]])

        ctx = [0, 4, 8]
        state = {"ctx_c2w": np.stack([cams[c][1] for c in ctx]), "ctx_intr": np.stack([crop_norm(cams[c][0]) for c in ctx]),
                 "size": size, "tgt_c2w": np.stack([cams[2][1]])}
        g = np.arange(size, dtype=np.float64)
        gx, gy = np.meshgrid(g, g)
        vd = []
        for c in ctx:  # plane depth on each input camera's crop grid (what VGGT would predict)
            Kr = crop_K_to_raw(state["ctx_intr"][ctx.index(c)], win)
            gp = np.c_[gx.reshape(-1) * win[2] / size + win[0], gy.reshape(-1) * win[2] / size + win[1]]
            X = ray_plane(Kr, cams[c][1], gp, n, d)
            vd.append(((X - cams[c][1][:3, 3]) @ cams[c][1][:3, :3])[:, 2].reshape(size, size))
        state["vggt_depth"] = np.stack(vd)

        class MockRunner:
            def render_px(self, state, c2w, K, w, h, keep=None, depth=False, **kw):
                return np.zeros((1, h, w, 3), np.float32), np.ones((1, h, w), np.float32), np.full((1, h, w), 29.0)

        al = aligned_setup(MockRunner(), state, ctx, [2], win, hs, {"aligned_ref": 4, "aligned_pad_px": 160},
                           {2: 0.5}, (1080, 1920))
        views, stats = reproject_targets(MockRunner(), state, al, "aligned_vggt", ctx, [2], {2: 0.5}, raws, win,
                                         "vggt", {"vis_tol": 0.05}, None)
        img, alpha, valid = views[0]
        exp = cv2.warpPerspective(raws[2].astype(np.float32) / 255, hs[2], (1920, 1080), flags=cv2.INTER_LINEAR)
        m = (alpha > 0.99) & valid
        m[:, :700] = m[:, 1300:] = False  # central part, away from the crop window's edges
        m[:150] = m[950:] = False
        self.assertGreater(m.sum(), 100000)
        self.assertLess(np.abs(img - exp)[m].mean(), 0.02)
        self.assertGreater(stats[0]["seen_by_both"], 0.3)

    def test_tto_factor_roundtrip_and_local_update(self):
        import torch
        from types import SimpleNamespace
        from lgs.tto import cov_factor, cov_from, optimize
        rng = np.random.default_rng(0)
        n = 6
        R = torch.linalg.qr(torch.as_tensor(rng.normal(size=(n, 3, 3)), dtype=torch.float32))[0]
        s = torch.as_tensor(rng.uniform(0.01, 0.1, size=(n, 3)), dtype=torch.float32)
        s[0, 2] = 0.0  # flat surfel, as AnySplat predicts many
        cov = R @ torch.diag_embed(s ** 2) @ R.transpose(-1, -2)
        back = cov_from(cov_factor(cov), torch.zeros(n, 3, 3))
        self.assertLess(float((back - cov).abs().max() / cov.abs().max()), 1e-4)

        g = SimpleNamespace(means=torch.zeros(1, n, 3), covariances=cov[None], opacities=torch.full((1, n), 0.5),
                            rotations=torch.zeros(1, n, 4), scales=s[None],
                            harmonics=torch.as_tensor(rng.uniform(0, 1, size=(1, n, 3, 1)), dtype=torch.float32))

        class MockRunner:  # differentiable stand-in for gsplat: opacity-weighted mean colour everywhere
            device = "cpu"
            model = SimpleNamespace(decoder=SimpleNamespace(background_color=torch.zeros(3)))

            def rasterization(self, means, quats, scales, op, feat, viewmats, Ks, w, h, covars=None, **kw):
                col = (op[:, None] * feat[:, 0, :]).sum(0) / op.sum() + 0 * means.sum() + 0 * covars.sum()
                return col.expand(len(viewmats), h, w, 3), None, None

        state = {"gaussians": g, "ctx_c2w": np.repeat(np.eye(4)[None], 2, 0), "ctx_intr": np.repeat(np.eye(3)[None], 2, 0)}
        state["ctx_c2w"][1, 0, 3] = 1.0
        target = np.full((2, 16, 16, 3), 0.2, np.float32)
        train = np.array([True, True, True, False, False, False])
        snaps, log = optimize(MockRunner(), state, target, 16, [5, 60], train, {"lr": {"dc": 0.05}, "ssim_weight": 0.0})
        self.assertEqual(sorted(snaps), [5, 60])
        self.assertLess(log[-1]["loss"], log[0]["loss"] * 0.5)
        h0, h60 = g.harmonics[0, :, :, 0], snaps[60].harmonics[0, :, :, 0]
        torch.testing.assert_close(h60[3:], h0[3:])          # frozen Gaussians untouched
        self.assertGreater(float((h60[:3] - h0[:3]).abs().max()), 1e-3)

    def test_target_poses_calib_identity_frame(self):
        from scripts.run_anysplat import target_poses
        c2w = np.repeat(np.eye(4)[None], 4, 0)
        c2w[:, 0, 3] = [0, 1, 2, 3]
        rig = {k: {"c2w": c2w[k], "K": np.array([[1000.0, 0, 959.5], [0, 1000.0, 539.5], [0, 0, 1]]),
                   "dist": np.zeros(5)} for k in range(4)}
        state = {"ctx_c2w": c2w[[0, 3]] * 1, "tgt_c2w": c2w[[1, 2]], "tgt_intr": np.repeat(np.eye(3)[None], 2, 0)}
        out = target_poses(state, [0, 3], [1, 2], (420, 0, 1080), rig, ["vggt", "calib", "calibK"])
        np.testing.assert_allclose(out["calib"][0], c2w[[1, 2]], atol=1e-9)
        np.testing.assert_allclose(out["calibK"][1][0], [[1000 / 1080, 0, 539.5 / 1080], [0, 1000 / 1080, 539.5 / 1080],
                                                         [0, 0, 1]], atol=1e-12)

    def test_ecc_registration_recovers_shift(self):
        import cv2
        from scripts.evaluate import register, warp_by
        rng = np.random.default_rng(0)
        gt = cv2.GaussianBlur(rng.random((240, 320, 3)).astype(np.float32), (0, 0), 3)
        gt = (gt - gt.min()) / (gt.max() - gt.min())
        shift = np.array([[1, 0, 3.0], [0, 1, -2.0], [0, 0, 1]])
        moved = cv2.warpPerspective(gt, shift, (320, 240), flags=cv2.INTER_LINEAR)
        mask = np.zeros((240, 320), bool)
        mask[30:210, 30:290] = True
        hmat, disp = register(moved, gt, mask)
        self.assertIsNotNone(hmat)
        self.assertAlmostEqual(disp, np.hypot(3, 2), delta=0.3)
        back = warp_by(moved, hmat)
        self.assertLess(np.abs(back - gt)[mask].mean(), 0.01)

    def test_python39_type_backport_preserves_bitwise(self):
        code = 'ViewSamplerCfg = int | str\nx: int | None = None\ny = 1 | 2\ndef f(a: int | str) -> int | None: return 1\n'
        tree = ast.fix_missing_locations(Annotations().visit(ast.parse(code)))
        scope = {}
        exec(compile(tree, '<compat-test>', 'exec'), scope)
        self.assertEqual(scope['y'], 3)
        self.assertEqual(scope['f'](1), 1)


if __name__ == '__main__':
    unittest.main()
