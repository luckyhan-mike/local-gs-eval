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

        # round 11: cropped supervision renders a smaller window and logs checkpoint times
        state["gaussians"].means[0, :, 2] = 2.0
        state["ctx_intr"] = np.repeat(np.array([[[1.0, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1]]]), 2, 0)
        sizes = []

        class CropRunner(MockRunner):
            def rasterization(self, *a, **kw):
                sizes.append((a[7], a[8]))
                return super().rasterization(*a, **kw)
        snaps, log = optimize(CropRunner(), state, np.full((2, 64, 64, 3), 0.2, np.float32), 64, [3], train,
                              {"crop": {"enabled": True, "margin_px": 4}})
        self.assertLess(sizes[0][0] * sizes[0][1], 64 * 64)
        self.assertIn("box", log[0])
        self.assertIn("time_s", log[-1])

        # round 12: per-view boxes -> one rasterisation call per view and step, each with its own size
        state["ctx_c2w"][1, 0, 3] = 0.3
        g.means[0, :3, 0] = torch.tensor([-0.4, 0.0, 0.4])  # spread in x: per-view boxes differ in position
        sizes.clear()
        snaps, log = optimize(CropRunner(), state, np.full((2, 64, 64, 3), 0.2, np.float32), 64, [2], train,
                              {"crop": {"enabled": True, "per_view": True, "margin_px": 2}})
        self.assertEqual(len(sizes), 2 * 2)
        self.assertEqual(len(log[0]["box"]["per_view"]), 2)
        self.assertLess(log[0]["box"]["area_frac"], 1.0)
        self.assertEqual(len(log[-1]["ctx_psnr"]), 2)

    def test_tto_position_and_covariance_caps(self):
        import torch
        from types import SimpleNamespace
        from lgs.tto import optimize
        n = 5
        rng = np.random.default_rng(1)
        s = torch.full((n, 3), 0.05)
        cov = torch.diag_embed(s ** 2)
        means = torch.as_tensor(rng.normal(size=(1, n, 3)), dtype=torch.float32)
        means[0, :, 2] = 3.0
        g = SimpleNamespace(means=means, covariances=cov[None], opacities=torch.full((1, n), 0.5),
                            rotations=torch.zeros(1, n, 4), scales=s[None],
                            harmonics=torch.full((1, n, 3, 1), 0.5))

        class PullRunner:  # loss rewards moving every Gaussian far along +x and growing its covariance
            device = "cpu"
            model = SimpleNamespace(decoder=SimpleNamespace(background_color=torch.zeros(3)))

            def rasterization(self, means, quats, scales, op, feat, viewmats, Ks, w, h, covars=None, **kw):
                v = torch.sigmoid(-means[:, 0].mean() - covars.diagonal(dim1=-2, dim2=-1).sum(-1).mean() * 100)
                return v.expand(len(viewmats), h, w, 3), None, None

        state = {"gaussians": g, "ctx_c2w": np.repeat(np.eye(4)[None], 2, 0), "ctx_intr": np.repeat(np.eye(3)[None], 2, 0)}
        state["ctx_c2w"][1, 0, 3] = 1.0
        train = np.ones(n, bool)
        snaps, _ = optimize(PullRunner(), state, np.zeros((2, 8, 8, 3), np.float32), 8, [40], train,
                            {"lr": {"means": 1.0, "cov": 0.5}, "ssim_weight": 0.0,
                             "max_disp_scale": 1.0, "max_cov_scale": 1.5})
        size = float(np.sqrt(3 * 0.05 ** 2))
        disp = (snaps[40].means[0] - means[0]).norm(dim=1)
        self.assertLessEqual(float(disp.max()), size * 1.001)
        self.assertGreater(float(disp.min()), size * 0.5)          # the pull was strong enough to hit the cap
        ratio = snaps[40].covariances[0].diagonal(dim1=-2, dim2=-1) / cov.diagonal(dim1=-2, dim2=-1)
        self.assertLessEqual(float(ratio.max()), 1.5 ** 2 * 1.001)

    def test_tto_crop_box_is_exact_subwindow(self):
        import torch
        from lgs.tto import crop_boxes
        pts = torch.tensor([[-0.2, -0.1, 2.0], [0.3, 0.2, 2.0], [0.0, 0.0, 4.0]])
        vm = torch.eye(4)[None].repeat(2, 1, 1)
        vm[1, 0, 3] = 0.25  # second camera shifted: same box size, different position
        K = torch.tensor([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]])[None].repeat(2, 1, 1)
        w, h, xy = crop_boxes(pts, vm, K, 100, margin=2, q=0.0)
        uv = 100 * pts[:, :2] / pts[:, 2:] + 50
        self.assertGreaterEqual(w, int(uv[:, 0].max() - uv[:, 0].min()) + 4)
        self.assertGreaterEqual(h, int(uv[:, 1].max() - uv[:, 1].min()) + 4)
        for (x, y), shift in zip(xy, (0.0, 0.25)):
            u = 100 * (pts[:, 0] + shift) / pts[:, 2] + 50
            self.assertTrue(bool(((u >= x) & (u < x + w)).all()) and bool(((uv[:, 1] >= y) & (uv[:, 1] < y + h)).all()))

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

    def test_dataset6_calib_t_and_wall_key(self):
        import cv2
        from lgs.common import chroma_fg
        from scripts.run_scenes import calib_t
        centers = [[0, 0, 0], [100, 2, 0], [250, -1, 5], [360, 0, 0], [470, 0, 0], [600, 0, 0]]
        t = calib_t({"rig": {"centers_mm": centers}}, [0, 3, 5], [1, 2, 4])
        self.assertAlmostEqual(t["1"], 100 / 360, places=3)
        self.assertAlmostEqual(t["2"], 250 / 360, places=3)
        self.assertAlmostEqual(t["4"], 110 / 240, places=3)
        # blue-grey wall (H~105, S~43, V~99) keyed out, a dark shirt and an orange pillow kept
        hsv = np.zeros((120, 160, 3), np.uint8)
        hsv[:] = (105, 43, 99)
        hsv[30:90, 20:70] = (0, 0, 25)
        hsv[30:90, 90:140] = (15, 170, 200)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        fg = chroma_fg(rgb, dict(h_lo=95, h_hi=130, s_min=10, s_max=110, v_min=50, min_area_frac=0.002))
        self.assertFalse(fg[5, 5])
        self.assertTrue(fg[60, 45] and fg[60, 115])

    def test_multi_window_grouping_partitions_tokens(self):
        from scripts.run_anysplat import group_windows, restrict_tokens
        h, w = 1296, 2304
        a, b = np.zeros((h, w), bool), np.zeros((h, w), bool)
        a[300:400, 300:420] = True        # face, cam a
        b[310:410, 380:500] = True        # same face shifted in cam b: overlaps -> one component
        a[600:640, 520:560] = True        # hand 200 px below: merged (square stays within 640)
        b[900:1000, 1900:2000] = True     # tripod far right: its own window
        groups = group_windows([a, b], (h, w), {"max_side": 640, "max_windows": 4, "pad_px": 16}, 448)
        self.assertEqual(len(groups), 2)
        for (x0, y0, side), region in groups:
            self.assertEqual(side, 448)   # min side: no window above 448 here, so no upsampling at all
            ys, xs = np.nonzero(region & (a | b))
            self.assertTrue(x0 <= xs.min() and xs.max() < x0 + side and y0 <= ys.min() and ys.max() < y0 + side)
        regions = [r for _, r in groups]
        self.assertFalse((regions[0] & regions[1]).any())
        np.testing.assert_array_equal(regions[0] | regions[1], a | b)
        one = group_windows([a, b], (h, w), {"max_side": 640, "max_windows": 1}, 448)
        self.assertEqual(len(one), 1)
        ts = {"core": [a, b], "recon": [a, b], "pairs": {(0, 3): {"core": [a, b], "recon": [a, b]}}}
        r = restrict_tokens(ts, regions[0])
        self.assertEqual(int(r["recon"][0].sum() + r["recon"][1].sum()), int((regions[0] & a).sum() + (regions[0] & b).sum()))
        self.assertTrue(np.array_equal(r["pairs"][(0, 3)]["core"][1], b & regions[0]))

    def test_multi_window_composite_and_external_tokens(self):
        import tempfile
        from pathlib import Path
        import cv2
        from lgs.common import imread_rgb, imwrite, load_json, save_json, write_mask
        from scripts.run_anysplat import composite_windows, read_gray
        from scripts.detect_tokens import external_mask
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            roots = [d / "gs_win" / f"v_w{i}" for i in range(2)]
            for i, r in enumerate(roots):
                out = r / "s" / "gs_tokv_mw448_x_local"
                img = np.zeros((20, 30, 3), np.uint8); img[:] = 100 * (i + 1)
                al = np.zeros((20, 30), np.float32); al[:, 15 * i:15 * i + 15] = 1.0
                imwrite(out / "cam1.png", img); imwrite(out / "cam1_alpha.png", al); imwrite(out / "cam1_alpha_core.png", al)
                write_mask(out / "cam1_valid.png", al > 0)
                save_json(out / "meta.json", {"window": [0, 0, 448 + 100 * i], "num_gaussians": 10, "time_encode_s": 0.3,
                                              "time_render_per_view_s": 0.01, "local_gaussians": {}, "render": {},
                                              "tto": {"train_gaussians": 5, "res": 448, "time_s": 1.0,
                                                      "log": [{"iter": 1, "loss": 0.1, "ctx_psnr": [20.0, 22.0]},
                                                              {"iter": 25, "loss": 0.05, "ctx_psnr": [24.0, 26.0],
                                                               "time_s": 0.5 + i}]}})
            composite_windows(d, "s", roots, [1])
            out = d / "gs" / "s" / "gs_tokv_mw448_x_local"
            img = imread_rgb(out / "cam1.png")
            self.assertEqual(int(img[5, 5, 0]), 100)
            self.assertEqual(int(img[5, 25, 0]), 200)
            self.assertTrue(np.allclose(read_gray(out / "cam1_alpha.png"), 1.0))
            m = load_json(out / "meta.json")
            self.assertEqual(m["window"][2], 548)
            self.assertEqual(len(m["windows"]), 2)
            self.assertAlmostEqual(m["time_encode_s"], 0.6)
            self.assertAlmostEqual([e for e in m["tto"]["log"] if e["iter"] == 25][0]["time_s"], 2.0)
            # external token mask: raw frame, per pair, path pattern with {shot}, warped by the raw->aligned H
            mk = np.zeros((40, 60), bool); mk[10:20, 30:40] = True
            write_mask(d / "tok" / "changjing1_shot_0021" / "cnn" / "pair0-3" / "cam3.png", mk)
            H = np.array([[1, 0, -5], [0, 1, -2], [0, 0, 1]], np.float64)
            got = external_mask({"shot": "changjing1/shot_0021"}, str(d / "tok" / "{shot}" / "cnn"), 0, 3, 3, H, (40, 60))
            self.assertTrue(got[8:18, 25:35].all() and got.sum() == mk.sum())

    def test_tto_periodic_covariance_cap_holds_at_checkpoints(self):
        import torch
        from types import SimpleNamespace
        from lgs.tto import optimize
        n = 4
        s = torch.full((n, 3), 0.05)
        cov = torch.diag_embed(s ** 2)
        means = torch.zeros(1, n, 3); means[0, :, 2] = 3.0
        g = SimpleNamespace(means=means, covariances=cov[None], opacities=torch.full((1, n), 0.5),
                            rotations=torch.zeros(1, n, 4), scales=s[None], harmonics=torch.full((1, n, 3, 1), 0.5))

        class GrowRunner:
            device = "cpu"
            model = SimpleNamespace(decoder=SimpleNamespace(background_color=torch.zeros(3)))

            def rasterization(self, means, quats, scales, op, feat, viewmats, Ks, w, h, covars=None, **kw):
                v = torch.sigmoid(-covars.diagonal(dim1=-2, dim2=-1).sum(-1).mean() * 100)
                return v.expand(len(viewmats), h, w, 3), None, None

        state = {"gaussians": g, "ctx_c2w": np.repeat(np.eye(4)[None], 2, 0), "ctx_intr": np.repeat(np.eye(3)[None], 2, 0)}
        state["ctx_c2w"][1, 0, 3] = 1.0
        snaps, _ = optimize(GrowRunner(), state, np.zeros((2, 8, 8, 3), np.float32), 8, [7, 37], np.ones(n, bool),
                            {"lr": {"cov": 0.5}, "ssim_weight": 0.0, "max_cov_scale": 1.5, "cov_every": 10})
        for it in (7, 37):
            ratio = snaps[it].covariances[0].diagonal(dim1=-2, dim2=-1) / cov.diagonal(dim1=-2, dim2=-1)
            self.assertLessEqual(float(ratio.max()), 1.5 ** 2 * 1.001)
        self.assertGreater(float((snaps[37].covariances[0].diagonal(dim1=-2, dim2=-1) / cov.diagonal(dim1=-2, dim2=-1)).max()),
                           1.5 ** 2 * 0.9)  # the pull reached the cap

    def test_pixgs_moves_occludes_and_fills(self):
        from lgs.pixgs import depth_sign, fill_background, make_set, render_view
        h, w = 40, 80
        # source a: wall (displacement +4 px from a to b) and a near square (displacement -10 px); b is the same scene
        img_a = np.zeros((h, w, 3), np.float32); img_a[:] = 0.2
        img_a[10:20, 30:40] = 0.9
        ga = np.zeros((2, h, w), np.float32); ga[0] = 4.0; ga[0, 10:20, 30:40] = -10.0
        img_b = np.zeros((h, w, 3), np.float32); img_b[:] = 0.2
        img_b[10:20, 20:30] = 0.9
        gb = np.zeros((2, h, w), np.float32); gb[0] = -4.0; gb[0, 10:20, 20:30] = 10.0
        gb[0, 10:20, 30:34] = np.nan                              # b pixels hidden behind the square in a
        fg = np.zeros((h, w), bool); fg[10:20, 30:40] = True
        sign = depth_sign(ga[0], fg, np.isfinite(ga[0]))
        self.assertEqual(sign, -1.0)                               # key = -g_x: square (+10) in front of the wall (-4)
        need = np.ones((h, w), bool)
        gb_f = fill_background(gb, need, -sign, 32)
        self.assertTrue(np.allclose(gb_f[0, 10:20, 30:34], -4.0))  # filled from the wall side, not the square
        t = 0.5
        sets = [make_set(img_a, ga, need, t, sign, 0.5, 3.0), make_set(img_b, gb_f, need, 1 - t, -sign, 0.5, 3.0)]
        rgb, alpha, backend = render_view(sets, t, (h, w), {"depth_tol_px": 3.0}, backend="torch")
        self.assertEqual(backend, "torch")
        self.assertTrue(np.allclose(rgb[12:18, 27:33], 0.9, atol=0.05))   # square at x + t*g = 25..35 (centre kept)
        self.assertTrue(np.allclose(rgb[12:18, 50:60], 0.2, atol=0.05))   # wall
        self.assertGreater(float(alpha[12:18, 27:33].min()), 0.9)

    def _two_views(self):
        """Wall (displacement a->b +4 px) behind a near square (-10 px), as in the round-15 test, raw (b derived)."""
        h, w = 40, 80
        img_a = np.full((h, w, 3), 0.2, np.float32); img_a[10:20, 30:40] = 0.9
        img_b = np.full((h, w, 3), 0.2, np.float32); img_b[10:20, 20:30] = 0.9
        ga = np.zeros((2, h, w), np.float32); ga[0] = 4.0; ga[0, 10:20, 30:40] = -10.0
        gb = np.zeros((2, h, w), np.float32); gb[0] = -4.0; gb[0, 10:20, 20:30] = 10.0
        gb[0, 10:20, 30:34] = np.nan
        fg = np.zeros((h, w), bool); fg[10:20, 30:40] = True
        return img_a, img_b, ga, gb, fg

    def test_round16_prep_side_numpy_equals_torch(self):
        from lgs.pixgs import prep_side, prep_side_torch, depth_sign
        _, _, ga, gb, fg = self._two_views()
        sign = depth_sign(ga[0], fg, np.isfinite(ga[0]))
        for g, s in ((ga, sign), (gb, -sign)):
            f_np, ok_np = prep_side(g, s)
            f_t, ok_t = prep_side_torch(g, s, "cpu")
            self.assertTrue(np.array_equal(ok_np, ok_t.numpy()))
            self.assertTrue(np.allclose(np.nan_to_num(f_np, nan=-99), np.nan_to_num(f_t.numpy(), nan=-99), atol=1e-5))
        f, ok = prep_side(gb, -sign)
        self.assertFalse(ok[10:20, 30:34].any())                       # missing pixels are LR-failed ...
        self.assertTrue(np.allclose(f[0, 10:20, 30:34], -4.0))         # ... and take the wall (background) displacement
        _, ok_a = prep_side(ga, sign)
        self.assertFalse(ok_a[10:20, 16:26].any())                     # wall of a landing under the square in b (x+4 in 20..29)
        self.assertTrue(ok_a[10:20, 26:30].all())                      # wall of a landing at 30..33 stays visible in b

    def test_round16_pix2_and_p3h_render_square_and_paste_region(self):
        from lgs.pixgs import PixSet2, depth_sign, make_set2, prep_side, render_view2, gate_by_flow
        from lgs import stereo_synth
        img_a, img_b, ga, gb, fg = self._two_views()
        h, w = fg.shape
        sign = depth_sign(ga[0], fg, np.isfinite(ga[0]))
        fa, oka = prep_side(ga, sign); fb, okb = prep_side(gb, -sign)
        need = np.ones((h, w), bool)
        core_a = np.zeros((h, w), bool); core_a[8:22, 28:42] = True
        core_b = np.zeros((h, w), bool); core_b[8:22, 18:32] = True
        t = 0.5
        sets = [make_set2(img_a, fa, oka, core_a, need, t, sign, 0.5, 3.0, 1.0),
                make_set2(img_b, fb, okb, core_b, need, 1 - t, -sign, 0.5, 3.0, 1.0)]
        rgb, alpha, core, backend = render_view2(sets, t, (h, w), {"depth_tol_px": 3.0}, backend="torch")
        self.assertEqual(backend, "torch")
        self.assertTrue(np.allclose(rgb[12:18, 27:33], 0.9, atol=0.05))    # square at x + t*g = 25..35
        self.assertTrue(np.allclose(rgb[12:18, 50:60], 0.2, atol=0.05))
        self.assertGreater(float(core[12:18, 27:33].min()), 0.9)           # paste region follows the square
        self.assertLess(float(core[12:18, 50:60].max()), 0.1)
        # build-once set gives the same primitives as make_set2 at any t
        ps = PixSet2(img_a, fa, oka, core_a, need, sign, 0.5, 3.0, 1.0, "cpu")
        for tt in (0.25, 0.7):
            pos, col, key, sig = ps.at(tt)
            ref = make_set2(img_a, fa, oka, core_a, need, tt, sign, 0.5, 3.0, 1.0)
            for x, y in zip((pos, col, key, sig), ref):
                self.assertTrue(np.allclose(x.numpy(), y, atol=1e-5))
        P = stereo_synth.prepare(img_a, img_b, fa, fb, oka, okb, core_a, core_b, need, need, sign)
        rgb2, alpha2, core2 = stereo_synth.render(P, t)
        self.assertTrue(np.allclose(rgb2[12:18, 27:33], 0.9, atol=0.05))
        self.assertTrue(np.allclose(rgb2[12:18, 50:60], 0.2, atol=0.05))
        self.assertEqual(float(alpha2.min()), 1.0)                         # covered or filled everywhere here
        self.assertGreater(float(core2[12:18, 27:33].min()), 0.9)
        # gate: identical flow -> nothing pasted; a flow that misses the square -> the square is pasted
        self.assertEqual(float(gate_by_flow(rgb2, rgb2, core2, 0.1, 1, 2).max()), 0.0)
        flow = np.full_like(rgb2, 0.2)
        self.assertGreater(float(gate_by_flow(rgb2, flow, core2, 0.1, 1, 2)[12:18, 27:33].min()), 0.9)

    def test_round17_fused_set_mutual_regions_fusion_and_optimisation(self):
        from lgs import pixfuse
        from lgs.pixgs import depth_sign
        img_a, img_b, ga, gb, fg = self._two_views()
        h, w = fg.shape
        sign = depth_sign(ga[0], fg, np.isfinite(ga[0]))
        rec_a = np.zeros((h, w), bool); rec_a[6:24, 26:46] = True           # a marks the square and its right side
        rec_b = np.zeros((h, w), bool)                                        # b marks nothing
        core_a = np.zeros((h, w), bool); core_a[10:20, 30:40] = True
        G = pixfuse.build(img_a, img_b, ga, gb, rec_a, rec_b, core_a, np.zeros((h, w), bool), sign)
        self.assertTrue(G["rec_b"][10:20, 20:30].all())                       # a's square is now also taken from b
        self.assertGreater(int(G["n_b"]), 0)                                  # b pixels without a partner (hidden in a)
        n_two_view = int(G["both"].sum())
        self.assertEqual(int(G["n_a"]) + int(G["n_b"]), len(G["u"]))
        G0 = pixfuse.build(img_a, img_b, ga, gb, rec_a, rec_b, core_a, np.zeros((h, w), bool), sign, mutual=False)
        self.assertLessEqual(int(G0["rec_b"].sum()), 0)
        self.assertGreater(n_two_view, 0)
        S = pixfuse.make(G, 0.5, 3.0, "cpu")
        rgb, alpha, core, vis = pixfuse.render_view(S, 0.5, (h, w), 0.3, "torch")
        self.assertTrue(np.allclose(rgb[12:18, 27:33], 0.9, atol=0.05))       # square at x + t*g
        self.assertGreater(float(core[12:18, 27:33].min()), 0.9)
        self.assertGreater(float(vis[12:18, 28:32].min()), 0.75)              # seen by both views
        # joint optimisation: start from wrong colours, the loss on the two real views must drop
        S2 = pixfuse.make(G, 0.5, 3.0, "cpu")
        S2.col += 1.0
        log = pixfuse.optimise(S2, img_a, img_b, (h, w), 30, {"color": 0.1, "disp": 0.0, "opacity": 0.0, "scale": 0.0},
                               0.3, "torch")
        self.assertLess(log["log"][-1]["loss"], 0.5 * log["log"][0]["loss"])


if __name__ == '__main__':
    unittest.main()
