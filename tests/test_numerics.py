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
