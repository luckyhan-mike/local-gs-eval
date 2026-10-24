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
        cov2, op2, std = mip_filter3d(means, cov, op, c2w, intr, size, var=0.2)
        s2 = 0.2 * (d / (fx_n * size)) ** 2
        np.testing.assert_allclose(cov2[0].numpy(), np.eye(3) * (1e-4 + s2), rtol=1e-5)
        np.testing.assert_allclose(op2.item(), 0.8 * ((1e-4 / (1e-4 + s2)) ** 1.5), rtol=1e-4)

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
