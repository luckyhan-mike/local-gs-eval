"""Region-restricted PSNR / SSIM / LPIPS and a rank-based AUC."""
import numpy as np
import torch
from skimage.metrics import structural_similarity
from scipy.stats import rankdata

from .common import bbox


def psnr(pred, gt, m):
    if m.sum() == 0:
        return float("nan")
    mse = np.mean((pred[m] - gt[m]) ** 2)
    return float(-10 * np.log10(max(mse, 1e-10)))


def ssim_map(pred, gt):
    _, smap = structural_similarity(gt, pred, channel_axis=-1, data_range=1.0,
                                    gaussian_weights=True, sigma=1.5, full=True)
    return smap.mean(-1)


class LPIPSRegion:
    """LPIPS on the region's bounding box; pixels outside the region are copied from GT."""

    def __init__(self, device="cuda"):
        import lpips
        self.net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, pred, gt, m, pad=16, min_side=64):
        box = bbox(m, pad)
        if box is None:
            return float("nan")
        x0, y0, x1, y1 = box
        h, w = m.shape
        if x1 - x0 < min_side:
            c = (x0 + x1) // 2
            x0, x1 = max(c - min_side // 2, 0), min(c + min_side // 2, w)
        if y1 - y0 < min_side:
            c = (y0 + y1) // 2
            y0, y1 = max(c - min_side // 2, 0), min(c + min_side // 2, h)
        p = np.where(m[..., None], pred, gt)[y0:y1, x0:x1]
        g = gt[y0:y1, x0:x1]

        def t(x):
            return torch.from_numpy(x).permute(2, 0, 1)[None].float().to(self.device) * 2 - 1

        return float(self.net(t(p), t(g)).item())


def auc(score, label, max_n=200000, seed=0):
    """Mann-Whitney AUC of score predicting a boolean label."""
    score, label = score.ravel(), label.ravel()
    if len(score) > max_n:
        idx = np.random.default_rng(seed).choice(len(score), max_n, replace=False)
        score, label = score[idx], label[idx]
    npos, nneg = label.sum(), (~label).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    ranks = rankdata(score, method="average")
    return float((ranks[label].sum() - npos * (npos + 1) / 2) / (npos * nneg))
