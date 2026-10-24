"""Thin wrapper around Practical-RIFE IFNet that also exposes flow, mask and both warps."""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F


class RIFE:
    def __init__(self, rife_dir, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        sys.path.insert(0, rife_dir)
        from train_log.IFNet_HDv3 import IFNet  # noqa: E402  (Practical-RIFE layout)
        from model.warplayer import warp        # noqa: E402

        self.warp = warp
        self.device = device
        net = IFNet()
        sd = torch.load(os.path.join(rife_dir, "train_log", "flownet.pkl"), map_location="cpu")
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        missing = net.load_state_dict(sd, strict=False).missing_keys
        if missing:
            raise RuntimeError(f"RIFE weights do not match IFNet: missing {missing[:5]}")
        self.net = net.to(device).eval()

    def _tensor(self, img):
        return torch.from_numpy(img).permute(2, 0, 1)[None].float().div(255.0).to(self.device)

    @torch.no_grad()
    def synth(self, img0, img1, t, scale=1.0):
        """img0/img1: HxWx3 uint8 RGB. Returns float32 numpy maps at the input size."""
        x0, x1 = self._tensor(img0), self._tensor(img1)
        h, w = x0.shape[-2:]
        mult = max(128, int(128 / scale))
        ph, pw = (mult - h % mult) % mult, (mult - w % mult) % mult
        x0 = F.pad(x0, (0, pw, 0, ph), mode="replicate")
        x1 = F.pad(x1, (0, pw, 0, ph), mode="replicate")
        scale_list = [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]
        flow_list, mask_logit, _ = self.net(torch.cat((x0, x1), 1), float(t), scale_list)
        flow = flow_list[-1]
        m = torch.sigmoid(mask_logit)
        w0 = self.warp(x0, flow[:, :2])
        w1 = self.warp(x1, flow[:, 2:4])
        pred = w0 * m + w1 * (1 - m)

        def out(x):
            return x[0, :, :h, :w].permute(1, 2, 0).float().cpu().numpy()

        return {
            "pred": np.clip(out(pred), 0, 1),
            "w0": np.clip(out(w0), 0, 1),
            "w1": np.clip(out(w1), 0, 1),
            "mask": out(m)[..., 0],
            "flow": out(flow),  # [F_t->0 (x,y), F_t->1 (x,y)]
        }
