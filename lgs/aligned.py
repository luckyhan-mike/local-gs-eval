"""Aligned (off-axis) domain geometry: render GS views in the same image domain the flow branch uses.

The flow pipeline homography-aligns every camera to a reference camera on the zero-parallax plane
(the calibration board plane). In that domain each camera's individual aim is gone; only parallax
from its centre position remains, which is exactly the off-axis camera model of a light field display.

Key fact used here: two cameras with the same centre differ by a homography for *all* points. So
  aligned image of any camera at centre C
    = G_C( image of the aligned reference camera (K_al, R_al) moved to C )
where G_C is the zero-parallax-plane homography that puts plane points back where the reference
camera sees them. The real camera's rotation and intrinsics drop out; only C and the plane matter.
"""
import cv2
import numpy as np
from scipy.linalg import rq
from scipy.optimize import least_squares


def crop_K_to_raw(intr_norm, window):
    """Crop-normalised intrinsics (AnySplat) -> raw-image pixel intrinsics."""
    x0, y0, side = window
    k = np.array(intr_norm, np.float64)
    return np.array([[k[0, 0] * side, k[0, 1] * side, k[0, 2] * side + x0],
                     [0, k[1, 1] * side, k[1, 2] * side + y0], [0, 0, 1.0]])


def aligned_reference(K_raw_ref, c2w_ref, H_ref):
    """Aligned-domain camera of the reference: aligned pixel = H_ref K_ref R_wc (X - C).

    RQ-decomposes H_ref K_ref = K_al Q, so the aligned camera has intrinsics K_al and world-to-camera
    rotation Q R_wc (H_ref is close to a similarity, so Q is close to identity).
    Returns (K_al, c2w_al)."""
    M = np.asarray(H_ref, np.float64) @ K_raw_ref
    M = M / M[2, 2]
    K, Q = rq(M)
    D = np.diag(np.sign(np.diag(K)))
    K, Q = K @ D, D @ Q
    if np.linalg.det(Q) < 0:
        K, Q = -K, -Q
    K = K / K[2, 2]
    c2w = np.eye(4)
    c2w[:3, :3] = c2w_ref[:3, :3] @ Q.T
    c2w[:3, 3] = c2w_ref[:3, 3]
    return K, c2w


def project(K, c2w, X):
    pc = (X - c2w[:3, 3]) @ c2w[:3, :3]
    p = pc @ K.T
    return p[:, :2] / p[:, 2:3], pc[:, 2]


def ray_plane(K, c2w, px, n, d):
    """Intersect pixel rays of a camera with the plane n.X = d."""
    rays = np.c_[px, np.ones(len(px))] @ np.linalg.inv(K).T @ c2w[:3, :3].T
    c = c2w[:3, 3]
    lam = (d - n @ c) / (rays @ n)
    return c + lam[:, None] * rays


def _plane(params):
    th, ph, d = params
    n = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    return n, d


def fit_plane(cams, K_al, c2w_al, depth0):
    """Zero-parallax plane in the GS frame from the context cameras' known homographies.

    cams: list of (K_raw, c2w, H, sample_px) for context cameras (raw pixels; H raw->aligned).
    For plane points, H maps the camera's pixel to where the aligned reference camera sees them.
    depth0: rough distance of the plane along the reference optical axis (initialisation).
    Returns (n, d, rms_px)."""
    n0 = c2w_al[:3, 2]
    d0 = n0 @ (c2w_al[:3, 3] + depth0 * n0)
    th0, ph0 = np.arccos(np.clip(n0[2], -1, 1)), np.arctan2(n0[1], n0[0])

    def res(p):
        n, d = _plane(p)
        out = []
        for K, c2w, H, px in cams:
            X = ray_plane(K, c2w, px, n, d)
            q_hat, _ = project(K_al, c2w_al, X)
            q = cv2.perspectiveTransform(px[None].astype(np.float64), H)[0]
            out.append((q_hat - q).ravel())
        return np.concatenate(out)

    sol = least_squares(res, [th0, ph0, d0], method="lm")
    n, d = _plane(sol.x)
    return n, d, float(np.sqrt(np.mean(sol.fun ** 2)))


def plane_homography(K_al, c2w_al, center, n, d, size):
    """G_C: image of (K_al, R_al) at `center` -> aligned image, from the zero-parallax plane."""
    w, h = size
    grid = np.stack(np.meshgrid(np.linspace(0, w - 1, 8), np.linspace(0, h - 1, 6)), -1).reshape(-1, 2)
    X = ray_plane(K_al, c2w_al, grid, n, d)          # plane points seen by the reference
    moved = c2w_al.copy()
    moved[:3, 3] = center
    j, _ = project(K_al, moved, X)                    # where they land from the new centre
    G, _ = cv2.findHomography(j, grid, 0)
    return G
