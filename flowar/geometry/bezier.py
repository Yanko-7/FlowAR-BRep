import numpy as np


def get_bernstein_poly(t: np.ndarray) -> np.ndarray:
    return np.array([(1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t**2 * (1 - t), t**3])


def eval_rational_bezier_curves(ctrl_pts: np.ndarray, res: int = 50) -> np.ndarray:
    t = np.linspace(0, 1, res)
    B = get_bernstein_poly(t)
    hw_ctrls = ctrl_pts.copy()
    hw_ctrls[..., :3] *= hw_ctrls[..., 3:4]
    curves_hw = np.einsum("pi, mpk -> mik", B, hw_ctrls)
    xyz, w = curves_hw[..., :3], curves_hw[..., 3:4]
    return xyz / np.where(w == 0, 1e-8, w)


def eval_rational_bezier_surfaces(ctrl_pts: np.ndarray, res: int = 20) -> np.ndarray:
    u = np.linspace(0, 1, res)
    v = np.linspace(0, 1, res)
    Bu = get_bernstein_poly(u)
    Bv = get_bernstein_poly(v)
    hw_ctrls = ctrl_pts.copy()
    hw_ctrls[..., :3] *= hw_ctrls[..., 3:4]
    surfs_hw = np.einsum("iu, jv, nijk -> nuvk", Bu, Bv, hw_ctrls)
    xyz, w = surfs_hw[..., :3], surfs_hw[..., 3:4]
    return xyz / np.where(w == 0, 1e-8, w)
