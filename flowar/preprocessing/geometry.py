"""Convert analytic and spline geometry to rational cubic Bezier control points.

Geometry that cannot be represented exactly as one patch uses sampled cubic fitting.
"""

import math

import numpy as np
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepTools import breptools
from OCC.Core.Geom import (
    Geom_BezierCurve,
    Geom_BezierSurface,
    Geom_BSplineCurve,
    Geom_BSplineSurface,
    Geom_Circle,
    Geom_ConicalSurface,
    Geom_CylindricalSurface,
    Geom_Ellipse,
    Geom_Hyperbola,
    Geom_Line,
    Geom_Parabola,
    Geom_Plane,
    Geom_RectangularTrimmedSurface,
    Geom_SphericalSurface,
    Geom_ToroidalSurface,
    Geom_TrimmedCurve,
)
from OCC.Core.GeomAbs import GeomAbs_C1
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline, GeomAPI_PointsToBSplineSurface
from OCC.Core.GeomConvert import geomconvert
from OCC.Core.gp import gp_Pnt
from OCC.Core.Precision import precision
from OCC.Core.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt
from OCC.Core.TopoDS import topods

# ---------------------------------------------------------------------------
# Rational cubic Bézier arc helper (shared by circle, ellipse, conic arcs)
# ---------------------------------------------------------------------------


def _cubic_bezier_arc(theta):
    """Return 4 control points (2D) and weights for a rational cubic Bézier
    representation of a circular arc of angle *theta* (0 < theta <= pi).

    Points are in the local frame where the arc starts at (1, 0) and sweeps
    counter-clockwise by *theta*.  Returns (pts_2d[4,2], weights[4]).
    """
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    half = theta / 2.0
    cos_h = math.cos(half)
    sin_h = math.sin(half)

    # Exact rational quadratic for arc from 0 to theta on unit circle
    # Q0=(1,0) w0=1, Q1=tangent intersection w1=cos(θ/2), Q2=(cosθ,sinθ) w2=1
    # In homogeneous coords H=(w*x, w*y, w):
    H0 = np.array([1.0, 0.0, 1.0])
    H1 = np.array([cos_h, sin_h, cos_h])
    H2 = np.array([cos_t, sin_t, 1.0])

    # Degree-elevate quadratic → cubic
    G0 = H0
    G1 = (1.0 / 3.0) * H0 + (2.0 / 3.0) * H1
    G2 = (2.0 / 3.0) * H1 + (1.0 / 3.0) * H2
    G3 = H2

    weights = np.array([G0[2], G1[2], G2[2], G3[2]])
    pts = np.zeros((4, 2))
    for i, G in enumerate([G0, G1, G2, G3]):
        pts[i] = G[:2] / G[2] if abs(G[2]) > 1e-30 else G[:2]

    return pts, weights


def _conic_arc_3d(center, x_axis, y_axis, rx, ry, theta_start, theta_span):
    """Build a 4-control-point rational cubic Bézier for an elliptical /
    circular arc in 3-D.

    Parameters
    ----------
    center : array-like (3,)
    x_axis, y_axis : array-like (3,)  – unit vectors of the local plane
    rx, ry : float – semi-axes (rx along x_axis, ry along y_axis)
    theta_start : float – start angle in the local frame
    theta_span : float – arc span (0 < theta_span <= pi)

    Returns (pts[4,3], weights[4]).
    """
    pts2d, w = _cubic_bezier_arc(theta_span)

    # rotate 2d arc by theta_start
    cs, ss = math.cos(theta_start), math.sin(theta_start)
    R = np.array([[cs, -ss], [ss, cs]])
    pts2d = pts2d @ R.T

    center = np.asarray(center, dtype=np.float64)
    xa = np.asarray(x_axis, dtype=np.float64)
    ya = np.asarray(y_axis, dtype=np.float64)

    pts3d = np.zeros((4, 3))
    for i in range(4):
        pts3d[i] = center + rx * pts2d[i, 0] * xa + ry * pts2d[i, 1] * ya

    return pts3d, w


def _linear_bezier_3d(p_start, p_end):
    """Degree-elevate a line segment to a cubic Bézier with 4 control points, w=1."""
    p0 = np.asarray(p_start, dtype=np.float64)
    p3 = np.asarray(p_end, dtype=np.float64)
    p1 = p0 + (p3 - p0) / 3.0
    p2 = p0 + 2.0 * (p3 - p0) / 3.0
    return np.array([p0, p1, p2, p3]), np.ones(4)


# ---------------------------------------------------------------------------
# Curve converters  (curve, t1, t2) -> np.ndarray [4, 4]
# ---------------------------------------------------------------------------


def _convert_line(curve, t1, t2):
    line = Geom_Line.DownCast(curve)
    p0 = line.Value(t1)
    p1 = line.Value(t2)
    pts, w = _linear_bezier_3d([p0.X(), p0.Y(), p0.Z()], [p1.X(), p1.Y(), p1.Z()])
    return _pack_curve(pts, w)


def _convert_circle(curve, t1, t2):
    circ = Geom_Circle.DownCast(curve)
    c = circ.Circ()
    center = c.Location()
    ax = c.Position()
    xa = ax.XDirection()
    ya = ax.YDirection()
    R = c.Radius()
    pts, w = _conic_arc_3d(
        [center.X(), center.Y(), center.Z()],
        [xa.X(), xa.Y(), xa.Z()],
        [ya.X(), ya.Y(), ya.Z()],
        R,
        R,
        t1,
        t2 - t1,
    )
    return _pack_curve(pts, w)


def _convert_ellipse(curve, t1, t2):
    ell = Geom_Ellipse.DownCast(curve)
    e = ell.Elips()
    center = e.Location()
    ax = e.Position()
    xa = ax.XDirection()
    ya = ax.YDirection()
    pts, w = _conic_arc_3d(
        [center.X(), center.Y(), center.Z()],
        [xa.X(), xa.Y(), xa.Z()],
        [ya.X(), ya.Y(), ya.Z()],
        e.MajorRadius(),
        e.MinorRadius(),
        t1,
        t2 - t1,
    )
    return _pack_curve(pts, w)


def _convert_parabola(curve, t1, t2):
    par = Geom_Parabola.DownCast(curve)
    pb = par.Parab()
    center = pb.Location()
    ax = pb.Position()
    xa = ax.XDirection()
    ya = ax.YDirection()
    focal = pb.Focal()
    cx = np.array([center.X(), center.Y(), center.Z()])
    xd = np.array([xa.X(), xa.Y(), xa.Z()])
    yd = np.array([ya.X(), ya.Y(), ya.Z()])

    def eval_parab(t):
        return cx + (t * t / (4.0 * focal)) * xd + t * yd

    # Quadratic Bézier → degree-elevate to cubic
    P0 = eval_parab(t1)
    P2 = eval_parab(t2)
    tm = (t1 + t2) / 2.0
    Pm = eval_parab(tm)
    # Quadratic Bézier: P0, P1_q, P2 where Pm = 0.25*P0 + 0.5*P1_q + 0.25*P2
    P1_q = 2.0 * Pm - 0.5 * P0 - 0.5 * P2

    # Degree elevate from quadratic to cubic
    C0 = P0
    C1 = P0 / 3.0 + 2.0 * P1_q / 3.0
    C2 = 2.0 * P1_q / 3.0 + P2 / 3.0
    C3 = P2
    pts = np.array([C0, C1, C2, C3])
    return _pack_curve(pts, np.ones(4))


def _convert_hyperbola(curve, t1, t2):
    hyp = Geom_Hyperbola.DownCast(curve)
    h = hyp.Hypr()
    center = h.Location()
    ax = h.Position()
    xa = ax.XDirection()
    ya = ax.YDirection()
    a = h.MajorRadius()
    b = h.MinorRadius()
    cx = np.array([center.X(), center.Y(), center.Z()])
    xd = np.array([xa.X(), xa.Y(), xa.Z()])
    yd = np.array([ya.X(), ya.Y(), ya.Z()])

    def eval_hyp(t):
        return cx + a * math.cosh(t) * xd + b * math.sinh(t) * yd

    def eval_hyp_d(t):
        return a * math.sinh(t) * xd + b * math.cosh(t) * yd

    # Exact rational quadratic for a hyperbola arc, then degree-elevate
    dt = t2 - t1
    half_dt = dt / 2.0
    alpha = math.tanh(half_dt)
    w1 = math.cosh(half_dt)

    Q0 = eval_hyp(t1)
    Q2 = eval_hyp(t2)
    Q1 = Q0 + alpha * eval_hyp_d(t1)

    # Homogeneous coordinates (4-component: x, y, z, w)
    H0 = np.concatenate([Q0, [1.0]])
    H1 = np.concatenate([w1 * Q1, [w1]])
    H2 = np.concatenate([Q2, [1.0]])

    G0 = H0
    G1 = (1.0 / 3.0) * H0 + (2.0 / 3.0) * H1
    G2 = (2.0 / 3.0) * H1 + (1.0 / 3.0) * H2
    G3 = H2

    weights = np.array([G0[3], G1[3], G2[3], G3[3]])
    pts = np.zeros((4, 3))
    for k, G in enumerate([G0, G1, G2, G3]):
        pts[k] = G[:3] / G[3]
    return _pack_curve(pts, weights)


def _convert_bspline_curve(curve, t1, t2):
    bsp = Geom_BSplineCurve.DownCast(curve.Copy())
    bsp.Segment(t1, t2)
    bsp.IncreaseDegree(max(3, bsp.Degree()))
    if bsp.NbPoles() == 4:
        return _extract_bspline_curve(bsp)
    return None


def _convert_bezier_curve(curve, t1, t2):
    bez = Geom_BezierCurve.DownCast(curve.Copy())
    bez.Segment(t1, t2)
    if bez.Degree() < 3:
        bez.Increase(3)
    if bez.NbPoles() == 4:
        pts = np.zeros((4, 3))
        w = np.ones(4)
        for i in range(4):
            p = bez.Pole(i + 1)
            pts[i] = [p.X(), p.Y(), p.Z()]
            if bez.IsRational():
                w[i] = bez.Weight(i + 1)
        return _pack_curve(pts, w)
    return None


def _convert_trimmed_curve(curve, t1, t2):
    tc = Geom_TrimmedCurve.DownCast(curve)
    basis = tc.BasisCurve()
    return analytic_curve_to_rational_bezier(basis, t1, t2)


def _extract_bspline_curve(bsp):
    is_rational = bsp.IsRational()
    pts = np.zeros((4, 3))
    w = np.ones(4)
    for i in range(4):
        p = bsp.Pole(i + 1)
        pts[i] = [p.X(), p.Y(), p.Z()]
        if is_rational:
            w[i] = bsp.Weight(i + 1)
    return _pack_curve(pts, w)


def _pack_curve(pts, weights):
    """Pack (4,3) pts and (4,) weights into (4,4) array [x,y,z,w]."""
    out = np.zeros((4, 4), dtype=np.float64)
    out[:, :3] = pts
    out[:, 3] = weights
    return out


# ---------------------------------------------------------------------------
# Surface converters  (surf, u1, u2, v1, v2) -> np.ndarray [4, 4, 4]
# ---------------------------------------------------------------------------


def _gp_to_np3(p):
    return np.array([p.X(), p.Y(), p.Z()])


def _gp_dir_to_np3(d):
    return np.array([d.X(), d.Y(), d.Z()])


def _convert_plane(surf, u1, u2, v1, v2):
    pln = Geom_Plane.DownCast(surf)
    ax3 = pln.Pln().Position()
    origin = _gp_to_np3(ax3.Location())
    xd = _gp_dir_to_np3(ax3.XDirection())
    yd = _gp_dir_to_np3(ax3.YDirection())

    u_vals = np.linspace(u1, u2, 4)
    v_vals = np.linspace(v1, v2, 4)
    poles = np.zeros((4, 4, 4), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            poles[i, j, :3] = origin + u_vals[i] * xd + v_vals[j] * yd
            poles[i, j, 3] = 1.0
    return poles


def _convert_cylinder(surf, u1, u2, v1, v2):
    cyl = Geom_CylindricalSurface.DownCast(surf)
    c = cyl.Cylinder()
    ax3 = c.Position()
    origin = _gp_to_np3(ax3.Location())
    xd = _gp_dir_to_np3(ax3.XDirection())
    yd = _gp_dir_to_np3(ax3.YDirection())
    zd = _gp_dir_to_np3(ax3.Direction())
    R = c.Radius()

    # u-direction: circular arc
    arc_pts, arc_w = _conic_arc_3d(origin, xd, yd, R, R, u1, u2 - u1)
    # v-direction: linear along axis
    v_pts, v_w = _linear_bezier_3d(v1 * zd, v2 * zd)

    return _build_tensor_product_surface(arc_pts, arc_w, v_pts, v_w)


def _convert_cone(surf, u1, u2, v1, v2):
    con = Geom_ConicalSurface.DownCast(surf)
    c = con.Cone()
    ax3 = c.Position()
    origin = _gp_to_np3(ax3.Location())
    xd = _gp_dir_to_np3(ax3.XDirection())
    yd = _gp_dir_to_np3(ax3.YDirection())
    zd = _gp_dir_to_np3(ax3.Direction())
    ref_r = c.RefRadius()
    semi = c.SemiAngle()

    # v-direction ruling: radius changes linearly with v
    # At parameter v, radius = ref_r + v * sin(semi), z = v * cos(semi)
    v_vals = np.linspace(v1, v2, 4)
    radii = ref_r + v_vals * math.sin(semi)
    z_vals = v_vals * math.cos(semi)

    # For each v slice, build the circular arc, then assemble
    arc_pts_u, arc_w_u = _conic_arc_3d(np.zeros(3), xd, yd, 1.0, 1.0, u1, u2 - u1)
    # arc_pts_u are unit circle arc points (relative to origin)

    poles = np.zeros((4, 4, 4), dtype=np.float64)
    for j in range(4):
        for i in range(4):
            pt = origin + radii[j] * arc_pts_u[i] + z_vals[j] * zd
            poles[i, j, :3] = pt
            poles[i, j, 3] = arc_w_u[i]  # v-direction is polynomial (w=1)
    return poles


def _convert_sphere(surf, u1, u2, v1, v2):
    sph = Geom_SphericalSurface.DownCast(surf)
    s = sph.Sphere()
    ax3 = s.Position()
    origin = _gp_to_np3(ax3.Location())
    xd = _gp_dir_to_np3(ax3.XDirection())
    yd = _gp_dir_to_np3(ax3.YDirection())
    zd = _gp_dir_to_np3(ax3.Direction())
    R = s.Radius()

    # u = longitude, v = latitude
    # P(u,v) = center + R*cos(v)*cos(u)*xd + R*cos(v)*sin(u)*yd + R*sin(v)*zd

    # v-direction (latitude) arc: in the (cos(u0)*xd, zd) plane, this is a
    # circular arc of radius R.  We build it for a single reference u, then
    # tensor-product with u-arc.

    # latitude arc in (local_x, zd) plane
    lat_pts, lat_w = _conic_arc_3d(
        np.zeros(3),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        R,
        R,
        v1,
        v2 - v1,
    )
    # lat_pts[:,0] = R*cos(v), lat_pts[:,2] = R*sin(v)

    # longitude arc (unit circle in xd, yd plane)
    lon_pts_2d, lon_w = _cubic_bezier_arc(u2 - u1)
    cs, ss = math.cos(u1), math.sin(u1)
    Rot = np.array([[cs, -ss], [ss, cs]])
    lon_pts_2d = lon_pts_2d @ Rot.T

    poles = np.zeros((4, 4, 4), dtype=np.float64)
    for i in range(4):  # u (longitude)
        for j in range(4):  # v (latitude)
            cos_v_R = lat_pts[j, 0]  # R*cos(v)
            sin_v_R = lat_pts[j, 2]  # R*sin(v)
            pt = (
                origin
                + cos_v_R * lon_pts_2d[i, 0] * xd
                + cos_v_R * lon_pts_2d[i, 1] * yd
                + sin_v_R * zd
            )
            poles[i, j, :3] = pt
            poles[i, j, 3] = lon_w[i] * lat_w[j]
    return poles


def _convert_torus(surf, u1, u2, v1, v2):
    tor = Geom_ToroidalSurface.DownCast(surf)
    t = tor.Torus()
    ax3 = t.Position()
    origin = _gp_to_np3(ax3.Location())
    xd = _gp_dir_to_np3(ax3.XDirection())
    yd = _gp_dir_to_np3(ax3.YDirection())
    zd = _gp_dir_to_np3(ax3.Direction())
    R_major = t.MajorRadius()
    r_minor = t.MinorRadius()

    # u = major angle (around z-axis), v = minor angle (tube cross-section)
    # P(u,v) = center + (R + r*cos(v))*cos(u)*xd + (R + r*cos(v))*sin(u)*yd + r*sin(v)*zd

    # minor circle arc in local (radial_out, zd) plane
    minor_pts, minor_w = _conic_arc_3d(
        np.zeros(3),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        r_minor,
        r_minor,
        v1,
        v2 - v1,
    )
    # minor_pts[:,0] = r*cos(v), minor_pts[:,2] = r*sin(v)

    # major circle (unit) in xd, yd
    major_2d, major_w = _cubic_bezier_arc(u2 - u1)
    cs, ss = math.cos(u1), math.sin(u1)
    Rot = np.array([[cs, -ss], [ss, cs]])
    major_2d = major_2d @ Rot.T

    poles = np.zeros((4, 4, 4), dtype=np.float64)
    for i in range(4):  # u (major)
        for j in range(4):  # v (minor)
            r_cos_v = minor_pts[j, 0]  # r*cos(v)
            r_sin_v = minor_pts[j, 2]  # r*sin(v)
            radial = R_major + r_cos_v
            pt = origin + radial * major_2d[i, 0] * xd + radial * major_2d[i, 1] * yd + r_sin_v * zd
            poles[i, j, :3] = pt
            poles[i, j, 3] = major_w[i] * minor_w[j]
    return poles


def _convert_bspline_surface(surf, u1, u2, v1, v2):
    bsp = Geom_BSplineSurface.DownCast(surf.Copy())
    bsp.Segment(u1, u2, v1, v2)
    bsp.IncreaseDegree(max(3, bsp.UDegree()), max(3, bsp.VDegree()))
    if bsp.NbUPoles() == 4 and bsp.NbVPoles() == 4:
        return _extract_bspline_surface(bsp)
    return None


def _convert_bezier_surface(surf, u1, u2, v1, v2):
    bsp = geomconvert.SurfaceToBSplineSurface(Geom_BezierSurface.DownCast(surf))
    bsp.Segment(u1, u2, v1, v2)
    bsp.IncreaseDegree(max(3, bsp.UDegree()), max(3, bsp.VDegree()))
    if bsp.NbUPoles() == 4 and bsp.NbVPoles() == 4:
        return _extract_bspline_surface(bsp)
    return None


def _convert_trimmed_surface(surf, u1, u2, v1, v2):
    ts = Geom_RectangularTrimmedSurface.DownCast(surf)
    basis = ts.BasisSurface()
    return analytic_surface_to_rational_bezier(basis, u1, u2, v1, v2)


def _extract_bspline_surface(bsp):
    is_rational = bsp.IsURational() or bsp.IsVRational()
    poles = np.zeros((4, 4, 4), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            p = bsp.Pole(i + 1, j + 1)
            w = bsp.Weight(i + 1, j + 1) if is_rational else 1.0
            poles[i, j, :3] = [p.X(), p.Y(), p.Z()]
            poles[i, j, 3] = w
    return poles


def _build_tensor_product_surface(u_pts, u_w, v_pts, v_w):
    """Build (4,4,4) poles from u-curve (4,3) and v-curve (4,3) via tensor product.
    v_pts are *offsets* added to u_pts.
    """
    poles = np.zeros((4, 4, 4), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            poles[i, j, :3] = u_pts[i] + v_pts[j]
            poles[i, j, 3] = u_w[i] * v_w[j]
    return poles


# ---------------------------------------------------------------------------
# Fallback converters (point-sampling)
# ---------------------------------------------------------------------------


def _compute_tol(pts, is_curve):
    pts = np.asarray(pts)
    diag = np.linalg.norm(np.ptp(pts, axis=0))
    if diag < 1e-10:
        return 1e-7
    cov = np.cov(pts.T)
    extents = np.ptp(pts @ np.linalg.eigh(cov)[1], axis=0) if cov.ndim == 2 else np.ptp(pts, axis=0)
    feat_size = np.sort(extents)[-1 if is_curve else -2]
    return np.clip(diag * 1e-3, 1e-7, feat_size * 0.2)


def _fallback_curve(curve, t1, t2):
    pts_arr = TColgp_Array1OfPnt(1, 16)
    raw = []
    for i, t in enumerate(np.linspace(t1, t2, 16), 1):
        p = curve.Value(float(t))
        pts_arr.SetValue(i, gp_Pnt(p.X(), p.Y(), p.Z()))
        raw.append([p.X(), p.Y(), p.Z()])
    tol = _compute_tol(raw, is_curve=True)
    for _ in range(7):
        fitter = GeomAPI_PointsToBSpline(pts_arr, 3, 3, GeomAbs_C1, tol)
        if fitter.IsDone() and fitter.Curve().NbPoles() == 4:
            return _extract_bspline_curve(fitter.Curve())
        tol *= 2
    raise RuntimeError("Fallback curve fitting failed.")


def _fallback_surface(surf, u1, u2, v1, v2):
    pts_arr = TColgp_Array2OfPnt(1, 4, 1, 4)
    raw = []
    for i, u in enumerate(np.linspace(u1, u2, 4), 1):
        for j, v in enumerate(np.linspace(v1, v2, 4), 1):
            p = surf.Value(float(u), float(v))
            pts_arr.SetValue(i, j, gp_Pnt(p.X(), p.Y(), p.Z()))
            raw.append([p.X(), p.Y(), p.Z()])
    tol = _compute_tol(raw, is_curve=False)
    for _ in range(7):
        fitter = GeomAPI_PointsToBSplineSurface(pts_arr, 3, 3, GeomAbs_C1, tol)
        if (
            fitter.IsDone()
            and fitter.Surface().NbUPoles() == 4
            and fitter.Surface().NbVPoles() == 4
        ):
            return _extract_bspline_surface(fitter.Surface())
        tol *= 2
    raise RuntimeError("Fallback surface fitting failed.")


# ---------------------------------------------------------------------------
# Dispatch dictionaries
# ---------------------------------------------------------------------------

CURVE_CONVERTERS = {
    "Geom_Line": _convert_line,
    "Geom_Circle": _convert_circle,
    "Geom_Ellipse": _convert_ellipse,
    "Geom_Parabola": _convert_parabola,
    "Geom_Hyperbola": _convert_hyperbola,
    "Geom_BSplineCurve": _convert_bspline_curve,
    "Geom_BezierCurve": _convert_bezier_curve,
    "Geom_TrimmedCurve": _convert_trimmed_curve,
}

SURFACE_CONVERTERS = {
    "Geom_Plane": _convert_plane,
    "Geom_CylindricalSurface": _convert_cylinder,
    "Geom_ConicalSurface": _convert_cone,
    "Geom_SphericalSurface": _convert_sphere,
    "Geom_ToroidalSurface": _convert_torus,
    "Geom_BSplineSurface": _convert_bspline_surface,
    "Geom_BezierSurface": _convert_bezier_surface,
    "Geom_RectangularTrimmedSurface": _convert_trimmed_surface,
}


# ---------------------------------------------------------------------------
# Public dispatch functions
# ---------------------------------------------------------------------------


def analytic_curve_to_rational_bezier(curve, t1, t2) -> np.ndarray:
    name = curve.DynamicType().Name()
    converter = CURVE_CONVERTERS.get(name)
    if converter:
        result = converter(curve, t1, t2)
        if result is not None:
            return result.astype(np.float64)
    return _fallback_curve(curve, t1, t2).astype(np.float64)


def analytic_surface_to_rational_bezier(surf, u1, u2, v1, v2) -> np.ndarray:
    name = surf.DynamicType().Name()
    converter = SURFACE_CONVERTERS.get(name)
    if converter:
        result = converter(surf, u1, u2, v1, v2)
        if result is not None:
            return result.astype(np.float64)
    return _fallback_surface(surf, u1, u2, v1, v2).astype(np.float64)


# ---------------------------------------------------------------------------
# Public entry points (updated)
# ---------------------------------------------------------------------------


def extract_or_fit_bicubic_patch(face) -> np.ndarray:
    face_ds = topods.Face(face)
    u1, u2, v1, v2 = breptools.UVBounds(face_ds)
    if any(map(precision.IsInfinite, (u1, u2, v1, v2))):
        raise ValueError("Invalid UV bounds for face.")

    surf = BRep_Tool.Surface(face_ds)
    poles = analytic_surface_to_rational_bezier(surf, u1, u2, v1, v2)

    # BRep_Tool.Surface(face) already includes the face location.
    return poles


def extract_or_fit_cubic_curve(edge) -> np.ndarray:
    edge_ds = topods.Edge(edge)
    curve, t1, t2 = BRep_Tool.Curve(edge_ds)
    if curve is None:
        raise ValueError("Edge has no underlying 3D curve.")
    if any(map(precision.IsInfinite, (t1, t2))):
        raise ValueError("Invalid bounds for curve (Infinite).")

    poles = analytic_curve_to_rational_bezier(curve, t1, t2)

    # BRep_Tool.Curve(edge) already includes the edge location.
    return poles
