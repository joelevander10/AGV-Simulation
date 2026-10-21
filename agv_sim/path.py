"""Route geometry: waypoints -> "runs" of straight lines + tangent fillet arcs, sampled every ds.

A *run* is continuous motion between two zero-speed points (start, stops, pivots,
direction changes). Pass-through waypoints inside a run become tangent arcs with the
waypoint's corner radius (like a magnetic-tape curve or a Nav2 smoothed corner).

Rules
-----
* waypoint action 'stop' or 'pivot'          -> AGV stops there
* direction changes (forward <-> reverse)     -> AGV stops there
* radius 0 at a corner                        -> AGV stops and pivots in place
* heading differs between arriving/leaving leg at a stop -> pivot in place
* With sensor tracking (tape), at a pivot point the sensor overshoots by its offset
  so that the DRIVE AXLE ends up over the waypoint before spinning.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .config import AGVConfig, RouteConfig


def wrap(a):
    """Wrap angle(s) to [-pi, pi)."""
    return (np.asarray(a) + math.pi) % (2 * math.pi) - math.pi


def _left(d):
    return np.array([-d[1], d[0]])


def _signed_turn(d1, d2) -> float:
    return math.atan2(d1[0] * d2[1] - d1[1] * d2[0], d1[0] * d2[0] + d1[1] * d2[1])


@dataclass
class Arc:
    radius: float
    turn: float                   # signed turn angle, rad (+ = left)
    center: Tuple[float, float]
    s_start: float
    s_end: float
    waypoint: int                 # index in route.waypoints
    label: str
    requested_radius: float


@dataclass
class Run:
    index: int
    direction: int                # +1 forward, -1 reverse
    start_wp: int
    end_wp: int
    label: str
    x: np.ndarray                 # reference path of the tracking point
    y: np.ndarray
    heading: np.ndarray           # path tangent (travel direction)
    kappa: np.ndarray             # signed curvature 1/m
    s: np.ndarray                 # arc length
    vcap: np.ndarray              # per-sample speed cap from waypoint 'speed'
    arcs: List[Arc]
    track_offset: float
    end_dwell: float
    end_name: str
    end_pivot: bool
    # per-sample motion settings from the speed mode(s) of the leg(s)
    vmax: np.ndarray = None       # straight speed (v_straight or v_reverse of the mode)
    vcurve: np.ndarray = None     # curve speed of the mode
    acc: np.ndarray = None        # m/s^2
    dec: np.ndarray = None        # m/s^2
    grade: np.ndarray = None      # % slope along the drawn leg direction (+ uphill towards next point)
    modes: str = ""               # speed mode names used on this run (for the report)
    end_cart: str = ""            # '' | attach | detach at the end stop
    end_cart_mass: Optional[float] = None


@dataclass
class RoutePlan:
    runs: List[Run]
    loop: bool
    start_heading: float
    start_axle: Tuple[float, float]
    start_dwell: float
    start_name: str
    warnings: List[str] = field(default_factory=list)
    start_cart: str = ""                   # cart event at the first waypoint (at t = 0)
    start_cart_mass: Optional[float] = None

    @property
    def length(self) -> float:
        return float(sum(r.s[-1] for r in self.runs))


class _Sampler:
    """Samples lines/arcs every ds. Each sample remembers the leg it belongs to
    (la = incoming leg, lb = outgoing leg; they differ only on corner arcs)."""

    def __init__(self, p0, heading0, ds, leg):
        self.ds = ds
        self.x, self.y = [float(p0[0])], [float(p0[1])]
        self.h, self.k = [heading0], [0.0]
        self.la, self.lb = [leg], [leg]

    @property
    def n(self):
        return len(self.x)

    def line(self, p1, leg):
        p0 = np.array([self.x[-1], self.y[-1]])
        d = np.asarray(p1, float) - p0
        L = float(np.hypot(*d))
        if L < 1e-9:
            return
        n = max(1, math.ceil(L / self.ds))
        f = np.arange(1, n + 1) / n
        self.x.extend(p0[0] + f * d[0])
        self.y.extend(p0[1] + f * d[1])
        self.h.extend([math.atan2(d[1], d[0])] * n)
        self.k.extend([0.0] * n)
        self.la.extend([leg] * n)
        self.lb.extend([leg] * n)

    def arc(self, c, r, a0, turn, leg_in, leg_out):
        n = max(2, math.ceil(r * abs(turn) / self.ds))
        ang = a0 + turn * np.arange(1, n + 1) / n
        sgn = math.copysign(1.0, turn)
        self.x.extend(c[0] + r * np.cos(ang))
        self.y.extend(c[1] + r * np.sin(ang))
        self.h.extend(ang + sgn * math.pi / 2)
        self.k.extend([sgn / r] * n)
        self.la.extend([leg_in] * n)
        self.lb.extend([leg_out] * n)

    def arrays(self):
        x, y = np.array(self.x), np.array(self.y)
        s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
        return x, y, np.array(self.h), np.array(self.k), s, np.array(self.la), np.array(self.lb)


def build_plan(route: RouteConfig, agv: AGVConfig) -> RoutePlan:
    wps = route.waypoints
    n = len(wps)
    if n < 2:
        raise ValueError("The route needs at least 2 waypoints.")
    ds = agv.simulation.ds
    perf = agv.performance
    warnings: List[str] = []

    ids = list(range(n)) + ([0] if route.loop else [])
    P = np.array([[wps[i].x, wps[i].y] for i in ids], dtype=float)
    m = len(P)
    nleg = m - 1

    def name(i):
        return wps[i].name or f"WP{i}"

    tang = []
    for k in range(nleg):
        d = P[k + 1] - P[k]
        L = float(np.hypot(*d))
        if L < 1e-6:
            raise ValueError(f"{name(ids[k])} and {name(ids[k + 1])} are at the same position.")
        tang.append(d / L)
    dirs = [-1 if str(wps[ids[k]].direction).lower().startswith("r") else 1 for k in range(nleg)]
    caps = [float(wps[ids[k]].speed) if wps[ids[k]].speed else math.inf for k in range(nleg)]
    legmode = [agv.mode_params(wps[ids[k]].mode or route.default_mode) for k in range(nleg)]
    grades = [float(wps[ids[k]].grade or 0.0) for k in range(nleg)]
    for k in range(nleg):
        if abs(grades[k]) > 15:
            warnings.append(f"{name(ids[k])}: {grades[k]:.0f} % slope is very steep for an AGV.")

    def cart_event(i):
        ev = (wps[i].cart or "").strip().lower()
        if ev in ("", "none"):
            return ""
        if ev not in ("attach", "detach", "swap"):
            raise ValueError(f"{name(i)}: cart must be attach, detach or swap (got '{wps[i].cart}')")
        ld = (wps[i].cart_load or "").strip().lower()
        if ld not in ("", "full", "empty"):
            raise ValueError(f"{name(i)}: cart_load must be full or empty (got '{wps[i].cart_load}')")
        return ev

    def cart_mass(i):
        w = wps[i]
        if w.cart_mass is not None:
            return float(w.cart_mass)
        return agv.train_mass(w.cart_load or "full")

    def leg_heading(k):
        return math.atan2(tang[k][1], tang[k][0]) + (math.pi if dirs[k] < 0 else 0.0)

    def radius_at(j):
        r = wps[ids[j]].radius
        return route.default_radius if r is None else float(r)

    # ---- which waypoints are zero-speed points ---------------------------
    is_stop = [False] * m
    is_stop[0] = is_stop[-1] = True
    for j in range(1, m - 1):
        wp = wps[ids[j]]
        act = (wp.action or "pass").lower()
        if act not in ("pass", "stop", "pivot"):
            raise ValueError(f"{name(ids[j])}: action must be pass, stop or pivot (got '{wp.action}')")
        phi = _signed_turn(tang[j - 1], tang[j])
        if act in ("stop", "pivot") or dirs[j - 1] != dirs[j] or cart_event(ids[j]):
            is_stop[j] = True
        elif abs(phi) > 1e-3 and radius_at(j) <= 0:
            is_stop[j] = True
        elif abs(phi) > math.radians(170):
            is_stop[j] = True
            warnings.append(f"{name(ids[j])}: {math.degrees(abs(phi)):.0f} deg turn is too sharp for a curve; "
                            f"the AGV stops and pivots there.")

    end_pivot = [False] * m
    for j in range(1, m):
        if not is_stop[j]:
            continue
        k_out = j if j < nleg else (0 if route.loop else None)
        if k_out is None:
            continue
        if abs(float(wrap(leg_heading(k_out) - leg_heading(j - 1)))) > math.radians(1.0):
            end_pivot[j] = True

    # ---- build runs --------------------------------------------------------
    stops = [j for j in range(m) if is_stop[j]]
    runs: List[Run] = []
    for ri, (a, b) in enumerate(zip(stops[:-1], stops[1:])):
        direction = dirs[a]
        d = agv.tracking_offset(direction)
        pts = P[a:b + 1].copy()
        if end_pivot[b] and d > 0:
            pts[-1] = pts[-1] + d * tang[b - 1]
        nloc = len(pts)
        seg_len = [float(np.hypot(*(pts[q + 1] - pts[q]))) for q in range(nloc - 1)]

        corners = {}
        for q in range(1, nloc - 1):
            j = a + q
            phi = _signed_turn(tang[j - 1], tang[j])
            if abs(phi) < math.radians(0.5):   # (nearly) in line: a tag point, not a corner
                continue
            corners[q] = [radius_at(j), phi]
        count = [0] * (nloc - 1)
        for q in corners:
            count[q - 1] += 1
            count[q] += 1
        for q in list(corners):
            r_req, phi = corners[q]
            half = math.tan(abs(phi) / 2)
            t_max = min(seg_len[q - 1] / count[q - 1], seg_len[q] / count[q])
            r = r_req
            if r * half > t_max:
                r = t_max / half
                warnings.append(f"{name(ids[a + q])}: corner radius reduced {r_req:.2f} -> {r:.2f} m "
                                f"(legs too short for the requested curve).")
            if r < perf.min_turn_radius - 1e-9:
                warnings.append(f"{name(ids[a + q])}: curve radius {r:.2f} m is below "
                                f"min_turn_radius {perf.min_turn_radius:.2f} m.")
            if d > 0 and r < d:
                warnings.append(f"{name(ids[a + q])}: curve radius {r:.2f} m is smaller than the tracking-sensor "
                                f"offset {d:.2f} m -> the drive axle cuts deep inside this curve.")
            corners[q] = (r, phi, r * half, r_req)

        smp = _Sampler(pts[0], math.atan2(tang[a][1], tang[a][0]), ds, a)
        arc_idx = []
        for q in range(1, nloc - 1):
            j = a + q
            if q not in corners:
                smp.line(pts[q], j - 1)
                continue
            r, phi, t, r_req = corners[q]
            d1, d2 = tang[j - 1], tang[j]
            T1 = pts[q] - t * d1
            T2 = pts[q] + t * d2
            smp.line(T1, j - 1)
            i_start = smp.n - 1
            c = T1 + r * math.copysign(1.0, phi) * _left(d1)
            a0 = math.atan2(T1[1] - c[1], T1[0] - c[0])
            smp.arc(c, r, a0, phi, j - 1, j)
            smp.x[-1], smp.y[-1] = float(T2[0]), float(T2[1])
            arc_idx.append((i_start, smp.n - 1, r, phi, c, j, r_req))
        smp.line(pts[-1], b - 1)
        x, y, h, k, s, la, lb = smp.arrays()
        vkey = "v_straight" if direction > 0 else "v_reverse"
        leg_v = np.array([min(legmode[q][vkey], caps[q]) for q in range(nleg)])
        leg_vc = np.array([legmode[q]["v_curve"] for q in range(nleg)])
        leg_acc = np.array([legmode[q]["accel"] for q in range(nleg)])
        leg_dec = np.array([legmode[q]["decel"] for q in range(nleg)])
        cap = np.minimum(np.array(caps)[la], np.array(caps)[lb])
        used = sorted({legmode[q]["name"] for q in range(a, b)})

        arcs = [Arc(radius=r, turn=phi, center=(float(c[0]), float(c[1])),
                    s_start=float(s[i0]), s_end=float(s[i1]), waypoint=ids[j],
                    label=name(ids[j]), requested_radius=r_req)
                for (i0, i1, r, phi, c, j, r_req) in arc_idx]
        closing = route.loop and b == m - 1
        runs.append(Run(
            index=ri, direction=direction, start_wp=ids[a], end_wp=ids[b],
            label=f"{name(ids[a])} -> {name(ids[b])}" + (" (reverse)" if direction < 0 else ""),
            x=x, y=y, heading=h, kappa=k, s=s, vcap=cap, arcs=arcs, track_offset=d,
            end_dwell=0.0 if closing else float(wps[ids[b]].dwell),
            end_name=name(ids[b]), end_pivot=end_pivot[b],
            vmax=np.minimum(leg_v[la], leg_v[lb]), vcurve=np.minimum(leg_vc[la], leg_vc[lb]),
            acc=leg_acc[la], dec=np.minimum(leg_dec[la], leg_dec[lb]), grade=np.array(grades)[la],
            modes=", ".join(used),
            end_cart="" if closing else cart_event(ids[b]),
            end_cart_mass=None if closing else cart_mass(ids[b])))

    d0 = agv.tracking_offset(dirs[0])
    start_pivot = route.loop and end_pivot[m - 1]
    axle0 = P[0] - d0 * tang[0] if (d0 > 0 and not start_pivot) else P[0]
    return RoutePlan(runs=runs, loop=route.loop, start_heading=leg_heading(0),
                     start_axle=(float(axle0[0]), float(axle0[1])),
                     start_dwell=float(wps[0].dwell), start_name=name(0), warnings=warnings,
                     start_cart=cart_event(0), start_cart_mass=cart_mass(0))
