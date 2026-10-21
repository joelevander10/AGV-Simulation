"""Auto route: plan an efficient tape route on the floor plan and tune it until the simulated AGV
(your PID, closed loop) stays on the tape and its swept area (AGV + trolleys) stays clear of obstacles.

Two modes
  plan : find the path between your stations yourself (start, every stop, every NAMED point, end),
         shortest + fewest corners, keeping the AGV/trolley half-width + margin away from obstacles
  keep : keep the points you drew (corners and stops), only re-do radii and speed tags

Then, for both
  radii   : every corner gets the largest radius that fits the legs (up to max_radius) and not
            below the smallest radius the AGV can follow
  tags    : speed-mode points are placed on the straights like RFID tags: after a curve (once the
            trolley train is out of it) switch to the fastest mode that still has room to brake,
            and switch back to the curve mode far enough before the next curve / stop
  verify  : simulate; corners with collisions get a smaller radius, corners where the tape error
            gets close to the sensor edge get an earlier slow-down, a larger radius or APPROACH
            speed; repeat until clean (or max_iter)
"""
from __future__ import annotations

import copy
import heapq
import math
from typing import List, Optional

import numpy as np

from .config import AGVConfig, RouteConfig, Waypoint
from .factory_map import FactoryMap

RADII = [3.0, 2.5, 2.0, 1.75, 1.5, 1.25, 1.0, 0.75]


# ============================================================================ planning grid
class Grid:
    def __init__(self, fmap: FactoryMap, max_cells=450_000):
        self.fmap = fmap
        mpp = fmap.mpp
        area = fmap.width_m * fmap.height_m
        target = max(0.10, math.sqrt(area / max_cells))
        self.f = max(1, int(math.ceil(target / mpp)))
        self.g = self.f * mpp
        f = self.f
        H = int(math.ceil(fmap.h / f)) * f
        W = int(math.ceil(fmap.w / f)) * f
        ob = np.ones((H, W), bool)
        ob[:fmap.h, :fmap.w] = fmap.obstacle
        self.obst = ob.reshape(H // f, f, W // f, f).any(axis=(1, 3))
        self.H, self.W = self.obst.shape
        self.obst[0, :] = self.obst[-1, :] = True
        self.obst[:, 0] = self.obst[:, -1] = True
        self.dist = self._distance(self.obst, 40)

    @staticmethod
    def _dilate(m, diag):
        o = m.copy()
        o[1:, :] |= m[:-1, :]
        o[:-1, :] |= m[1:, :]
        o[:, 1:] |= m[:, :-1]
        o[:, :-1] |= m[:, 1:]
        if diag:
            o[1:, 1:] |= m[:-1, :-1]
            o[1:, :-1] |= m[:-1, 1:]
            o[:-1, 1:] |= m[1:, :-1]
            o[:-1, :-1] |= m[1:, 1:]
        return o

    def _distance(self, obst, kmax):
        """Approximate distance to the nearest obstacle in cells (octagonal metric), capped."""
        d = np.full(obst.shape, float(kmax), np.float32)
        m = obst.copy()
        d[m] = 0.0
        for k in range(1, kmax):
            m2 = self._dilate(m, diag=(k % 2 == 0))
            new = m2 & ~m
            if not new.any():
                break
            d[new] = k
            m = m2
        return d

    def free_mask(self, clearance_m):
        k = clearance_m / self.g
        return self.dist > k

    def cell(self, x, y):
        col = int(x / self.fmap.mpp) // self.f
        row = int(self.fmap.h - 1 - math.floor(y / self.fmap.mpp)) // self.f
        return min(max(row, 0), self.H - 1), min(max(col, 0), self.W - 1)

    def xy(self, r, c):
        x = (c + 0.5) * self.g
        y = self.fmap.height_m - (r + 0.5) * self.g
        return x, y


DIRS4 = [(0, 1), (1, 0), (0, -1), (-1, 0)]          # E, S, W, N  (grid rows grow downwards)


def _astar4(free, cost, start, goal, turn_pen, start_dir=None, k_turn=1, k_start=0, k_goal=0,
            turn_ok=None, max_expand=6_000_000):
    """Orthogonal A* over (cell, heading, straight run so far).

    * every 90 deg turn costs turn_pen, U-turns are not allowed
    * a turn is only allowed after k_turn straight cells (room for two fillets)
    * with start_dir the AGV leaves the station straight on for at least k_start cells
    * the last k_goal cells before the goal are straight (no corner right at a stop)
    Returns the list of (row, col, heading) or None."""
    H, W = free.shape
    sr, sc = start
    gr, gc = goal
    KM = max(k_turn, k_goal, 1)
    freef = free.ravel().tolist()
    costf = cost.ravel().tolist()
    turnf = turn_ok.ravel().tolist() if turn_ok is not None else None
    best = {}
    parent = {}
    heap = []
    if start_dir is None:
        inits = [(d, KM) for d in range(4)]
    else:
        inits = [(start_dir, 0)]
    kst = k_start if start_dir is not None else 0
    for d, run in inits:
        st = ((sr * W + sc) * 4 + d) * (KM + 1) + run
        best[st] = 0.0
        heapq.heappush(heap, (abs(sr - gr) + abs(sc - gc), 0.0, sr, sc, d, run, 0))
    n = 0
    while heap:
        f, g, r, c, d, run, steps = heapq.heappop(heap)
        st = ((r * W + c) * 4 + d) * (KM + 1) + run
        if g > best.get(st, math.inf):
            continue
        if r == gr and c == gc and run >= min(k_goal, steps):
            path = []
            while True:
                cell = st // ((KM + 1) * 4)
                dd = (st // (KM + 1)) % 4
                path.append((cell // W, cell % W, dd))
                if st not in parent:
                    break
                st = parent[st]
            return path[::-1]
        n += 1
        if n > max_expand:
            return None
        can_turn = run >= k_turn and steps >= kst and (turnf is None or turnf[r * W + c])
        for nd in ((d, (d + 1) % 4, (d + 3) % 4) if can_turn else (d,)):
            dr, dc = DIRS4[nd]
            nr, nc = r + dr, c + dc
            if nr < 0 or nc < 0 or nr >= H or nc >= W:
                continue
            k = nr * W + nc
            if not freef[k]:
                continue
            nrun = min(KM, run + 1) if nd == d else 1
            ng = g + costf[k] + (turn_pen if nd != d else 0.0)
            ns = (k * 4 + nd) * (KM + 1) + nrun
            if ng < best.get(ns, math.inf):
                best[ns] = ng
                parent[ns] = st
                heapq.heappush(heap, (ng + abs(nr - gr) + abs(nc - gc), ng, nr, nc, nd, nrun, steps + 1))
    return None


def _astar8(free, cost, start, goal, max_expand=4_000_000):
    H, W = free.shape
    sr, sc = start
    gr, gc = goal
    freef = free.ravel().tolist()
    costf = cost.ravel().tolist()
    moves = [(0, 1, 1.0), (1, 0, 1.0), (0, -1, 1.0), (-1, 0, 1.0),
             (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142)]
    best = {sr * W + sc: 0.0}
    parent = {}

    def h(r, c):
        dr, dc = abs(r - gr), abs(c - gc)
        return max(dr, dc) + 0.4142 * min(dr, dc)
    heap = [(h(sr, sc), 0.0, sr, sc)]
    n = 0
    while heap:
        f, g, r, c = heapq.heappop(heap)
        s = r * W + c
        if g > best.get(s, math.inf):
            continue
        if r == gr and c == gc:
            path = [(r, c)]
            while s in parent:
                s = parent[s]
                path.append(divmod(s, W))
            return path[::-1]
        n += 1
        if n > max_expand:
            return None
        for dr, dc, w in moves:
            nr, nc = r + dr, c + dc
            if nr < 0 or nc < 0 or nr >= H or nc >= W:
                continue
            k = nr * W + nc
            if not freef[k] or (dr and dc and not (freef[r * W + nc] and freef[nr * W + c])):
                continue
            ng = g + w * costf[k]
            if ng < best.get(k, math.inf):
                best[k] = ng
                parent[k] = s
                heapq.heappush(heap, (ng + h(nr, nc), ng, nr, nc))
    return None


def _line_free(free, a, b):
    (r0, c0), (r1, c1) = a, b
    n = int(max(abs(r1 - r0), abs(c1 - c0)) * 2) + 1
    rr = np.round(np.linspace(r0, r1, n)).astype(int)
    cc = np.round(np.linspace(c0, c1, n)).astype(int)
    return bool(free[rr, cc].all())


def _relax(free, grid: Grid, xy, radius_m):
    """Let the path reach a station that sits closer to equipment than the clearance: open the
    raw free cells within radius of it."""
    r, c = grid.cell(*xy)
    k = int(math.ceil(radius_m / grid.g))
    r0, r1, c0, c1 = max(0, r - k), min(grid.H, r + k + 1), max(0, c - k), min(grid.W, c + k + 1)
    rr, cc = np.mgrid[r0:r1, c0:c1]
    near = (rr - r) ** 2 + (cc - c) ** 2 <= k * k
    sub = free[r0:r1, c0:c1]
    sub[near & ~grid.obst[r0:r1, c0:c1]] = True
    return (r, c)


# ============================================================================ geometry helpers
def _turn(a, b, c):
    d1 = np.subtract(b, a)
    d2 = np.subtract(c, b)
    n1, n2 = np.hypot(*d1), np.hypot(*d2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    return math.atan2(d1[0] * d2[1] - d1[1] * d2[0], d1[0] * d2[0] + d1[1] * d2[1])


class _Pt:
    """A point of the route skeleton (station or corner)."""

    def __init__(self, x, y, wp: Optional[Waypoint] = None, corner=False):
        self.x, self.y = float(x), float(y)
        self.wp = wp                     # original station waypoint (keeps name / stop / dwell / cart)
        self.corner = corner
        self.radius = 0.0
        self.curve_mode = None           # mode to run this corner in (None -> default)
        self.slow_extra = 0.0            # extra slow-down distance found by verification
        self.flag_collide = 0
        self.exit_extra = 0.0            # extra straight after the curve before speeding up (settling)
        self.radius_tried = set()

    @property
    def is_stop(self):
        return self.wp is not None and ((self.wp.action or "pass") in ("stop", "pivot") or bool(self.wp.cart)
                                         or (self.wp.direction or "forward").startswith("r"))


def _is_station(wp: Waypoint, i: int, n: int) -> bool:
    return i == 0 or i == n - 1 or (wp.action or "pass") in ("stop", "pivot") or bool(wp.cart) or bool(wp.name) \
        or (wp.direction or "forward").startswith("r")


def _branch_blocks(route):
    """Reverse branches (Route tab tool 'Reverse branch', or drawn by hand): [(a, b, j)] with a = the
    point where the route switches to reverse (REVERSE tag), b = the forward point back on the junction
    (or the end of the branch when there is none) and j = the junction point of the reverse run.
    Indices may run past the end of a loop (use % n). The auto route keeps these parts as drawn."""
    wps = route.waypoints
    n = len(wps)
    if n < 3:
        return []

    def rev(i):
        if not route.loop and not 0 <= i < n:
            return False
        return str(wps[i % n].direction or "forward")[:1].lower() == "r"
    out = []
    for a in range(n):
        if not rev(a) or (route.loop or a > 0) and rev(a - 1):
            continue
        e = a
        while rev(e) and e < a + n:
            e += 1
        if not route.loop and e >= n:
            e = n - 1
        b = e
        j = a + 1 if a + 1 < e else None
        if j is not None:
            for k in range(e + 1, e + 16):
                if not route.loop and k >= n:
                    break
                w, J = wps[k % n], wps[j % n]
                if math.hypot(w.x - J.x, w.y - J.y) < 0.3:    # back on the junction (forward)
                    b = k
                    break
        out.append((a, b, j))
    return out


# ============================================================================ main class
class AutoRoute:
    def __init__(self, agv: AGVConfig, route: RouteConfig, fmap: FactoryMap, mode="plan", style="orthogonal",
                 margin: Optional[float] = None, max_radius=2.5, min_radius: Optional[float] = None,
                 fast_modes: Optional[List[str]] = None, curve_mode="SLOW", max_iter=8, log=None,
                 tape_limit_pct: float = 70.0, trolley_safe: bool = True, push_allow: Optional[float] = None,
                 shared_tags: bool = True):
        self.agv, self.route, self.fmap = agv, route, fmap
        self.mode, self.style = mode, style
        n_ = len(route.waypoints)
        self.branches = _branch_blocks(route)
        self.locked, self.lock_inner = set(), set()
        for (a, b, _j) in self.branches:
            self.locked.update(i % n_ for i in range(a, b + 1))
            self.lock_inner.update(i % n_ for i in range(a, b))
        self.margin = agv.simulation.safety_margin if margin is None else float(margin)
        self.max_radius = float(max_radius)
        self.min_radius = float(min_radius) if min_radius else 0.75
        self.curve_mode = curve_mode
        self.max_iter = int(max_iter)
        self.exit_after = 0.3            # m after the curve before speeding up
        self.brake_margin = 0.3          # m: reach the curve speed this far before the curve
        self.tape_limit = max(0.2, min(1.0, tape_limit_pct / 100.0))   # share of the sensor range the tape error may use
        self.straight_min = max(self.min_radius + 1.0, 2.0)   # m of straight at every stop
        self.extra_clear = 0.0           # grows when the trolleys hit things next to the lane
        self.soft_blocks = []            # (x, y, r): spots the planned lane must keep away from
        self.hook_type_tow = agv.hook_type == "tow"
        n_tr = len(agv.train_masses()) if agv.hook_type == "tow" else 0
        self.turn_extra_step = 0.15 + 0.15 * n_tr   # extra corner room added after a collision near a corner
        self.turn_extra = 0.0
        self.replans = 0
        # trolley-safe slowing down: the trolleys have no brakes - slowing faster than their own rolling
        # friction lets them push the AGV. Allowed push (N) default = 10 % of the AGV weight.
        self.trolley_safe = bool(trolley_safe) and agv.hook_type == "tow"
        self.push_allow = float(push_allow) if push_allow is not None else 0.10 * agv.chassis.mass_kg * 9.81
        self.crr_c = agv.cart.rolling_resistance
        self.closed = bool(agv.line_follow.enabled)
        self.brake_decel = agv.line_follow.brake_decel if self.closed else agv.performance.decel
        self.push_notes = {}
        self.shared_tags = bool(shared_tags)
        self.fast_min_len = {}                 # mode -> shortest straight it may be used on (target search)
        self._shared_log = ""
        self.logs: List[str] = []
        self._log_cb = log
        # mode speeds (closed loop uses line_follow + table; open loop uses the table)
        lf = agv.line_follow
        self.speeds = {}
        for m in agv.speed_modes:
            self.speeds[m.name.strip().upper()] = (m.v_straight, m.accel, m.decel)
        if lf.enabled:                     # like the closed-loop controller drives them (modes.py)
            keys = list(self.speeds) + ["HIGH", "SLOW", "APPROACH"]
            keys += ["MAX"] if lf.speed_max is not None else []
            keys += ["MED"] if lf.speed_med is not None and "MEDIUM" not in self.speeds else []
            for k in dict.fromkeys(keys):
                if lf.mode_key(k):
                    v, a_, d_ = agv.lf_mode(k)
                    self.speeds[k] = (min(v, agv.max_wheel_speed), a_, d_)
        # ramps with the load (trolleys speed up / slow down slower, drive.py)
        from .drive import DriveLimits
        m_tow = sum(agv.train_masses()) if agv.hook_type == "tow" else 0.0
        m_lurk = agv.cart.mass_kg if agv.hook_type == "lurking" else 0.0
        st = DriveLimits(agv).state(m_tow, m_lurk, 0.0)
        a_lim, d_lim = max(st["a_max"], 0.01), max(st["d_max"], 0.01)
        if agv.simulation.load_effects:
            self.speeds = {k: (v, min(a, a_lim), min(d, d_lim)) for k, (v, a, d) in self.speeds.items()}
        if curve_mode.upper() not in self.speeds:
            # no such mode on this AGV: slow mode if there is one, else the plain performance
            # section (its curve speed is applied at every curve anyway)
            if "SLOW" in self.speeds:
                self.curve_mode = "SLOW"
            elif self.speeds:
                self.curve_mode = min(self.speeds, key=lambda m: self.speeds[m][0])
            else:
                p = agv.performance
                self.speeds["DEFAULT"] = (p.v_straight, p.accel, p.decel)
                self.curve_mode = "DEFAULT"
        else:
            self.curve_mode = curve_mode.upper()
        fm = fast_modes or [k for k in self.speeds if k not in (self.curve_mode, "APPROACH")]
        self.fast_modes = sorted([m.upper() for m in fm if m.upper() in self.speeds],
                                 key=lambda m: -self.speeds[m][0])
        # train length behind the AGV (to leave a curve before speeding up)
        c = agv.cart
        n_tr = len(agv.train_masses()) if agv.hook_type == "tow" else 0
        self.train_len = (abs(agv.hook.x) + n_tr * (c.tongue_length + c.coupling_behind_axle)) if n_tr else 0.0
        half = agv.chassis.width / 2
        if agv.hook_type in ("tow", "lurking"):
            half = max(half, c.width / 2)
        self.half_width = half

    def log(self, msg):
        self.logs.append(msg)
        if self._log_cb:
            self._log_cb(msg)

    # ------------------------------------------------------------------ skeleton
    def _lock_pt(self, i, x=None, y=None, inner=None, wp=None):
        """A point of a reverse branch: kept exactly as drawn (position, radius, direction, mode, stop)."""
        wp = wp if wp is not None else self.route.waypoints[i]
        p = _Pt(wp.x if x is None else x, wp.y if y is None else y, wp, corner=False)
        p.lock = wp
        p.lock_inner = (i in self.lock_inner) if inner is None else inner
        return p

    def skeleton_keep(self) -> List[_Pt]:
        wps = self.route.waypoints
        n = len(wps)
        pts: List[_Pt] = []
        for i, wp in enumerate(wps):
            if i in self.locked:
                pts.append(self._lock_pt(i))
                continue
            keep = _is_station(wp, i, n)
            if not keep and 0 < i < n - 1:
                phi = _turn((wps[i - 1].x, wps[i - 1].y), (wp.x, wp.y), (wps[i + 1].x, wps[i + 1].y))
                keep = abs(phi) > math.radians(3)
            elif not keep:
                keep = True
            if keep:
                p = _Pt(wp.x, wp.y, wp if _is_station(wp, i, n) else None, corner=not _is_station(wp, i, n))
                if pts and math.hypot(p.x - pts[-1].x, p.y - pts[-1].y) < 0.05:
                    continue
                pts.append(p)
        self.log(f"Kept {len(pts)} of your {n} points (tags and in-line points are re-made).")
        if self.branches:
            self.log(f"{len(self.branches)} reverse branch(es) kept exactly as drawn (REVERSE tag, curves, stops).")
        return pts

    def _plan_stations(self):
        """Stations for planning. A reverse branch is a fixed chain: an approach point on the junction's
        line before the REVERSE tag (so the planned lane arrives along the right tape), the branch as
        drawn, and an exit point on the same line after it."""
        wps = self.route.waypoints
        n = len(wps)
        out = []
        starts = {a % n: (a, b, j) for (a, b, j) in self.branches}
        i = 0
        while i < n:
            if i in starts:
                a, b, j = starts[i]
                chain = list(range(a, b + 1))
                R = wps[a % n]
                if j is not None:
                    J = wps[j % n]
                    L = math.hypot(R.x - J.x, R.y - J.y)
                    if L > 1e-6:
                        ux, uy = (R.x - J.x) / L, (R.y - J.y) / L
                        pre = Waypoint(x=round(J.x - ux * 2.0, 3), y=round(J.y - uy * 2.0, 3), mode="")
                        pre._inner, pre._lock = True, None
                        out.append(pre)
                for k in chain:
                    w = copy.copy(wps[k % n])
                    w._inner, w._lock = (k < b), wps[k % n]
                    out.append(w)
                if j is not None and L > 1e-6:
                    out[-1]._inner = True
                    post = Waypoint(x=round(R.x + ux * 1.0, 3), y=round(R.y + uy * 1.0, 3), mode="")
                    post._inner, post._lock = False, None
                    out.append(post)
                if b >= n:                                  # the branch wraps over the start of the loop
                    raise ValueError("A reverse branch runs over the start point of the route - move the start "
                                     "(HOME) out of the branch, or use 'Keep my path'.")
                i = b + 1
                continue
            if _is_station(wps[i], i, n):
                w = wps[i]
                out.append(w)
            i += 1
        return out

    def _station_pt(self, w):
        lk = getattr(w, "_lock", None)
        if lk is not None:
            p = _Pt(w.x, w.y, lk, corner=False)
            p.lock, p.lock_inner = lk, bool(getattr(w, "_inner", False))
            return p
        if getattr(w, "_inner", None) is not None:      # approach / exit point of a branch: a plain corner
            return _Pt(w.x, w.y, None, corner=True)
        return _Pt(w.x, w.y, w)

    def skeleton_plan(self) -> List[_Pt]:
        wps = self.route.waypoints
        n = len(wps)
        stations = self._plan_stations()
        if self.route.loop and len(stations) > 1:
            stations = stations + [stations[0]]
        grid = getattr(self, "grid", None) or Grid(self.fmap)
        self.grid = grid
        clearance = self.half_width + self.margin + self.extra_clear
        self.clearance = clearance
        if not self.replans:
            self.log(f"Planning grid {grid.W} x {grid.H} cells of {grid.g * 100:.0f} cm; keeping "
                     f"{clearance:.2f} m (half width {self.half_width:.2f} + margin {self.margin:.2f}) from obstacles.")
        pts: List[_Pt] = [self._station_pt(stations[0])]
        arrive_dir = None
        for k in range(len(stations) - 1):
            a, b = stations[k], stations[k + 1]
            if getattr(a, "_inner", False):              # inside a reverse branch: as drawn, no planning
                pts.append(self._station_pt(b))
                arrive_dir = None
                continue
            free = grid.free_mask(clearance)
            for (bx, by, br) in self.soft_blocks:
                r0, c0 = grid.cell(bx, by)
                k = int(math.ceil(br / grid.g))
                rr, cc = np.ogrid[max(0, r0 - k):min(grid.H, r0 + k + 1), max(0, c0 - k):min(grid.W, c0 + k + 1)]
                disc = (rr - r0) ** 2 + (cc - c0) ** 2 <= k * k
                free[max(0, r0 - k):min(grid.H, r0 + k + 1), max(0, c0 - k):min(grid.W, c0 + k + 1)][disc] = False
            s_cell = _relax(free, grid, (a.x, a.y), clearance + 1.0)
            g_cell = _relax(free, grid, (b.x, b.y), clearance + 1.0)
            # prefer the middle of the aisles: extra cost close to obstacles
            pref = (clearance + 1.0) / grid.g
            cost = grid.g * (1.0 + 0.8 * np.clip((pref - grid.dist) / pref, 0.0, 1.0))
            cost = cost / grid.g                  # in cells
            turn_pen = 8.0 / grid.g               # a corner costs about 8 m of straight driving
            if self.style == "any":
                cells = _astar8(free, cost, s_cell, g_cell)
                if cells is None:
                    raise ValueError(f"No free path from {a.name or 'point %d' % k} to {b.name or 'point %d' % (k + 1)} "
                                     f"with {clearance:.2f} m clearance. Lower the margin or check the floor plan.")
                # string pulling
                keep = [cells[0]]
                i = 0
                while i < len(cells) - 1:
                    j = len(cells) - 1
                    while j > i + 1 and not _line_free(free, cells[i], cells[j]):
                        j -= 1
                    keep.append(cells[j])
                    i = j
                corners = [grid.xy(*c) for c in keep[1:-1]]
                for x, y in corners:
                    pts.append(_Pt(x, y, corner=True))
            else:
                K = max(2, int(math.ceil(self.straight_min / grid.g)))
                cells = self._plan4(free, cost, s_cell, g_cell, turn_pen, arrive_dir, K)
                if cells is None and arrive_dir is not None:
                    self.log(f"  {a.name or 'stop'}: cannot leave straight ahead - the AGV has to turn on the spot "
                             f"(or reverse) there.")
                    cells = self._plan4(free, cost, s_cell, g_cell, turn_pen, None, K)
                if cells is None:
                    raise ValueError(f"No free path from {a.name or 'point %d' % k} to {b.name or 'point %d' % (k + 1)} "
                                     f"with {clearance:.2f} m clearance. Lower the margin or check the floor plan.")
                # corners where the heading changes, snapped so every leg is exactly straight
                seq = []
                for i in range(1, len(cells)):
                    if cells[i][2] != cells[i - 1][2]:
                        seq.append((cells[i - 1][0], cells[i - 1][1], cells[i - 1][2], cells[i][2]))
                prev = (a.x, a.y)
                corner_pts = []
                for (r, c, d_in, d_out) in seq:
                    x, y = grid.xy(r, c)
                    if d_in in (0, 2):          # arriving horizontally: same y as the previous point
                        y = prev[1]
                    else:
                        x = prev[0]
                    corner_pts.append([x, y, d_in, d_out])
                    prev = (x, y)
                if corner_pts:                  # last leg must hit the station exactly
                    last = corner_pts[-1]
                    if last[3] in (0, 2):
                        last[1] = b.y
                    else:
                        last[0] = b.x
                    if len(corner_pts) >= 2:    # keep the leg before straight too
                        pr = corner_pts[-2]
                        if last[2] in (0, 2):
                            pr[1] = last[1]
                        else:
                            pr[0] = last[0]
                for x, y, _, _ in corner_pts:
                    pts.append(_Pt(x, y, corner=True))
                arrive_dir = cells[-1][2]
            pts.append(self._station_pt(b))
            if not self.replans:
                self.log(f"  {a.name or 'start'} -> {b.name or 'next stop'}: {len(pts) - 1} points so far")
        if self.route.loop and len(pts) > 1:
            pts = pts[:-1]                      # the loop closes back to the first point
        # drop corners that ended up (almost) collinear or duplicated after snapping
        clean = [pts[0]]
        for i in range(1, len(pts)):
            p = pts[i]
            if math.hypot(p.x - clean[-1].x, p.y - clean[-1].y) < 0.05 and p.wp is None:
                continue
            clean.append(p)
        out = [clean[0]]
        for i in range(1, len(clean) - 1):
            p = clean[i]
            if p.corner and abs(_turn((out[-1].x, out[-1].y), (p.x, p.y), (clean[i + 1].x, clean[i + 1].y))) < math.radians(2):
                continue
            out.append(p)
        out.append(clean[-1])
        return out

    def _plan4(self, free, cost, s_cell, g_cell, turn_pen, start_dir, K):
        g = self.grid.g
        k_turn = min(16, max(1, int(math.ceil(2 * self.min_radius / g))))
        # corners need more room than straights: the fillet cuts inside and the trolleys
        # cut inside even more (off-tracking)
        turn_ok = None
        if self.turn_extra > 0:
            turn_ok = free & (self.grid.dist > (self.clearance + self.turn_extra) / g)
        r = _astar4(free, cost, s_cell, g_cell, turn_pen, start_dir=start_dir, k_turn=k_turn,
                    k_start=min(16, K), k_goal=min(16, K), turn_ok=turn_ok)
        if r is None and turn_ok is not None:
            r = _astar4(free, cost, s_cell, g_cell, turn_pen, start_dir=start_dir, k_turn=k_turn,
                        k_start=min(16, K), k_goal=min(16, K))
        return r

    def _seg_free(self, free, a, b):
        return _line_free(free, self.grid.cell(*a), self.grid.cell(*b))

    def _push_legs(self, pts, collisions, step=0.25, max_shift=1.5):
        """Move the corner-to-corner leg next to each hit spot away from it (both corners together,
        so the legs stay straight), if the floor there is free. Trolleys cut inside corners;
        moving the lane outwards is the direct cure."""
        if len(pts) < 4:
            return 0
        n = len(pts)
        loop = self.route.loop
        free = self.grid.free_mask(self.half_width + self.margin)
        moved = 0
        done = set()
        for c in collisions:
            cx, cy = c["x"], c["y"]
            best, bd = None, 3.0
            for i in range(n if loop else n - 1):
                a, b = pts[i], pts[(i + 1) % n]
                if not (a.corner and b.corner):
                    continue
                vx, vy = b.x - a.x, b.y - a.y
                L2 = vx * vx + vy * vy
                if L2 < 1e-9:
                    continue
                t = max(0.0, min(1.0, ((cx - a.x) * vx + (cy - a.y) * vy) / L2))
                d = math.hypot(a.x + t * vx - cx, a.y + t * vy - cy)
                if d < bd:
                    bd, best = d, i
            if best is None or best in done:
                continue
            i = best
            a, b = pts[i], pts[(i + 1) % n]
            L = math.hypot(b.x - a.x, b.y - a.y)
            nx, ny = -(b.y - a.y) / L, (b.x - a.x) / L
            if nx * (a.x - cx) + ny * (a.y - cy) < 0:
                nx, ny = -nx, -ny                         # away from the hit spot
            shift = getattr(a, "pushed", 0.0)
            if shift >= max_shift - 1e-6:
                continue
            p0, p3 = pts[(i - 1) % n], pts[(i + 2) % n]
            na = (a.x + nx * step, a.y + ny * step)
            nb = (b.x + nx * step, b.y + ny * step)
            if not (self._seg_free(free, (p0.x, p0.y), na) and self._seg_free(free, na, nb)
                    and self._seg_free(free, nb, (p3.x, p3.y))):
                continue
            # the neighbouring legs must stay straight too: only move along them
            ok = True
            for (q, nq) in ((p0, na), (p3, nb)):
                if abs((nq[0] - q.x) * ny - (nq[1] - q.y) * nx) < 1e-6 and q.wp is None:
                    pass
            if not ok:
                continue
            a.x, a.y = na
            b.x, b.y = nb
            a.pushed = b.pushed = shift + step
            done.add(i)
            moved += 1
        return moved

    # ------------------------------------------------------------------ radii + tags
    def _decel(self, m):
        return self.speeds.get(m, self.speeds[self.curve_mode])[2]

    def _ramp(self, m_from, m_to):
        """Deceleration used when the speed mode changes m_from -> m_to (closed loop: the controller
        ramps down at the rate of the TARGET mode; kinematic model: the gentler of the two)."""
        if self.closed:
            return self.speeds[m_to][2]
        return min(self._decel(m_from), self._decel(m_to))

    def _brake_dist(self, m_from, m_to):
        v1 = self.speeds[m_from][0]
        v0 = self.speeds[m_to][0]
        return max(0.0, (v1 * v1 - v0 * v0) / (2 * max(self._ramp(m_from, m_to), 1e-3)))

    def _a_safe(self, m_tow):
        """Gentlest-needed deceleration so the trolleys push the AGV with at most push_allow N."""
        if not self.trolley_safe or m_tow <= 1e-6:
            return math.inf
        return self.crr_c * 9.81 + self.push_allow / m_tow

    def _slowdown(self, m_from, m_to, m_tow):
        """Tags to go from m_from down to m_to: [(mode, metres before the point where m_to must be
        reached)], far to near. With heavy trolleys a gentler intermediate mode is put in between
        when the direct ramp would let the trolleys push."""
        v_from, v_to = self.speeds[m_from][0], self.speeds[m_to][0]
        d = self._brake_dist(m_from, m_to)
        if v_to >= v_from - 1e-6:
            return [(m_to, 0.0)], 0.0
        a_safe = self._a_safe(m_tow)
        steps = [(m_to, d)]
        r_to = self._ramp(m_from, m_to)
        if r_to > a_safe + 1e-6:
            best = None
            for m, (vm, _, _) in self.speeds.items():
                if m in (m_from, m_to) or not (v_to + 0.02 < vm < v_from - 0.02):
                    continue
                r_m = self._ramp(m_from, m)
                if r_m >= r_to - 1e-6:
                    continue
                # prefer a gentle mode as close to the final speed as possible (small steep last step)
                key = (0 if r_m <= a_safe + 1e-6 else 1, vm if r_m <= a_safe + 1e-6 else r_m)
                if best is None or key < best[2]:
                    best = (m, r_m, key)
            if best is not None:
                m, r_m, _ = best
                vm = self.speeds[m][0]
                d2 = (vm * vm - v_to * v_to) / (2 * max(r_to, 1e-3))
                d1 = (v_from * v_from - vm * vm) / (2 * max(r_m, 1e-3))
                steps = [(m, d1 + d2 + 0.2), (m_to, d2)]
                d = d1 + d2 + 0.2
                self._note_push(m_from, m, r_m, m_tow)
                self._note_push(m, m_to, r_to, m_tow, dv=(vm, v_to))
            else:
                self._note_push(m_from, m_to, r_to, m_tow)
        else:
            self._note_push(m_from, m_to, r_to, m_tow)
        return steps, d

    def _note_push(self, m_from, m_to, ramp, m_tow, dv=None):
        if m_tow <= 1e-6 or self.hook_type_tow is False:
            return
        push = max(0.0, m_tow * (ramp - self.crr_c * 9.81))
        k = (m_to, round(ramp, 3))
        old = self.push_notes.get(k)
        if old is None or push > old["push"]:
            self.push_notes[k] = {"to": m_to, "ramp": ramp, "push": push, "m": m_tow}

    def _leg_masses(self, pts):
        """kg of trolleys on the leg leaving each point (tow only)."""
        n = len(pts)
        if self.agv.hook_type != "tow":
            return [0.0] * n
        evs = [(p.wp.cart or "").strip().lower() if p.wp is not None else "" for p in pts]
        first = next((e for e in evs if e), "")
        m = 0.0 if first == "attach" else self.agv.train_mass(getattr(self.route, "start_load", "full"))
        out = []
        for p, e in zip(pts, evs):
            if e == "detach":
                m = 0.0
            elif e in ("attach", "swap"):
                w = p.wp
                m = float(w.cart_mass) if w.cart_mass is not None else self.agv.train_mass(w.cart_load or "full")
            out.append(m)
        return out

    def _acc_dist(self, m_from, m_to):
        v0 = self.speeds[m_from][0]
        v1, a, _ = self.speeds[m_to]
        return max(0.0, (v1 * v1 - v0 * v0) / (2 * max(a, 1e-3)))

    def fit_radii(self, pts: List[_Pt], loop: bool):
        n = len(pts)
        for i, p in enumerate(pts):
            end = (not loop) and (i == 0 or i == n - 1)
            p.corner = (not p.is_stop) and not end
        idx = list(range(n))

        def nb(i, k):
            j = i + k
            if loop:
                return pts[j % n]
            return pts[j] if 0 <= j < n else None
        for i in idx:
            p = pts[i]
            a, c = nb(i, -1), nb(i, 1)
            if not p.corner or a is None or c is None:
                p.radius = 0.0
                p.phi = 0.0
                continue
            p.phi = _turn((a.x, a.y), (p.x, p.y), (c.x, c.y))
            half = math.tan(abs(p.phi) / 2)
            L1 = math.hypot(p.x - a.x, p.y - a.y) / (2 if a.corner else 1)
            L2 = math.hypot(c.x - p.x, c.y - p.y) / (2 if c.corner else 1)
            fit = min(L1, L2) / max(half, 1e-6)
            if not hasattr(p, "want"):
                p.want = self.max_radius
            p.radius = float(max(min(p.want, fit), 0.05))
            p.fit = fit
        for i in idx:                                    # reverse branches: radii as drawn
            p = pts[i]
            lk = getattr(p, "lock", None)
            if lk is None:
                continue
            a, c = nb(i, -1), nb(i, 1)
            stop = (lk.action or "pass") in ("stop", "pivot") or bool(lk.cart)
            p.phi = _turn((a.x, a.y), (p.x, p.y), (c.x, c.y)) if (a is not None and c is not None) else 0.0
            r_ = lk.radius if lk.radius is not None else self.route.default_radius
            p.corner = (not stop) and a is not None and c is not None and abs(p.phi) > math.radians(0.5) and r_ > 0
            p.radius = float(r_) if p.corner else 0.0
            p.fit = p.radius

    def build_route(self, pts: List[_Pt]) -> RouteConfig:
        loop = self.route.loop
        n = len(pts)
        wps: List[Waypoint] = []
        cm = self.curve_mode
        leg_m = self._leg_masses(pts)
        self.push_notes = {}
        app = "APPROACH" if ("APPROACH" in self.speeds and self.closed) else None
        for i in range(n):
            p = pts[i]
            lk = getattr(p, "lock", None)
            if lk is not None:
                w = copy.deepcopy(lk)                    # reverse branch: exactly as drawn
                w.x, w.y = round(p.x, 3), round(p.y, 3)
                for a_ in ("_inner", "_lock"):
                    if a_ in w.__dict__:
                        del w.__dict__[a_]
            elif p.wp is not None:
                w = copy.deepcopy(p.wp)
                w.x, w.y = p.x, p.y
                w.mode = ""
                w.radius = 0.0 if p.is_stop else None
                if not p.is_stop and p.corner:
                    w.radius = p.radius
            else:
                w = Waypoint(x=round(p.x, 3), y=round(p.y, 3), radius=round(p.radius, 3))
            wps.append(w)
            if lk is not None and getattr(p, "lock_inner", False):
                if not (loop or i + 1 < n):
                    break
                continue                                 # no speed tags inside the branch
            # leg to the next point
            q = pts[(i + 1) % n] if (loop or i + 1 < n) else None
            if q is None:
                break
            L = math.hypot(q.x - p.x, q.y - p.y)
            if L < 1e-6:
                continue
            ux, uy = (q.x - p.x) / L, (q.y - p.y) / L
            t_a = p.radius * math.tan(abs(getattr(p, "phi", 0.0)) / 2) if p.corner else 0.0
            t_b = q.radius * math.tan(abs(getattr(q, "phi", 0.0)) / 2) if q.corner else 0.0
            exit_d = (t_a + self.exit_after + p.exit_extra) if p.corner else 0.0
            end_mode = (q.curve_mode or cm) if q.corner else cm
            margin = self.brake_margin + q.slow_extra
            m_tow = leg_m[i]
            if self.trolley_safe and m_tow > 0 and q.is_stop and app and self.speeds[app][0] < self.speeds[cm][0]:
                # come into the stop slowly: the stop brake is hard and the trolleys have no brakes
                v_a = self.speeds[app][0]
                end_mode = app
                margin += 0.3 + v_a * v_a / (2 * max(self.brake_decel, 1e-3))
                k_ = ("STOP", round(self.brake_decel, 3))
                if k_ not in self.push_notes or m_tow > self.push_notes[k_]["m"]:
                    self.push_notes[k_] = {
                        "to": "STOP", "ramp": self.brake_decel, "m": m_tow, "v": v_a,
                        "push": max(0.0, m_tow * (self.brake_decel - self.crr_c * 9.81))}
            # base speed of this stretch: the curve mode, or between two faster curves the slower of them
            base = cm
            if p.corner and q.corner:
                mp, mq = (p.curve_mode or cm), (q.curve_mode or cm)
                slower = mp if self._v(mp) <= self._v(mq) else mq
                if self._v(slower) > self._v(cm) + 1e-6:
                    base = slower
            # pick the mode that drives this straight in the shortest time (incl. its ramps)
            vs = self.speeds[base][0]
            Lf = L - t_b - exit_d
            best_t = Lf / max(vs, 1e-3)
            chosen = None
            for m in self.fast_modes:
                vm, am, dm = self.speeds[m]
                if vm <= vs + 1e-6:
                    continue
                if Lf < self.fast_min_len.get(m, 0.0):
                    continue                            # (target cycle time: this mode only on long straights)
                da = self._acc_dist(base, m)
                db = self._slowdown(m, end_mode, m_tow)[1]
                cruise = Lf - da - db - margin
                if cruise < 0.3:
                    continue
                tm = (da / (0.5 * (vs + vm)) + cruise / vm + db / (0.5 * (vm + self.speeds[end_mode][0]))
                      + margin / max(self.speeds[end_mode][0], 1e-3))
                if tm < best_t - 0.2:
                    best_t, chosen = tm, m
            if chosen is None:
                # slow leg: run in the curve mode, APPROACH only where a corner needs it
                if not p.corner:
                    wps[-1].mode = base
                elif (p.curve_mode or cm) != base and exit_d < L - t_b - 0.3:
                    wps.append(Waypoint(x=round(p.x + ux * exit_d, 3), y=round(p.y + uy * exit_d, 3), mode=base))
                if end_mode != base:
                    steps, _ = self._slowdown(base, end_mode, m_tow)
                    for (mm, dd) in steps:
                        s_tag = L - t_b - margin - dd
                        s_tag = max(s_tag, (exit_d if p.corner else 0.0) + 0.05)
                        if s_tag < L - t_b:
                            wps.append(Waypoint(x=round(p.x + ux * s_tag, 3), y=round(p.y + uy * s_tag, 3), mode=mm))
                continue
            # fast leg: tag to the fast mode, then back to the curve / stop mode
            if p.corner:
                wps.append(Waypoint(x=round(p.x + ux * exit_d, 3), y=round(p.y + uy * exit_d, 3), mode=chosen))
            else:
                wps[-1].mode = chosen
            steps, _ = self._slowdown(chosen, end_mode, m_tow)
            for (mm, dd) in steps:
                s_slow = L - t_b - margin - dd
                wps.append(Waypoint(x=round(p.x + ux * s_slow, 3), y=round(p.y + uy * s_slow, 3), mode=mm))
        r = copy.deepcopy(self.route)
        r.waypoints = wps
        r.default_mode = cm
        r.name = (self.route.name or "route").replace(" (auto)", "") + " (auto)"
        if self.shared_tags:
            r = self._harmonize(r)
        return r

    # ------------------------------------------------------------------ shared tape (both directions)
    def _read_sequences(self, route):
        """Per forward run: design (own tags only) and actual (own + tags of other legs on the
        same tape) mode changes, as the simulation / the real RFID reader sees them."""
        from .linefollow import Tape, _tags_for_run
        from .path import build_plan
        plan = build_plan(route, self.agv)
        tape = Tape(plan)
        out = []
        for run in plan.runs:
            if run.direction < 0:
                continue
            nxt = plan.runs[run.index + 1] if run.index + 1 < len(plan.runs) else (plan.runs[0] if plan.loop else None)
            tags, _, foreign = _tags_for_run(route, plan, tape, run, nxt)
            fset = {(round(sf, 6), route.waypoints[wi].mode.strip().upper()): wi for sf, wi in foreign}
            seq = []
            for (st, kind, val) in tags:
                if kind != "mode":
                    continue
                wi = fset.get((round(st, 6), val))
                seq.append((st, val, wi))          # wi None = own tag
            end_s = max([t[0] for t in tags if t[1] == "stop"] or [0.0])
            lo = tape.range[run.index][0]
            out.append({"run": run, "seq": seq, "end": end_s, "off": float(tape.s[lo]) - float(run.s[0])})
        return out

    def _v(self, m):
        m = (m or "").upper()
        if m in self.speeds:
            return self.speeds[m][0]
        return self.speeds.get("HIGH", max(self.speeds.values()))[0]

    def _conflicts(self, route):
        """Over-speed: after a tag of another leg the AGV would run faster than this leg wants.
        Returns (overs: [(wi, run, mode_wanted)], unders: [(run, s_from, s_to, mode_wanted)])."""
        overs, unders = [], []
        carry = (route.default_mode or self.curve_mode).upper()
        for R in self._read_sequences(route):
            design = actual = carry
            last_foreign = None
            pts = sorted(R["seq"], key=lambda t: t[0]) + [(R["end"], None, "END")]
            s_prev = None
            for (st, val, wi) in pts:
                if s_prev is not None and st > s_prev + 1e-6:
                    if self._v(actual) > self._v(design) + 1e-6 and last_foreign is not None:
                        overs.append((last_foreign, R["run"], design))
                    elif self._v(actual) < self._v(design) - 1e-6 and last_foreign is not None:
                        unders.append((R["run"], s_prev - R["off"], st - R["off"], design, last_foreign))
                if wi == "END":
                    break
                if wi is None:                         # own tag: design and actual both switch
                    design = actual = val
                    last_foreign = None
                else:
                    actual = val
                    last_foreign = wi
                s_prev = st
            carry = design
        return overs, unders

    def _is_pure_tag(self, w):
        return (w.radius is None and (w.action or "pass") == "pass" and not w.name and not w.cart
                and bool(w.mode))

    def _harmonize(self, route):
        """Make the physical speed tags work for every leg that drives over them (one tape used in
        both directions): a speed-UP tag that another leg would read where it must be slow is moved
        further along its own leg; if that is not possible it gets the slower mode. Then tags of other
        legs that slow a leg down for nothing get a 'restore' tag behind them where that is safe."""
        r = copy.deepcopy(route)
        wps = r.waypoints
        moved = slowed = added = 0
        for _ in range(40):
            overs, _u = self._conflicts(r)
            if not overs:
                break
            wi, run, want = overs[0]
            w = wps[wi]
            prev_mode = None                            # mode the owner had before this tag
            for j in range(wi - 1, -1, -1):
                if wps[j].mode:
                    prev_mode = wps[j].mode.upper()
                    break
            up = prev_mode is not None and self._v(w.mode) > self._v(prev_mode) + 1e-6
            nxt = wps[(wi + 1) % len(wps)]
            L = math.hypot(nxt.x - w.x, nxt.y - w.y)
            fixed = False
            if up and self._is_pure_tag(w) and L > 0.6:
                # speed up later on its own leg: find the first place where no other leg reads it too early
                ux, uy = (nxt.x - w.x) / L, (nxt.y - w.y) / L
                x0, y0 = w.x, w.y
                n_over = len(overs)
                for d in [0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0]:
                    if d > L - 0.3:
                        break
                    w.x, w.y = round(x0 + ux * d, 3), round(y0 + uy * d, 3)
                    o2, _ = self._conflicts(r)
                    if not any(c[0] == wi for c in o2) and len(o2) < n_over:
                        fixed = True
                        moved += 1
                        break
                if not fixed:
                    w.x, w.y = x0, y0
            if not fixed:
                if self._v(want) < self._v(w.mode) - 1e-6:
                    w.mode = want                       # slower for its owner too (always safe)
                    slowed += 1
                else:
                    break                               # nothing safe left to do
        # restore tags behind foreign slow tags (only where no other leg reads them)
        for _ in range(20):
            _o, unders = self._conflicts(r)
            done = False
            for (run, s0, s1, want, wi) in unders:
                if s1 - s0 < 3.0:
                    continue
                # position a bit after the foreign tag, on this run's reference path
                import numpy as np
                for off in (0.5, 1.0, 1.5, 2.0):
                    s_t = s0 + off
                    if s_t > s1 - 1.0:
                        break
                    k = int(np.searchsorted(run.s, s_t))
                    k = min(max(k, 0), len(run.x) - 1)
                    cand = Waypoint(x=round(float(run.x[k]), 3), y=round(float(run.y[k]), 3), mode=want)
                    # insert after the waypoint the run starts from, keeping order along the run
                    ins = self._insert_index(r.waypoints, run, cand)
                    if ins is None:
                        continue
                    trial = copy.deepcopy(r)
                    trial.waypoints.insert(ins, cand)
                    o2, _ = self._conflicts(trial)
                    if not o2:
                        r = trial
                        added += 1
                        done = True
                        break
                if done:
                    break
            if not done:
                break
        if moved or slowed or added:
            self._shared_log = (f"Shared tape (RFID tags are read in both directions): {moved} speed-up tag(s) moved "
                                f"further along their leg, {slowed} tag(s) set to a slower mode, {added} tag(s) added "
                                f"to speed up again after another leg's slow tag.")
        return r

    def _insert_index(self, wps, run, cand):
        """Index to insert a tag waypoint on the leg of `run` so that it lies between its neighbours."""
        n = len(wps)
        i = run.start_wp
        best = None
        k = i
        for _ in range(n):
            a, b = wps[k], wps[(k + 1) % n]
            vx, vy = b.x - a.x, b.y - a.y
            L2 = vx * vx + vy * vy
            if L2 > 1e-9:
                t = ((cand.x - a.x) * vx + (cand.y - a.y) * vy) / L2
                d = abs((cand.x - a.x) * vy - (cand.y - a.y) * vx) / math.sqrt(L2)
                if 0.02 < t < 0.98 and d < 0.1:
                    best = k + 1
                    break
            if k == run.end_wp:
                break
            k = (k + 1) % n
        return best

    # ------------------------------------------------------------------ verify
    def _nearest_corner(self, pts, x, y, reach=4.0):
        best, bd = None, reach
        for p in pts:
            if not p.corner:
                continue
            d = math.hypot(p.x - x, p.y - y) - p.radius
            if d < bd:
                bd, best = d, p
        return best

    def evaluate(self, route):
        from . import run_simulation
        res, s = run_simulation(self.agv, route, self.fmap)
        lf = s.get("line_follow") or {}
        score = (len(s.get("collisions", [])), int(bool(lf.get("aborted"))) + lf.get("losses", 0),
                 len(s.get("clearance_violations", [])), s["cycle_time_s"])
        return res, s, score

    def _valid(self, s):
        lf = s.get("line_follow") or {}
        rng = self.agv.line_follow.sensor_range_mm
        return (not s.get("collisions") and not lf.get("aborted") and not lf.get("losses")
                and (not lf or lf.get("max_abs_mm", 0) <= self.tape_limit * rng))

    def _score(self, s):
        lf = s.get("line_follow") or {}
        rng = self.agv.line_follow.sensor_range_mm
        over = max(0.0, lf.get("max_abs_mm", 0) - self.tape_limit * rng) if lf else 0.0
        return (len(s.get("collisions", [])), int(bool(lf.get("aborted"))) + lf.get("losses", 0),
                round(over), len(s.get("clearance_violations", [])), s["cycle_time_s"])

    def _describe(self, it, s):
        lf = s.get("line_follow") or {}
        txt = (f"Try {it}: cycle {s['cycle_time_s']:.1f} s, avg {s.get('avg_speed_overall', 0):.3f} m/s, "
               f"collisions {len(s.get('collisions', []))}, margin spots {len(s.get('clearance_violations', []))}")
        if lf:
            txt += f", tape max {lf.get('max_abs_mm', 0):.0f} mm"
            if lf.get("aborted") or lf.get("losses"):
                txt += " - TAPE LOST"
        return txt

    def _hot_corners(self, pts, res, s):
        """Corners near collisions / tape trouble."""
        rng = self.agv.line_follow.sensor_range_mm
        col = set()
        for c in s.get("collisions", []):
            p = self._nearest_corner(pts, c["x"], c["y"])
            if p is not None:
                col.add(id(p))
        tape = []
        lf = s.get("line_follow") or {}
        bad = [(e["x"], e["y"]) for e in lf.get("events", []) if e.get("kind") == "tape lost"]
        if res.lf is not None:
            pv = np.abs(np.nan_to_num(res.lf["pv"]))
            hot = np.nonzero(pv > self.tape_limit * rng)[0]
            bad += [(float(res.x[i]), float(res.y[i])) for i in hot[::10]]
        seen = set()
        self._exit_hot = set()
        n = len(pts)
        for (x, y) in bad:
            # on the straight after a corner (settling while speeding up)?
            for i, p in enumerate(pts):
                if not p.corner:
                    continue
                q = pts[(i + 1) % n]
                L = math.hypot(q.x - p.x, q.y - p.y)
                if L < 1e-6:
                    continue
                ux, uy = (q.x - p.x) / L, (q.y - p.y) / L
                t = (x - p.x) * ux + (y - p.y) * uy
                off = abs(-(x - p.x) * uy + (y - p.y) * ux)
                tan = p.radius * math.tan(abs(getattr(p, "phi", 0.0)) / 2)
                if off < 0.3 and tan + 0.2 < t < min(L - 0.5, tan + self.exit_after + p.exit_extra + 6.0):
                    self._exit_hot.add(id(p))
                    if id(p) not in seen:
                        seen.add(id(p))
                        tape.append(p)
                    break
            else:
                p = self._nearest_corner(pts, x, y, reach=6.0)
                if p is not None and id(p) not in seen:
                    seen.add(id(p))
                    tape.append(p)
        return col, tape

    def run(self):
        pts = self.skeleton_keep() if self.mode == "keep" else self.skeleton_plan()
        loop = self.route.loop
        best = None
        it = 0
        # ---- phase A: make it work (start with generous radii)
        for _ in range(self.max_iter):
            it += 1
            self.fit_radii(pts, loop)
            route = self.build_route(pts)
            res, s, _ = self.evaluate(route)
            self.log(self._describe(it, s))
            sc = self._score(s)
            if best is None or sc < best[2]:
                best = (route, s, sc, res, copy.deepcopy(pts))
            if self._valid(s):
                break
            if self.mode == "plan" and self.replans and best[2] < sc and not getattr(self, "_reverted", False):
                # the new lane is worse than the best one so far: go back to it and fix it with
                # radii / speed instead of planning further away
                self._reverted = True
                self.replans = 99
                pts = copy.deepcopy(best[4])
                route, s, _, res = best[0], best[1], best[2], best[3]
                self.log(f"  that lane is worse - back to the best one (try with {best[2][0]} collision(s)) "
                         f"to fix it with other radii / speeds")
            col, tape = self._hot_corners(pts, res, s)
            changed = False
            if self.mode == "plan" and s.get("collisions"):
                k = self._push_legs(pts, s["collisions"])
                if k:
                    self.log(f"  {len(s['collisions'])} collision(s) -> moved {k} lane section(s) "
                             f"{0.25:.2f} m away from the hit spot")
                    continue
            if self.mode == "plan" and s.get("collisions") and self.replans < 4:
                # keep the planned lane further away from the spots that were hit (trolleys cut
                # inside corners): block a disc around each hit point for the planner and plan again
                grow = self.half_width + self.margin + 0.3
                for c in s["collisions"]:
                    for i, (bx, by, br) in enumerate(self.soft_blocks):
                        if math.hypot(bx - c["x"], by - c["y"]) < 0.6:     # hit again: keep a bit further away
                            self.soft_blocks[i] = (bx, by, min(br + 0.3, grow + 0.6))
                            break
                    else:
                        self.soft_blocks.append((c["x"], c["y"], grow))
                if any(self._nearest_corner(pts, c["x"], c["y"]) is not None for c in s["collisions"]) \
                        and self.turn_extra < 3 * self.turn_extra_step - 1e-6:
                    self.turn_extra += self.turn_extra_step      # corners need more room than straights
                self.replans += 1
                self.log(f"  {len(s['collisions'])} collision(s) -> planning again further away from those spots")
                try:
                    pts = self.skeleton_plan()
                    continue
                except ValueError as ex:
                    self.replans = 99
                    self.log(f"  (no other path: {ex})")
            # collisions away from any corner: the lane itself is too narrow -> wider clearance
            n_col_far = sum(1 for c in s.get("collisions", []) if self._nearest_corner(pts, c["x"], c["y"]) is None)
            for p in pts:
                if id(p) in col:
                    # try other radii for this corner: smaller first, then larger
                    if not hasattr(p, "alts"):
                        base = p.radius
                        cands = [r for r in RADII if r < base - 1e-3 and r >= self.min_radius][:2] + \
                                [r for r in sorted(RADII) if r > base + 1e-3 and r <= max(self.max_radius, 3.0)
                                 and r <= getattr(p, "fit", 99)][:2]
                        p.alts = cands
                    if p.alts:
                        p.want = p.alts.pop(0)
                        p.flag_collide += 1
                        changed = True
                    else:
                        n_col_far += 1
            if self.mode == "plan" and n_col_far and self.extra_clear < 1.2:
                self.extra_clear += 0.3
                self.log(f"  collisions on the lane / with the trolleys -> planning again with "
                         f"{self.half_width + self.margin + self.extra_clear:.2f} m clearance")
                try:
                    pts = self.skeleton_plan()
                    changed = True
                    continue
                except ValueError as ex:
                    self.extra_clear -= 0.3
                    self.log(f"  (no path with more clearance: {ex})")
            for p in tape:
                if id(p) in col:
                    continue
                if id(p) in self._exit_hot and p.exit_extra < 4.0:
                    p.exit_extra += 1.0             # let the steering settle before speeding up
                    changed = True
                    continue
                v_in = self._entry_speed(res, route, p)
                v_curve = self.speeds[p.curve_mode or self.curve_mode][0]
                if v_in is not None and v_in > v_curve + 0.03 and p.slow_extra < 6:
                    p.slow_extra += 1.0
                    changed = True
                elif p.radius < getattr(p, "fit", 0) - 0.05 and p.radius < 3.0 and not p.flag_collide:
                    p.want = min(getattr(p, "fit"), p.radius + 0.5)
                    changed = True
                elif (p.curve_mode or self.curve_mode) != "APPROACH" and "APPROACH" in self.speeds:
                    p.curve_mode = "APPROACH"
                    changed = True
            if not changed:
                break
        # ---- phase B: faster - tighter corners (shorter slow zones) while it stays clean
        if self._valid(best[1]):
            pts = copy.deepcopy(best[4])
            frozen = set()
            for _ in range(4):
                cand = [p for p in pts if p.corner and id(p) not in frozen]
                prev = {id(p): getattr(p, "want", self.max_radius) for p in cand}
                moved = False
                for p in cand:
                    smaller = [r for r in RADII if r < min(p.radius, prev[id(p)]) - 1e-3 and r >= self.min_radius]
                    if smaller:
                        p.want = smaller[0]
                        moved = True
                    else:
                        frozen.add(id(p))
                if not moved:
                    break
                it += 1
                self.fit_radii(pts, loop)
                route = self.build_route(pts)
                res, s, _ = self.evaluate(route)
                self.log(self._describe(it, s) + "  (tighter corners)")
                if self._valid(s):
                    if s["cycle_time_s"] < best[1]["cycle_time_s"] - 0.05:
                        best = (route, s, self._score(s), res, copy.deepcopy(pts))
                    else:
                        break                   # no gain any more
                else:
                    col, tape = self._hot_corners(pts, res, s)
                    hot = col | {id(p) for p in tape}
                    if not hot:
                        break
                    for p in pts:
                        if id(p) in hot:
                            p.want = prev.get(id(p), p.want)
                            frozen.add(id(p))
        # ---- phase C: your target cycle time
        tgt = float(getattr(self.route, "target_cycle_s", 0.0) or 0.0)
        if tgt > 0 and self._valid(best[1]):
            best = self._aim_target(best, tgt, loop, it)
        route, s, sc, res, _ = best
        if self._shared_log:
            self.log(self._shared_log)
        self._push_report()
        if sc[0]:
            spots = ", ".join(f"({c['x']:.1f}, {c['y']:.1f}) {c.get('part', '')}".strip()
                              for c in s.get("collisions", [])[:4])
            self.log(f"Still {sc[0]} collision(s) at {spots}: the floor is too tight there for this vehicle "
                     f"(trolleys cut inside corners) - give that turn more room, use fewer trolleys, "
                     f"or move the stops.")
        if sc[1]:
            self.log("The AGV still loses the tape somewhere - see the TAPE LOST mark.")
        elif sc[2]:
            self.log("The tape error still goes above the target somewhere (see the orange ring).")
        if not sc[0] and not sc[1] and not sc[2]:
            self.log(f"Best: cycle {s['cycle_time_s']:.1f} s, average {s.get('avg_speed_overall', 0):.3f} m/s - "
                     f"no collisions, stays on the tape.")
        return route, s, res

    def _aim_target(self, best, tgt, loop, it):
        """Approach the target cycle time. Too slow even at full speed -> say where the time goes.
        Faster than needed -> drive calmer: the fastest mode is used only on the longest straights
        (then not at all, then the next mode ...) as long as the cycle stays within the target -
        less tape error, trolley push and wear."""
        T = best[1]["cycle_time_s"]
        if T > tgt + 0.05:
            best = self._speed_up(best, tgt, loop)
            T = best[1]["cycle_time_s"]
        s0 = best[1]
        if T > tgt + 0.05:
            self.log(f"Target {tgt:.1f} s: the fastest clean route takes {T:.1f} s - {T - tgt:.1f} s too slow. "
                     f"Time: driving {s0['drive_time_s']:.0f} s, waiting {s0['dwell_time_s']:.0f} s, turning on the spot "
                     f"{s0['pivot_time_s']:.0f} s. Allow faster speed modes, shorten waits / handling times, or raise "
                     f"the mode speeds on the AGV.")
            return best
        if T > tgt - 1.0:
            self.log(f"Target {tgt:.1f} s: reached ({T:.1f} s) - it needs full speed.")
            return best
        chosen = best
        pts0 = best[4]
        L_max = 0.0
        n = len(pts0)
        for i in range(n if loop else n - 1):
            a, b = pts0[i], pts0[(i + 1) % n]
            L_max = max(L_max, math.hypot(b.x - a.x, b.y - a.y))
        evals = [0]

        def trial(fmin):
            nonlocal it
            self.fast_min_len = dict(fmin)
            pts = copy.deepcopy(pts0)
            self.fit_radii(pts, loop)
            route = self.build_route(pts)
            res, s, _ = self.evaluate(route)
            it += 1
            evals[0] += 1
            return (route, s, self._score(s), res, pts)
        fmin = {}
        tiers = []                                    # fast modes grouped by speed (MAX 1.0 = HIGH 0.98 ...)
        for m in sorted(self.fast_modes, key=lambda m: -self._v(m)):
            if tiers and abs(self._v(tiers[-1][0]) - self._v(m)) < 0.05:
                tiers[-1].append(m)
            else:
                tiers.append([m])
        for tier in tiers:                           # fastest first
            for m in tier:
                fmin[m] = L_max + 1.0                 # can this speed go completely?
            c = trial(fmin)
            if self._valid(c[1]) and c[1]["cycle_time_s"] <= tgt + 0.05:
                chosen = c
                continue                              # yes - try without the next one too
            # no: keep it only on straights longer than some length (binary search, slowest that fits)
            lo, hi = 0.0, L_max + 1.0
            for _ in range(6):
                mid = 0.5 * (lo + hi)
                for m in tier:
                    fmin[m] = mid
                c = trial(fmin)
                if self._valid(c[1]) and c[1]["cycle_time_s"] <= tgt + 0.05:
                    chosen, lo = c, mid
                else:
                    hi = mid
            for m in tier:
                fmin[m] = lo
            break
        self.fast_min_len = {}
        Tc = chosen[1]["cycle_time_s"]
        if chosen is best:
            self.log(f"Target {tgt:.1f} s: reached ({T:.1f} s, {tgt - T:.1f} s to spare).")
        else:
            lim = [f"{m} only on straights over {v:.0f} m" if v <= L_max else f"no {m}" for m, v in fmin.items()
                   if v > 0]
            self.log(f"Target {tgt:.1f} s: {Tc:.1f} s ({tgt - Tc:.1f} s to spare) with calmer driving - "
                     + ", ".join(lim) + f" (full speed would be {T:.1f} s; {evals[0]} extra test drives).")
        return chosen

    def _speed_up(self, best, tgt, loop):
        """The route is slower than the target: take the curves faster, one speed mode step at a time
        (corners where the tape error / swept area gets too big step back), then trim the safety
        distances before / after the curves. Every step is checked with the full simulation."""
        T0 = best[1]["cycle_time_s"]
        t_slow = sum(r["time_s"] for r in best[1].get("speed_by_mode", []) if r["mode"] in (self.curve_mode, "APPROACH"))
        self.log(f"Target {tgt:.1f} s: {T0:.1f} s is {T0 - tgt:.1f} s too slow - {t_slow:.0f} s of it is driven in "
                 f"{self.curve_mode}/APPROACH (curves and slow zones). Trying faster curves ...")
        ladder = sorted([m for m in self.speeds if self._v(m) > self._v(self.curve_mode) + 1e-6 and m != "APPROACH"],
                        key=self._v)
        pts = copy.deepcopy(best[4])
        frozen = set()                                  # corners that cannot go faster

        def nxt_mode(p):
            cur = p.curve_mode or self.curve_mode
            nx = [m for m in ladder if self._v(m) > self._v(cur) + 1e-6]
            return cur, (nx[0] if nx else None)

        def check(label):
            self.fit_radii(pts, loop)
            route = self.build_route(pts)
            res, s, _ = self.evaluate(route)
            self.log(self._describe("curves", s) + f"  ({label})")
            return route, res, s

        def arc_len(p):
            return abs(getattr(p, "phi", 0.0)) * max(p.radius, 0.05)
        for _ in range(len(ladder) + 1):
            if best[1]["cycle_time_s"] <= tgt + 0.05:
                break
            up = [(p,) + nxt_mode(p) for p in pts if p.corner and id(p) not in frozen]
            up = [(p, cur, nx) for p, cur, nx in up if nx]
            if not up:
                break
            # 1) all corners one step faster at once
            for p, cur, nx in up:
                p.curve_mode = nx
            route, res, s = check("all curves one mode faster: " + ", ".join(sorted({nx for _, _, nx in up})))
            if self._valid(s):
                best = (route, s, self._score(s), res, copy.deepcopy(pts))
                continue
            for p, cur, nx in up:
                p.curve_mode = cur
            # 2) one corner at a time, the longest curves first (most time to win)
            for p, cur, nx in sorted(up, key=lambda u: -arc_len(u[0])):
                if best[1]["cycle_time_s"] <= tgt + 0.05:
                    break
                p.curve_mode = nx
                route, res, s = check(f"curve at ({p.x:.1f}, {p.y:.1f}) in {nx}")
                if self._valid(s) and s["cycle_time_s"] < best[1]["cycle_time_s"] - 0.05:
                    best = (route, s, self._score(s), res, copy.deepcopy(pts))
                else:
                    p.curve_mode = cur
                    frozen.add(id(p))
        if best[1]["cycle_time_s"] > tgt + 0.05:
            # trim the safety distances around the curves
            ea, bm = self.exit_after, self.brake_margin
            pts = copy.deepcopy(best[4])
            self.exit_after, self.brake_margin = 0.1, 0.1
            self.fit_radii(pts, loop)
            route = self.build_route(pts)
            res, s, _ = self.evaluate(route)
            self.log(self._describe("margins", s) + "  (speed changes 0.1 m before / after the curves)")
            if self._valid(s) and s["cycle_time_s"] < best[1]["cycle_time_s"]:
                best = (route, s, self._score(s), res, copy.deepcopy(pts))
            else:
                self.exit_after, self.brake_margin = ea, bm
        T1 = best[1]["cycle_time_s"]
        if T1 < T0 - 0.05:
            modes = sorted({p.curve_mode or self.curve_mode for p in best[4] if p.corner}, key=self._v)
            self.log(f"Faster curves: {T0:.1f} s -> {T1:.1f} s (curves now in {', '.join(modes)}).")
        if T1 > tgt + 0.05:
            # what else would it take? (information only - these are your decisions)
            tips = []
            cm_now = sorted({p.curve_mode or self.curve_mode for p in best[4] if p.corner}, key=self._v)
            top = cm_now[-1] if cm_now else self.curve_mode
            faster = [m for m in ladder if self._v(m) > self._v(top) + 1e-6]
            if faster and self._v(faster[0]) > 1.4 * self._v(top):
                v_mid = 0.5 * (self._v(top) + self._v(faster[0]))
                tips.append(f"the curves hold the tape at {top} ({self._v(top):.2f} m/s) but not at {faster[0]} "
                            f"({self._v(faster[0]):.2f} m/s) - add a speed mode in between (e.g. CURVE {v_mid:.2f} m/s, "
                            f"Vehicle → Speed modes, and in modes.py on the AGV) and run auto route again")
            if self.trolley_safe and self.hook_type_tow:
                ts = self.trolley_safe
                self.trolley_safe = False
                pts = copy.deepcopy(best[4])
                self.fit_radii(pts, loop)
                res, s, _ = self.evaluate(self.build_route(pts))
                self.trolley_safe = ts
                if self._valid(s) and s["cycle_time_s"] < T1 - 0.5:
                    tips.append(f"without 'Trolley-safe slowing down' (no slow approach into stops, direct ramps) it "
                                f"would be {s['cycle_time_s']:.1f} s - your call, the trolleys then push harder")
            tips.append(f"a higher 'Allowed tape error' (now {self.tape_limit * 100:.0f} % of the sensor range) lets "
                        f"curves go faster; shorter waits save time 1:1")
            self.log("To reach the target: " + "; ".join(tips) + ".")
        return best

    def _push_report(self):
        if not self.hook_type_tow:
            return
        if not self.trolley_safe:
            self.log("Trolley-safe slowing down is off.")
            return
        if not self.push_notes:
            return
        g = 9.81
        m_max = max(v["m"] for v in self.push_notes.values())
        self.log(f"Trolley weight: up to {m_max:.0f} kg. They have no brakes, so slowing down faster than their "
                 f"own rolling friction ({self.crr_c * g:.2f} m/s2) makes them push the AGV. Allowed push "
                 f"{self.push_allow:.0f} N -> gentle enough: {self._a_safe(m_max):.2f} m/s2 with {m_max:.0f} kg.")
        names = {"SLOW": "DECEL_RATE", "HIGH": "DECEL_RATE", "APPROACH": "APPROACH_DECEL_RATE"}
        for k, v in sorted(self.push_notes.items(), key=lambda kv: -kv[1]["push"]):
            if v["to"] == "STOP":
                self.log(f"  Stops: the stop brake ({v['ramp']:.1f} m/s2) makes {v['m']:.0f} kg push ~{v['push']:.0f} N for a "
                         f"moment - the route now comes into every stop at APPROACH ({v['v']:.2f} m/s), so it lasts only "
                         f"{v['v'] / max(v['ramp'], 1e-3):.2f} s.")
                continue
            par = names.get(v["to"], f"the Brake of speed mode {v['to']}")
            a_ok = self._a_safe(v["m"])
            if v["push"] <= self.push_allow + 1e-6:
                self.log(f"  Slowing down into {v['to']} at {v['ramp']:.2f} m/s2 ({par}): push {v['push']:.0f} N - OK.")
            else:
                self.log(f"  Slowing down into {v['to']} at {v['ramp']:.2f} m/s2 ({par}): the trolleys push {v['push']:.0f} N "
                         f"with {v['m']:.0f} kg - set {par} <= {a_ok:.2f} m/s2 on the AGV for a gentle slow-down.")

    def _entry_speed(self, res, route, p):
        """Speed of the AGV when the tape curve at corner p starts."""
        for run in res.plan.runs:
            for arc in run.arcs:
                w = route.waypoints[arc.waypoint]
                if abs(w.x - p.x) < 1e-3 and abs(w.y - p.y) < 1e-3:
                    m = (res.run_id == run.index) & (res.ref_s >= arc.s_start)
                    idx = np.nonzero(m)[0]
                    if len(idx):
                        return float(abs(res.v[idx[0]]))
        return None


def _keep_fork_tags(old, new):
    """LEFT / RIGHT track-selection tags of the input route survive the auto route (put back on the
    new route's tape where they were)."""
    from .combined import insert_point
    for w in old.waypoints:
        f = (getattr(w, "fork", "") or "").strip()
        if not f:
            continue
        if any((getattr(v, "fork", "") or "") == f and math.hypot(v.x - w.x, v.y - w.y) < 0.05 for v in new.waypoints):
            continue
        try:
            insert_point(new, w.x, w.y, snap=0.3, fork=f)
        except ValueError:
            pass
    return new


def auto_route(agv, route, fmap, **kw):
    ar = AutoRoute(agv, route, fmap, **kw)
    new_route, summary, res = ar.run()
    if new_route is not None:
        new_route = _keep_fork_tags(route, new_route)
    return {"route": new_route, "summary": summary, "result": res, "log": ar.logs}
