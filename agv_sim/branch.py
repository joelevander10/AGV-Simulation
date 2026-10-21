"""Reverse branch (spur): the AGV drives past a junction, stops at a REVERSE RFID tag, backs into a
branch (rear sensor), stops at the end of the branch, then drives forward out of it and carries on.

    add_reverse_branch(route, agv, junction=(x, y), branch=[(x, y), ..., end], **options)

Route points that are added (after the leg the junction is on, in the order they are driven):
    R      REVERSE tag  `past_m` beyond the junction - stop, then reverse   (direction: reverse)
    J      junction, reverse, into the branch (curve radius `radius`)
    ...    branch bend points (reverse)
    E      end of the branch: stop (wait, name, attach / detach), then forward
    ...    bend points again (forward)
    J      junction, forward, back onto the route (speed mode as before the branch)
The leg the junction is on must point AWAY from the branch: backing up has to bend into the branch
by less than about 90 degrees (a branch that points back the way the AGV came cannot be reversed into).
"""
from __future__ import annotations

import copy
import math
from typing import List, Sequence, Tuple

from .config import Waypoint

MAX_ENTRY_DEG = 80.0          # largest bend from the backing-up direction into the branch


def _legs(route):
    n = len(route.waypoints)
    return list(range(n - 1)) + ([n - 1] if route.loop and n > 2 else [])


def _mode_in_effect(route, k):
    """Speed mode of the leg leaving point k (the last tag at or before k, else the route default)."""
    wps, n = route.waypoints, len(route.waypoints)
    for j in range(n):
        i = (k - j) % n
        if not route.loop and i > k:
            break
        m = (wps[i].mode or "").strip()
        if m:
            return m
    return route.default_mode or ""


def _angle(a, b):
    return abs((b - a + math.pi) % (2 * math.pi) - math.pi)


def find_leg(route, x, y, first_dir: Tuple[float, float], tol=0.6):
    """The forward leg through (x, y) from which backing up can enter a branch going `first_dir`.
    Returns (leg index k, t along the leg (m), leg length, unit vector) or raises ValueError."""
    wps = route.waypoints
    n = len(wps)
    cands = []
    bdir = math.atan2(first_dir[1], first_dir[0])
    for k in _legs(route):
        a, b = wps[k], wps[(k + 1) % n]
        dx, dy = b.x - a.x, b.y - a.y
        L = math.hypot(dx, dy)
        if L < 1e-6:
            continue
        t = ((x - a.x) * dx + (y - a.y) * dy) / L
        if t < -1e-6 or t > L + 1e-6:
            continue
        d = abs((x - a.x) * dy - (y - a.y) * dx) / L
        if d > tol:
            continue
        ux, uy = dx / L, dy / L
        back = math.atan2(-uy, -ux)                     # direction of travel when backing up
        fwd = str(a.direction or "forward")[:1].lower() != "r"
        cands.append((d, k, t, L, (ux, uy), math.degrees(_angle(back, bdir)), fwd))
    if not cands:
        raise ValueError("The junction must be on the route (click on the route line).")
    ok = [c for c in cands if c[6] and c[5] <= MAX_ENTRY_DEG]
    if not ok:
        worst = min(cands, key=lambda c: c[5])
        raise ValueError(
            f"The AGV cannot back into this branch from here: backing up it would have to turn "
            f"{worst[5]:.0f} deg into the branch (max {MAX_ENTRY_DEG:.0f}). The route must pass the junction "
            f"driving AWAY from the direction the branch points to, so that it can stop past the junction "
            f"and back straight into it.")
    d, k, t, L, u, ang, _ = min(ok, key=lambda c: (c[0], c[5]))
    return k, max(0.0, min(L, t)), L, u, ang


def add_reverse_branch(route, agv, junction: Sequence[float], branch: List[Sequence[float]], past_m=None,
                       wait_rev=1.0, wait_end=5.0, end_name="", end_cart="", mode_in="SLOW", radius=1.0,
                       end_mode=""):
    """New route with the reverse branch added (the input route is not changed). Returns (route, info)."""
    if len(branch) < 1:
        raise ValueError("Click at least the end of the branch.")
    r = copy.deepcopy(route)
    jx, jy = float(junction[0]), float(junction[1])
    pts = [(float(p[0]), float(p[1])) for p in branch]
    if math.hypot(pts[0][0] - jx, pts[0][1] - jy) < 0.3:
        raise ValueError("The first branch point is on the junction - click further into the branch.")
    first = (pts[0][0] - jx, pts[0][1] - jy)
    k, t, L, (ux, uy), entry = find_leg(r, jx, jy, first)
    wps, n = r.waypoints, len(r.waypoints)
    a = wps[k]
    jx, jy = a.x + ux * t, a.y + uy * t                       # exactly on the leg
    if past_m is None:
        try:
            past_m = agv.tracking_offset(1) + agv.tracking_offset(-1) + 0.5
        except Exception:  # noqa: BLE001
            past_m = 1.0
    past_m = max(0.1, float(past_m))
    if t + past_m > L - 0.05:
        raise ValueError(f"Not enough straight route after the junction for the REVERSE tag: it needs "
                         f"{past_m:.2f} m, there are {max(0.0, L - t):.2f} m to the next route point. Put the "
                         f"junction further back, or make the distance smaller.")
    if t < 0.05:
        raise ValueError("The junction is on a route point - click a little after it, on the straight line.")
    mode_back = _mode_in_effect(r, k)
    rx, ry = jx + ux * past_m, jy + uy * past_m
    m_in = (mode_in or "").strip().upper()
    m_out = (end_mode or "").strip().upper() or m_in
    new = [Waypoint(x=round(rx, 3), y=round(ry, 3), name="REVERSE", action="stop", dwell=float(wait_rev),
                    direction="reverse", mode=m_in, radius=0.0),
           Waypoint(x=round(jx, 3), y=round(jy, 3), direction="reverse", mode=m_in, radius=float(radius))]
    for (x, y) in pts[:-1]:
        new.append(Waypoint(x=round(x, 3), y=round(y, 3), direction="reverse", radius=float(radius)))
    ex, ey = pts[-1]
    new.append(Waypoint(x=round(ex, 3), y=round(ey, 3), name=end_name or "", action="stop",
                        dwell=float(wait_end), direction="forward", mode=m_out, radius=0.0,
                        cart=(end_cart or "").strip().lower()))
    for (x, y) in reversed(pts[:-1]):
        new.append(Waypoint(x=round(x, 3), y=round(y, 3), direction="forward", radius=float(radius)))
    new.append(Waypoint(x=round(jx, 3), y=round(jy, 3), direction="forward", radius=float(radius),
                        mode=mode_back or ""))
    r.waypoints = wps[:k + 1] + new + wps[k + 1:]
    blen = math.hypot(pts[0][0] - jx, pts[0][1] - jy) + sum(
        math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1))
    return r, {"leg": k + 1, "inserted_at": k + 1, "points": len(new), "past_m": round(past_m, 2),
               "entry_deg": round(entry, 1), "branch_m": round(blen, 2), "mode_back": mode_back,
               "reverse_tag": [round(rx, 3), round(ry, 3)], "junction": [round(jx, 3), round(jy, 3)]}
