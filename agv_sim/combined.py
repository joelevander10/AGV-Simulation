"""Several routes on one map: the tape of all of them and the stop areas they use.

Used by the web app (tab "All routes") and by tools/combined_map.py.

combine(route_files, routes_dir, configs_dir) ->
    {"map": {"image", "meters_per_pixel"}, "routes": [...], "areas": [...], "warnings": [...]}

Every route is smoothed with its own vehicle (configs/<name>.yaml, found from the route file name,
e.g. 'AGV_Kayaba (ITG _ T6)_auto.yaml' -> 'AGV_Kayaba (ITG).yaml'; otherwise the first vehicle file),
so the drawn tape is the path the AGV really follows, curves included.
Stops of different routes closer than `merge` metres (to any stop already in the group) are one
stop area; each area lists which route stops there and what it does (start, detach, attach ...).
"""
from __future__ import annotations

import glob
import math
import os
import re
from typing import List, Optional

import numpy as np

from .config import load_agv, load_route
from .path import build_plan


def vehicle_for(route_file: str, configs_dir: str) -> Optional[str]:
    base = os.path.basename(route_file)
    cands = []
    m = re.search(r"\(([^)]*)\)", base)
    if m:
        for part in re.split(r"[_/,&+ ]+", m.group(1)):
            if part:
                cands.append(os.path.join(configs_dir, base[:m.start()] + f"({part}).yaml"))
    stem = re.sub(r"_auto$", "", os.path.splitext(base)[0])
    cands.append(os.path.join(configs_dir, stem + ".yaml"))
    for c in cands:
        if os.path.exists(c):
            return c
    allc = sorted(glob.glob(os.path.join(configs_dir, "*.yaml")))
    return allc[0] if allc else None


def label_for(route_file: str) -> str:
    base = os.path.splitext(os.path.basename(route_file))[0]
    m = re.search(r"\(([^)]*)\)", base)
    return m.group(1).replace(" _ ", " / ") if m else re.sub(r"_auto$", "", base)


def _decimate(x, y, step=0.1):
    """Keep a point every ~step m (and every corner) - enough for drawing."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return x, y
    d = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    keep = np.unique(np.r_[0, np.searchsorted(d, np.arange(0, d[-1], step)), len(x) - 1])
    return x[keep], y[keep]


def _stop_label(i, wp):
    cart = (getattr(wp, "cart", "") or "").lower()
    if cart:
        return cart                         # attach / detach / swap ...
    if wp.name:
        return wp.name
    return "start" if i == 0 else "stop"


def combine(route_files: List[str], routes_dir: str, configs_dir: str, merge: float = 2.5,
            agv_override=None) -> dict:
    routes, warnings, stops = [], [], []
    map0 = None
    for k, rf in enumerate(route_files):
        path = os.path.join(routes_dir, os.path.basename(rf))
        lab = label_for(rf)
        try:
            route = load_route(path)
            vf = vehicle_for(rf, configs_dir)
            agv = agv_override if (agv_override is not None and vf is None) else (load_agv(vf) if vf else agv_override)
            if agv is None:
                raise ValueError("no vehicle file in configs/")
            plan = build_plan(route, agv)
        except Exception as ex:  # noqa: BLE001 - one broken route should not hide the others
            warnings.append(f"{os.path.basename(rf)}: {ex}")
            continue
        mp = {"image": os.path.basename(route.map.image or ""), "meters_per_pixel": float(route.map.meters_per_pixel)}
        if map0 is None:
            map0 = mp
        elif mp["image"] != map0["image"] or abs(mp["meters_per_pixel"] - map0["meters_per_pixel"]) > 1e-9:
            warnings.append(f"{lab} uses another floor plan or scale ({mp['image']}, {mp['meters_per_pixel']} m/px) "
                            f"- drawn on {map0['image']} anyway, positions may not match.")
        runs = []
        for run in plan.runs:
            x, y = _decimate(run.x, run.y)
            runs.append({"x": [round(float(v), 3) for v in x], "y": [round(float(v), 3) for v in y],
                         "dir": int(run.direction)})
        wps = route.waypoints
        n = len(wps)
        rstops = []
        for i, wp in enumerate(wps):
            if wp.action != "stop":
                continue
            j = i + 1 if i + 1 < n else (0 if route.loop else i - 1)
            hx, hy = wps[j].x - wp.x, wps[j].y - wp.y
            if j < i and not (route.loop and j == 0):
                hx, hy = -hx, -hy
            st = {"route": lab, "ri": len(routes), "wp": i + 1, "x": float(wp.x), "y": float(wp.y),
                  "heading": math.atan2(hy, hx), "what": _stop_label(i, wp), "dwell": float(wp.dwell or 0.0)}
            rstops.append(st)
            stops.append(st)
        routes.append({"file": os.path.basename(rf), "label": lab, "vehicle": os.path.basename(vf) if vf else "",
                       "length_m": round(float(plan.length), 1), "loop": bool(route.loop), "runs": runs,
                       "stops": len(rstops), "target_cycle_s": route.target_cycle_s or None,
                       "tags": route_tags(route), "sig": file_sig(path),
                       "default_mode": (route.default_mode or "").upper()})

    # group the stops: single linkage (a stop joins a group when it is near ANY stop of it)
    par = list(range(len(stops)))

    def root(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i
    for i in range(len(stops)):
        for j in range(i):
            if math.hypot(stops[i]["x"] - stops[j]["x"], stops[i]["y"] - stops[j]["y"]) < merge:
                ri, rj = root(i), root(j)
                par[max(ri, rj)] = min(ri, rj)
    groups = {}
    for i, st in enumerate(stops):
        groups.setdefault(root(i), []).append(st)
    areas = []
    L, W = 2.0, 1.4                                  # drawn size of a stop area (m)
    for n_ar, (_, g) in enumerate(sorted(groups.items()), 1):
        cx = float(np.mean([s["x"] for s in g]))
        cy = float(np.mean([s["y"] for s in g]))
        th = g[0]["heading"]
        ext = max([0.0] + [math.hypot(s["x"] - cx, s["y"] - cy) for s in g])
        c, s_ = math.cos(th), math.sin(th)
        hl, hw = L / 2 + ext, W / 2
        poly = [[round(cx + c * u - s_ * v, 3), round(cy + s_ * u + c * v, 3)]
                for u, v in ((-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw))]
        by = {}
        for s in g:
            by.setdefault(s["what"], [])
            if s["route"] not in by[s["what"]]:
                by[s["what"]].append(s["route"])
        areas.append({"n": n_ar, "name": f"S{n_ar}", "x": round(cx, 3), "y": round(cy, 3), "heading": th,
                      "poly": poly, "shared": len({s["route"] for s in g}) > 1,
                      "uses": [{"what": w, "routes": r} for w, r in by.items()],
                      "stops": [{k: s[k] for k in ("route", "wp", "what", "dwell", "x", "y")} for s in g]})
    return {"map": map0 or {"image": "", "meters_per_pixel": 0.02}, "routes": routes, "areas": areas,
            "merge": merge, "warnings": warnings}


def area_text(a: dict) -> str:
    return f"{a['name']}: " + "; ".join(f"{u['what']} ({', '.join(u['routes'])})" for u in a["uses"])


# ----------------------------------------------------------------------------- follow a route in an area
def _inside(w, area):
    x0, y0, x1, y1 = area
    return min(x0, x1) <= w.x <= max(x0, x1) and min(y0, y1) <= w.y <= max(y0, y1)


def _block(wps, area, loop, what):
    """The one run of consecutive waypoints inside the area -> (first index, count).
    Cyclic for a loop. Raises ValueError when the route is not in the area, or goes in more than once."""
    n = len(wps)
    ins = [_inside(w, area) for w in wps]
    if not any(ins):
        raise ValueError(f"{what} has no route point inside the area.")
    if all(ins):
        raise ValueError(f"{what} lies completely inside the area - make the area smaller.")
    starts = [i for i in range(n) if ins[i] and not ins[i - 1]] if loop else \
             [i for i in range(n) if ins[i] and (i == 0 or not ins[i - 1])]
    if len(starts) != 1:
        raise ValueError(f"{what} goes into the area {len(starts)} times - draw the area so the route goes in "
                         f"and out only once (around the part to change).")
    s = starts[0]
    k = 0
    while k < n and ins[(s + k) % n] and (loop or s + k < n):
        k += 1
    return s, k


def _is_stop(w):
    return w.action == "stop" or bool(w.cart)


def _stop_name(w):
    return (w.name or (w.cart or "stop")) + (f" {w.dwell:g} s" if w.dwell else "")


def follow_in_area(ref, tgt, area, keep_own_waits=True):
    """Make route `tgt` drive like route `ref` inside the rectangle `area` = [x0, y0, x1, y1] (m).

    The target's points inside the area are replaced by the reference's points inside the area
    (path, corners, speed modes, stops and trolley actions - like the reference). Everything outside
    the area stays as it was. Where the target started inside the area, it now starts where the
    reference starts. keep_own_waits: the start stop keeps the target's wait time and name, and a
    trolley stop keeps the wait of the target's stop with the same action (attach / detach)."""
    import copy
    rw, tw = ref.waypoints, tgt.waypoints
    rs, rk = _block(rw, area, ref.loop, "The reference route")
    ts, tk = _block(tw, area, tgt.loop, "This route")
    rb = [copy.deepcopy(rw[(rs + k) % len(rw)]) for k in range(rk)]
    tb = [tw[(ts + k) % len(tw)] for k in range(tk)]
    m = len(tw)
    start_in = _inside(tw[0], area)
    if tgt.loop:
        out = [copy.deepcopy(tw[(ts + tk + k) % m]) for k in range(m - tk)]
        new = out + rb
        if start_in:   # start where the reference starts (or at its first stop in the area)
            i0 = next((i for i, w in enumerate(rb) if rw[0].x == w.x and rw[0].y == w.y and _inside(rw[0], area)), None)
            if i0 is None:
                i0 = next((i for i, w in enumerate(rb) if _is_stop(w)), 0)
            k0 = len(out) + i0
        else:
            k0 = next(i for i, w in enumerate(new) if w.x == tw[0].x and w.y == tw[0].y)
        new = new[k0:] + new[:k0]
    else:
        new = [copy.deepcopy(w) for w in tw[:ts]] + rb + [copy.deepcopy(w) for w in tw[ts + tk:]]
    old_stops = [w for w in tb if _is_stop(w)]
    if keep_own_waits:
        used = set()
        if start_in and _is_stop(tw[0]) and _is_stop(new[0]):
            new[0].dwell = tw[0].dwell
            if tw[0].name:
                new[0].name = tw[0].name
            used.add(id(tw[0]))
        for w in rb:
            if not w.cart or (start_in and w is new[0]):
                continue
            mt = next((o for o in old_stops if o.cart == w.cart and id(o) not in used), None)
            if mt is not None:
                w.dwell = mt.dwell
                used.add(id(mt))
    r2 = copy.deepcopy(tgt)
    r2.waypoints = new
    return r2, {"removed_stops": [_stop_name(w) for w in old_stops],
                "added_stops": [_stop_name(w) for w in rb if _is_stop(w)],
                "replaced_points": tk, "copied_points": rk, "points": [len(tw), len(new)]}


# ----------------------------------------------------------------------------- RFID tags
# Kinds of tags on a route (all stored in the route's waypoints):
#   speed  APPROACH / SLOW / MED / HIGH / MAX ...  -> waypoint.mode   (read by the closed-loop simulation)
#   fork   LEFT / RIGHT (track selection)          -> waypoint.fork   (layout only, the path is drawn)
#   stop   STOP (a stop point)                     -> waypoint.action = 'stop' (+ dwell); attach / detach too
#   home   HOME (homepost = where the cycle starts) -> the first waypoint
#   reverse REVERSE (stop, then back up)          -> a point where the route switches to reverse (shown only;
#          made with the Route tab's "Reverse branch" tool)
TAG_TOL = 0.35          # m: a tag this close to a route's tape is on that route
SPEED, FORK, STOP, HOME, REVERSE = "speed", "fork", "stop", "home", "reverse"


def file_sig(path: str) -> str:
    import hashlib
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:12]


def _turn(wps, i, loop):
    """Turn angle (rad) at waypoint i, 0 at the open ends."""
    n = len(wps)
    if not loop and (i == 0 or i == n - 1):
        return 0.0
    a, b, c = wps[(i - 1) % n], wps[i], wps[(i + 1) % n]
    h1 = math.atan2(b.y - a.y, b.x - a.x)
    h2 = math.atan2(c.y - b.y, c.x - b.x)
    return abs((h2 - h1 + math.pi) % (2 * math.pi) - math.pi)


def _plain(w):
    """A point that carries nothing (no stop, name, speed, tag ...)."""
    m = (w.mode or "").strip()
    return (w.action == "pass" and not w.cart and not w.name and not w.dwell and not w.speed and not w.grade
            and (not m or m.lower() == "default") and not (w.fork or ""))


def _removable(wps, i, loop):
    """Removing point i does not change the path (a straight, plain point)."""
    if i == 0 or (not loop and i == len(wps) - 1) or not _plain(wps[i]):
        return False
    if str(wps[i - 1].direction or "forward")[:1].lower() != str(wps[i].direction or "forward")[:1].lower():
        return False
    return _turn(wps, i, loop) < math.radians(1.0)


def _tag_removable(wps, i, loop):           # (kept for older callers) a point that only carries its speed tag
    import copy
    w2 = copy.copy(wps[i])
    w2.mode = ""
    tmp = list(wps)
    tmp[i] = w2
    return _removable(tmp, i, loop)


def route_tags(route) -> list:
    """All RFID tags of a route: [{i, x, y, kind, value, info, at, fixed}]. at = stop | corner | pass;
    fixed = cannot be removed / moved on the map (start point, trolley stops)."""
    wps = route.waypoints
    out = []
    for i, w in enumerate(wps):
        at = "stop" if (w.action == "stop" or w.cart) else ("corner" if _turn(wps, i, route.loop) > math.radians(1) else "pass")
        base = {"i": i, "x": round(float(w.x), 3), "y": round(float(w.y), 3), "at": at}
        rev_in = str(w.direction or "forward")[:1].lower() == "r" and \
            (i > 0 or route.loop) and str(wps[i - 1].direction or "forward")[:1].lower() != "r"
        if i == 0:
            out.append({**base, "kind": HOME, "value": "HOME", "fixed": True,
                        "info": " ".join(x for x in [w.name, w.cart, f"{w.dwell:g} s" if w.dwell else ""] if x)})
        elif rev_in:
            out.append({**base, "kind": REVERSE, "value": "REVERSE", "fixed": True,
                        "info": f"{w.dwell:g} s" if w.dwell else ""})
        elif w.action == "stop" or w.cart:
            out.append({**base, "kind": STOP, "value": "STOP", "fixed": bool(w.cart),
                        "info": " ".join(x for x in [w.name, w.cart, f"{w.dwell:g} s" if w.dwell else ""] if x)})
        m = (w.mode or "").strip()
        if m and m.lower() != "default":
            out.append({**base, "kind": SPEED, "value": m.upper(), "fixed": False, "info": ""})
        if (w.fork or "").strip():
            out.append({**base, "kind": FORK, "value": w.fork.strip().upper(), "fixed": False, "info": ""})
    return out


def _corner_cut(wps, i, loop, default_radius):
    """Length of straight tape the curve at waypoint i uses up on each side (tangent length)."""
    th = _turn(wps, i, loop)
    if th < math.radians(1):
        return 0.0
    r = wps[i].radius if wps[i].radius is not None else default_radius
    return float(r) * math.tan(th / 2) if r and r > 0 else 0.0


def insert_point(route, x, y, margin=0.05, snap=0.05, **attrs):
    """Put a point with `attrs` on the leg nearest to (x, y), on the straight part of the leg (not
    inside a curve). When a route point is already there (within `snap` m), that point gets the
    attributes instead. Returns (index, (x, y), moved_m)."""
    from .config import Waypoint
    wps = route.waypoints
    n = len(wps)
    for k, w in enumerate(wps):
        if math.hypot(w.x - x, w.y - y) <= snap:
            for a, v in attrs.items():
                setattr(w, a, v)
            return k, (w.x, w.y), math.hypot(w.x - x, w.y - y)
    legs = list(range(n - 1)) + ([n - 1] if route.loop and n > 2 else [])
    best = None
    for i in legs:
        a, b = wps[i], wps[(i + 1) % n]
        dx, dy = b.x - a.x, b.y - a.y
        L = math.hypot(dx, dy)
        if L < 1e-6:
            continue
        t = max(0.0, min(L, ((x - a.x) * dx + (y - a.y) * dy) / L))
        d = math.hypot(a.x + dx * t / L - x, a.y + dy * t / L - y)
        if best is None or d < best[0] - 1e-9:
            best = (d, i, t, L)
    if best is None:
        raise ValueError("The route has no leg to put the tag on.")
    d, i, t, L = best
    a, b = wps[i], wps[(i + 1) % n]
    c0 = _corner_cut(wps, i, route.loop, route.default_radius) + margin
    c1 = _corner_cut(wps, (i + 1) % n, route.loop, route.default_radius) + margin
    if c0 + c1 > L:
        raise ValueError(f"No straight tape there (between point {i + 1} and {(i + 1) % n + 1} it is all curve) - "
                         f"put the tag before or after the curve.")
    t2 = max(c0, min(L - c1, t))
    nx, ny = a.x + (b.x - a.x) * t2 / L, a.y + (b.y - a.y) * t2 / L
    w = Waypoint(x=round(nx, 3), y=round(ny, 3), direction=a.direction)
    for k, v in attrs.items():
        setattr(w, k, v)
    route.waypoints.insert(i + 1, w)
    return i + 1, (nx, ny), abs(t2 - t)


def insert_tag(route, x, y, mode, margin=0.05):          # speed tag (kept for older callers)
    return insert_point(route, x, y, margin=margin, snap=-1, mode=mode.upper())


def _attrs_for(kind, value, dwell=None):
    v = str(value or "").strip()
    if kind == SPEED:
        return {"mode": v.upper()}
    if kind == FORK:
        if v.lower() not in ("left", "right"):
            raise ValueError(f"Track selection must be LEFT or RIGHT, not '{v}'.")
        return {"fork": v.lower()}
    if kind == STOP:
        return {"action": "stop", "dwell": max(0.0, float(dwell if dwell is not None else 3.0))}
    raise ValueError(f"Unknown tag kind '{kind}'.")


def _clear(w, kind):
    if kind == SPEED:
        w.mode = ""
    elif kind == FORK:
        w.fork = ""
    elif kind == STOP:
        if w.cart:
            raise ValueError("this stop attaches / detaches trolleys - change it in the Route tab")
        w.action, w.dwell = "pass", 0.0


def apply_tag_changes(route, dels=(), sets=(), adds=(), home=None):
    """Edit the RFID tags of a route.
      dels: [{i, kind}]           remove a tag (indices = the route as it is now). A point that then
                                  carries nothing (and is straight) is removed; otherwise only the tag.
      sets: [{i, kind, value}]    change a tag (speed mode, LEFT/RIGHT, stop wait = value in s)
      adds: [{x, y, kind, value, dwell}]  new tag on the route's tape (moved out of a curve if needed)
      home: {x, y}                move the homepost: the cycle starts there (loop routes). The old start
                                  stays a stop only if it attaches / detaches trolleys.
    Old format (dels = [i], sets = {i: mode}, adds = [{x, y, mode}]) = speed tags."""
    import copy
    if isinstance(sets, dict):
        sets = [{"i": int(k), "kind": SPEED, "value": v} for k, v in sets.items()]
    dels = [d if isinstance(d, dict) else {"i": int(d), "kind": SPEED} for d in dels]
    adds = [a if "kind" in a else {**a, "kind": SPEED, "value": a.get("mode")} for a in adds]
    wps = route.waypoints
    n = len(wps)
    for d in list(dels) + list(sets):
        if not 0 <= int(d["i"]) < n:
            raise ValueError(f"Tag point {int(d['i']) + 1} does not exist (the route file changed?) - reload and try again.")
    notes = []
    touched = set()
    for st in sets:
        i, kind = int(st["i"]), st["kind"]
        if kind == STOP or kind == HOME:
            wps[i].dwell = max(0.0, float(st["value"]))
        else:
            for k, v in _attrs_for(kind, st["value"]).items():
                setattr(wps[i], k, v)
    for d in dels:
        i, kind = int(d["i"]), d["kind"]
        if kind == HOME:
            raise ValueError("the homepost (start) cannot be removed - move it instead")
        try:
            _clear(wps[i], kind)
        except ValueError as ex:
            raise ValueError(f"point {i + 1}: {ex}")
        touched.add(i)
    gone = {i for i in touched if _removable(wps, i, route.loop)}
    route.waypoints = [w for k, w in enumerate(wps) if k not in gone]
    for a in adds:
        attrs = _attrs_for(a["kind"], a.get("value"), a.get("dwell"))
        _, (nx, ny), moved = insert_point(route, float(a["x"]), float(a["y"]), **attrs)
        if moved > 0.2:
            notes.append(f"{str(a.get('value') or a['kind']).upper()} tag moved {moved:.1f} m to ({nx:.1f}, {ny:.1f}) - out of the curve")
    if home:
        if not route.loop:
            raise ValueError("the homepost can only be moved on a closed-loop route (the start of an open route is its first point)")
        old = route.waypoints[0]
        dwell, name = old.dwell, old.name
        k, (nx, ny), moved = insert_point(route, float(home["x"]), float(home["y"]), snap=0.3)
        if k != 0:
            w = route.waypoints[k]
            w.action = "stop"
            w.dwell = w.dwell or dwell
            w.name = w.name or name
            if not old.cart:                  # the old start was only the homepost
                old.action, old.dwell = "pass", 0.0
                if old.name == name:
                    old.name = ""
            route.waypoints = route.waypoints[k:] + route.waypoints[:k]
            if moved > 0.2:
                notes.append(f"HOME moved {moved:.1f} m to ({nx:.1f}, {ny:.1f}) - out of the curve")
    return route, {"removed": len(dels), "changed": len(sets), "added": len(adds), "home": bool(home), "notes": notes}
