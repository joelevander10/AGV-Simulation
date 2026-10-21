"""Post-processing: cycle time breakdown, turning radii, swept path, collisions, drivetrain checks."""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw

from .config import AGVConfig
from .factory_map import FactoryMap
from .simulate import DRIVE, SimResult

_trapz = getattr(np, "trapezoid", None) or np.trapz


# ----------------------------------------------------------------------------- geometry
def body_rect(agv: AGVConfig):
    w = agv.chassis.width / 2
    return (agv.x_rear, agv.x_front, -w, w)


def cart_rect(agv: AGVConfig):
    """Cart outline in the CART frame (the pose stored in res.cart_x / cart_y / cart_th).

    tow: origin = cart fixed axle, +x towards the hitch.  lurking: origin = hook point."""
    c = agv.cart
    if agv.hook_type == "tow":
        return (-c.axle_from_rear, c.length - c.axle_from_rear, -c.width / 2, c.width / 2)
    if agv.hook_type == "lurking":
        return (-c.length / 2, c.length / 2, -c.width / 2, c.width / 2)
    return None


def lurking_body_rect(agv: AGVConfig):
    """Carried lurking cart in the AGV body frame."""
    c, h = agv.cart, agv.hook
    return (h.x - c.length / 2, h.x + c.length / 2, h.y - c.width / 2, h.y + c.width / 2)


def rigid_rects(agv: AGVConfig, with_cart: bool = True):
    """Rectangles rigidly attached to the AGV body (chassis + carried lurking cart)."""
    rects = [body_rect(agv)]
    if with_cart and agv.hook_type == "lurking":
        rects.append(lurking_body_rect(agv))
    return rects


def rect_corners(r):
    x0, x1, y0, y1 = r
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def swept_radii(rects, kappa: float):
    """Outer / inner radius swept by the rigid footprint turning with signed curvature kappa."""
    R = 1.0 / kappa
    outer, inner = 0.0, math.inf
    for r in rects:
        x0, x1, y0, y1 = r
        for cx, cy in rect_corners(r):
            outer = max(outer, math.hypot(cx, cy - R))
        # distance from the turning centre (0, R) to the rectangle
        dx = max(x0, 0.0, -x1)
        dy = max(y0 - R, 0.0, R - y1)
        inner = min(inner, math.hypot(dx, dy))
    return outer, inner


def _transform(pts, x, y, th):
    c, s = math.cos(th), math.sin(th)
    return np.column_stack([x + c * pts[:, 0] - s * pts[:, 1], y + s * pts[:, 0] + c * pts[:, 1]])


def _grid(rect, spacing, margin):
    """Points on the outline (every `spacing`) plus a coarse interior grid. An obstacle can only
    get inside the footprint by crossing its outline, so the outline carries the check."""
    x0, x1, y0, y1 = rect
    x0, x1, y0, y1 = x0 - margin, x1 + margin, y0 - margin, y1 + margin
    nx = max(2, math.ceil((x1 - x0) / spacing) + 1)
    ny = max(2, math.ceil((y1 - y0) / spacing) + 1)
    xs, ys = np.linspace(x0, x1, nx), np.linspace(y0, y1, ny)
    edge = np.vstack([np.column_stack([xs, np.full(nx, y0)]), np.column_stack([xs, np.full(nx, y1)]),
                      np.column_stack([np.full(ny, x0), ys]), np.column_stack([np.full(ny, x1), ys])])
    X, Y = np.meshgrid(np.linspace(x0, x1, max(2, nx // 4)), np.linspace(y0, y1, max(2, ny // 4)))
    return np.vstack([edge, np.column_stack([X.ravel(), Y.ravel()])])


def _sample_indices(res: SimResult, reach: float, max_move: float = 0.03):
    speed = np.abs(res.v) + np.abs(res.w) * reach
    vmax = max(float(speed.max()), 1e-6)
    step = max(1, int(max_move / (vmax * res.agv.simulation.dt)))
    idx = np.arange(0, len(res.t), step)
    if idx[-1] != len(res.t) - 1:
        idx = np.append(idx, len(res.t) - 1)
    return idx


# ----------------------------------------------------------------------------- collisions
def collision_scan(res: SimResult, fmap: FactoryMap, margin: float) -> List[dict]:
    agv = res.agv
    sp = max(agv.simulation.collision_spacing, 0.5 * fmap.mpp)
    parts = [("AGV", _grid(body_rect(agv), sp, margin), None)]
    if agv.hook_type in ("tow", "lurking") and res.cart_x is not None:
        cg = _grid(cart_rect(agv), sp, margin)
        trains = res.train or [{"x": res.cart_x, "y": res.cart_y, "th": res.cart_th}]
        for k, tr in enumerate(trains):
            lab = ("carried cart" if agv.hook_type == "lurking" else
                   ("towed cart" if len(trains) == 1 else f"trolley {k + 1}"))
            parts.append((lab, cg, tr))
    reach = max(np.abs(rect_corners(body_rect(agv))).max(), 1.0) * 1.5
    idx_all = _sample_indices(res, reach)
    episodes = []
    for label, pts, frame in parts:
        if frame is None:
            idx = idx_all
            X0, Y0, TH = res.x[idx], res.y[idx], res.th[idx]
        else:
            idx = idx_all[np.isfinite(frame["x"][idx_all])]
            if not len(idx):
                continue
            X0, Y0, TH = frame["x"][idx], frame["y"][idx], frame["th"][idx]
        hit = np.zeros(len(idx), bool)
        hit_xy = np.zeros((len(idx), 2))
        for c0 in range(0, len(idx), 1500):
            sl = slice(c0, c0 + 1500)
            c, s = np.cos(TH[sl])[:, None], np.sin(TH[sl])[:, None]
            X = X0[sl][:, None] + c * pts[None, :, 0] - s * pts[None, :, 1]
            Y = Y0[sl][:, None] + s * pts[None, :, 0] + c * pts[None, :, 1]
            obs = fmap.is_obstacle(X, Y)
            h = obs.any(axis=1)
            hit[sl] = h
            first = obs.argmax(axis=1)
            rows = np.arange(X.shape[0])
            hit_xy[sl] = np.column_stack([X[rows, first], Y[rows, first]])
        # group consecutive hits into episodes
        k = 0
        while k < len(idx):
            if hit[k]:
                k1 = k
                while k1 + 1 < len(idx) and hit[k1 + 1] and idx[k1 + 1] - idx[k1] < 50:
                    k1 += 1
                episodes.append({"part": label, "t_start": float(res.t[idx[k]]),
                                 "t_end": float(res.t[idx[k1]]),
                                 "x": float(hit_xy[k, 0]), "y": float(hit_xy[k, 1])})
                k = k1 + 1
            else:
                k += 1
    episodes.sort(key=lambda e: e["t_start"])
    return episodes


def swept_mask(res: SimResult, fmap: FactoryMap) -> np.ndarray:
    agv = res.agv
    img = Image.new("L", (fmap.w, fmap.h), 0)
    dr = ImageDraw.Draw(img)
    body_polys = [rect_corners(body_rect(agv))]
    cart_poly = rect_corners(cart_rect(agv)) if (agv.hook_type in ("tow", "lurking")
                                                 and res.cart_x is not None) else None
    idx = _sample_indices(res, 2.0, max_move=0.5 * fmap.mpp + 0.02)
    trains = (res.train or [{"x": res.cart_x, "y": res.cart_y, "th": res.cart_th}]) if cart_poly is not None else []
    for i in idx:
        polys = [_transform(p, res.x[i], res.y[i], res.th[i]) for p in body_polys]
        for tr in trains:
            if np.isfinite(tr["x"][i]):
                polys.append(_transform(cart_poly, tr["x"][i], tr["y"][i], tr["th"][i]))
        for p in polys:
            px, py = fmap.to_pixel_float(p[:, 0], p[:, 1])
            dr.polygon(list(zip(px.tolist(), py.tolist())), fill=255)
    return np.asarray(img) > 0


# ----------------------------------------------------------------------------- laps
def laps(res: SimResult, home_radius: float = 1.5) -> List[dict]:
    """Split the simulated run into loops that start and end at the first route point.

    A lap runs from one departure at the start point to the next (so it contains one stop /
    wait at the start point). A closed-loop route (loop: true) is one lap = the cycle."""
    t, v = res.t, res.v
    T = float(t[-1])
    wps = res.route.waypoints
    p0 = (wps[0].x, wps[0].y)
    home = {i for i, w in enumerate(wps) if math.hypot(w.x - p0[0], w.y - p0[1]) <= home_radius}
    runs = {r.index: r for r in res.plan.runs}
    starts = [e.t0 for e in res.events if e.kind == "drive" and e.run_index in runs
              and runs[e.run_index].start_wp in home]
    returns_home = any(runs[e.run_index].end_wp in home for e in res.events
                       if e.kind == "drive" and e.run_index in runs)
    if res.plan.loop or not starts:
        bounds = [(0.0, T)]
    else:
        starts = sorted(set(starts))
        bounds = [(a, b) for a, b in zip(starts, starts[1:] + [T])]
        if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] < 1e-6:
            bounds = bounds[:-1]
    dt = np.diff(t, prepend=t[0])
    out = []
    for k, (a, b) in enumerate(bounds):
        m = (t > a) & (t <= b)
        dist = float(np.sum(np.abs(v[m]) * dt[m]))
        drv = m & (res.mode == DRIVE)
        tdrv = float(dt[drv].sum())
        T_lap = b - a
        stops = sum(1 for e in res.events if e.kind == "drive" and a < e.t1 <= b + 1e-9)
        out.append({"lap": k + 1, "t0": a, "t1": b, "time_s": T_lap, "drive_time_s": tdrv,
                    "distance_m": dist, "avg_speed": dist / T_lap if T_lap > 0 else 0.0,
                    "avg_speed_driving": dist / tdrv if tdrv > 0 else 0.0,
                    "max_speed": float(np.max(np.abs(v[m]))) if m.any() else 0.0, "stops": stops,
                    "closed": bool(res.plan.loop or returns_home)})
    return out


# ----------------------------------------------------------------------------- main analysis
def analyze(res: SimResult, fmap: Optional[FactoryMap] = None) -> dict:
    agv = res.agv
    t = res.t
    T = float(t[-1])
    L = res.loads
    dtr = agv.drivetrain
    warnings = list(res.warnings)
    checks = []

    kinds = {"drive": 0.0, "pivot": 0.0, "dwell": 0.0}
    for e in res.events:
        kinds[e.kind] += e.duration
    driving = res.mode == DRIVE
    dist = float(_trapz(np.abs(res.v), t))
    drive_time = kinds["drive"]

    s = {
        "agv": agv.name, "route": res.route.name,
        "cycle_time_s": T,
        "drive_time_s": kinds["drive"], "pivot_time_s": kinds["pivot"], "dwell_time_s": kinds["dwell"],
        "cycles_per_hour": 3600.0 / T if T > 0 else 0.0,
        "path_length_m": res.plan.length,
        "distance_axle_m": dist,
        "avg_speed_driving": dist / drive_time if drive_time > 0 else 0.0,
        "avg_speed_overall": dist / T if T > 0 else 0.0,
        "max_speed": float(np.max(np.abs(res.v))),
        "max_wheel_speed_limit": agv.max_wheel_speed,
        "events": [{"kind": e.kind, "label": e.label, "t0": e.t0, "t1": e.t1, "duration": e.duration,
                    "distance": e.distance, "angle_deg": e.angle_deg, "mode": e.mode} for e in res.events],
    }
    s["laps"] = laps(res)
    if agv.max_wheel_speed < agv.performance.v_straight:
        warnings.append(f"v_straight {agv.performance.v_straight:.2f} m/s exceeds what the motor can reach "
                        f"({agv.max_wheel_speed:.2f} m/s at {dtr.motor_max_rpm:.0f} rpm / 1:{dtr.gear_ratio:g}).")

    # ---- turning ----------------------------------------------------------------
    ok = driving & (np.abs(res.v) > 0.02) & (np.abs(res.w) > 1e-3)
    kap_all = res.w[ok] / res.v[ok] if ok.any() else np.array([0.0])
    kmax = float(np.max(np.abs(kap_all)))
    s["min_turn_radius_axle_m"] = 1.0 / kmax if kmax > 1e-6 else math.inf
    att = res.attached if res.attached is not None else np.zeros(len(t), bool)
    lurk_any = agv.hook_type == "lurking" and bool(att.any())
    rects = rigid_rects(agv, with_cart=lurk_any)
    s["pivot_swept_radius_m"] = float(max(np.hypot(*rect_corners(r).T).max() for r in rects))
    s["jerk"] = agv.performance.jerk
    s["modes_used"] = sorted({m.strip() for e in res.events if e.kind == "drive"
                              for m in (e.mode or "default").split(",")})

    path_pts = {}
    arcs = []
    for run in res.plan.runs:
        step = max(1, int(0.05 / agv.simulation.ds))
        path_pts[run.index] = np.column_stack([run.x[::step], run.y[::step]])
        for arc in run.arcs:
            if abs(math.degrees(arc.turn)) < 5.0:     # practically straight: not a curve
                continue
            m = (res.run_id == run.index) & driving & (res.ref_s >= arc.s_start) & (res.ref_s <= arc.s_end)
            idx = np.nonzero(m)[0]
            if len(idx) < 2:
                continue
            vv, ww = res.v[idx], res.w[idx]
            good = np.abs(vv) > 0.02
            if not good.any():
                continue
            kk = ww[good] / vv[good]
            k_t = float(kk[np.argmax(np.abs(kk))])
            R_ax = 1.0 / abs(k_t) if abs(k_t) > 1e-9 else math.inf
            rr = rigid_rects(agv, with_cart=agv.hook_type == "lurking" and bool(att[idx].mean() > 0.5))
            outer, inner = swept_radii(rr, k_t) if abs(k_t) > 1e-9 else (math.inf, math.inf)
            item = {"label": arc.label, "run": run.label, "path_radius": arc.radius,
                    "requested_radius": arc.requested_radius, "turn_deg": math.degrees(arc.turn),
                    "speed": float(np.mean(np.abs(vv))), "time_s": float(t[idx[-1]] - t[idx[0]]),
                    "axle_radius_min": R_ax, "inner_wheel_radius": R_ax - agv.wheels.track / 2,
                    "outer_wheel_radius": R_ax + agv.wheels.track / 2,
                    "outer_swept_radius": outer, "inner_swept_radius": inner,
                    "swept_width": outer - inner, "trailer_offtrack": None,
                    "pv_max_mm": (float(np.nanmax(np.abs(res.lf["pv"][idx])))
                                  if res.lf is not None and np.isfinite(res.lf["pv"][idx]).any() else None)}
            if agv.hook_type == "tow" and res.cart_x is not None:
                ntr = max(1, len(res.train))
                extra = ntr * (agv.cart.tongue_length + agv.cart.coupling_behind_axle) + abs(agv.hook.x)
                extra /= max(item["speed"], 0.05)
                win = (res.run_id == run.index) & driving & (t >= t[idx[0]]) & (t <= t[idx[-1]] + extra)
                off = 0.0
                for tr in (res.train or [{"x": res.cart_x, "y": res.cart_y}]):
                    wi = np.nonzero(win & np.isfinite(tr["x"]))[0][::3]
                    if len(wi):
                        P = path_pts[run.index]
                        cx, cy = tr["x"][wi], tr["y"][wi]
                        dmin = np.min(np.hypot(cx[:, None] - P[None, :, 0], cy[:, None] - P[None, :, 1]), axis=1)
                        off = max(off, float(dmin.max()))
                item["trailer_offtrack"] = off
            arcs.append(item)
    s["arcs"] = arcs

    # ---- drivetrain ---------------------------------------------------------------
    rpm = np.maximum(np.abs(L["motor_rpm_l"]), np.abs(L["motor_rpm_r"]))
    tq_l, tq_r = L["motor_torque_l"], L["motor_torque_r"]
    tq_peak = float(max(np.abs(tq_l).max(), np.abs(tq_r).max()))
    tq_rms = float(max(math.sqrt(_trapz(tq_l ** 2, t) / T), math.sqrt(_trapz(tq_r ** 2, t) / T)))
    s["motor"] = {
        "rated_torque_nm": dtr.rated_torque, "peak_torque_allowed_nm": dtr.peak_torque,
        "peak_torque_nm": tq_peak, "rms_torque_nm": tq_rms,
        "peak_rpm": float(rpm.max()), "max_rpm_allowed": dtr.motor_max_rpm,
        "peak_power_w": float(max(L["motor_power_l"].max(), L["motor_power_r"].max())),
        "peak_wheel_rpm": float(max(np.abs(L["wheel_rpm_l"]).max(), np.abs(L["wheel_rpm_r"]).max())),
        "cont_tractive_force_n": 2 * dtr.rated_torque * dtr.gear_ratio * dtr.gear_efficiency / agv.wheel_radius,
        "peak_tractive_force_n": 2 * dtr.peak_torque * dtr.gear_ratio * dtr.gear_efficiency / agv.wheel_radius,
    }
    mo = s["motor"]
    if mo["peak_rpm"] > dtr.motor_max_rpm * 1.001:
        warnings.append(f"Motor speed {mo['peak_rpm']:.0f} rpm exceeds max {dtr.motor_max_rpm:.0f} rpm.")
    if tq_peak > dtr.peak_torque:
        warnings.append(f"Peak motor torque {tq_peak:.2f} N*m exceeds motor peak {dtr.peak_torque:.2f} N*m "
                        f"-> reduce accel/decel or payload, or use a bigger motor / higher ratio.")
    if tq_rms > dtr.rated_torque:
        warnings.append(f"RMS motor torque {tq_rms:.2f} N*m exceeds rated {dtr.rated_torque:.2f} N*m "
                        f"-> motor will overheat in continuous cycling.")
    s["motor"]["castor_swivel_force_peak_n"] = L.get("castor_swivel_force_peak", 0.0)
    s["motor"]["drawbar_force_peak_n"] = float(np.max(np.abs(L["drawbar_force"]))) if "drawbar_force" in L else 0.0
    if agv.hook_type == "tow" and "drawbar_force" in L and len(L["drawbar_force"]):
        push = -np.asarray(L["drawbar_force"], float)          # compression: the trolleys push the AGV
        k = int(np.argmax(push))
        allow = 0.10 * agv.chassis.mass_kg * 9.81
        s["trolley_push"] = {"max_n": max(0.0, float(push[k])), "t": float(res.t[k]), "x": float(res.x[k]),
                             "y": float(res.y[k]), "allow_n": allow,
                             "over_s": float(np.sum(push > allow) * agv.simulation.dt)}
    trac = float(max(L["traction_l"].max(), L["traction_r"].max()))
    s["traction_usage_max"] = trac
    if trac > 1.0:
        # where: every stretch over the grip, worst spot of each (like the tape-lost markers)
        tr_ = np.maximum(L["traction_l"], L["traction_r"])
        over = np.nonzero(tr_ > 1.0)[0]
        groups = np.split(over, np.nonzero(np.diff(over) > 1)[0] + 1) if len(over) else []
        slips = []
        for g in groups:
            k = int(g[np.argmax(tr_[g])])
            what = ("braking" if res.a[k] < -0.05 else "speeding up" if res.a[k] > 0.05 else
                    "turning" if abs(res.w[k]) > 0.05 else "driving")
            slips.append({"t": float(res.t[k]), "x": float(res.x[k]), "y": float(res.y[k]),
                          "pct": float(100 * tr_[k]), "dur_s": float(len(g) * agv.simulation.dt), "what": what,
                          "with_train": bool(res.attached is not None and res.attached[k] and agv.hook_type == "tow")})
        slips.sort(key=lambda d: -d["pct"])
        s["slip_spots"] = slips[:30]
        worst = slips[0]
        whats = sorted({d["what"] for d in slips})
        warnings.append(
            f"WHEEL SLIP at {len(slips)} spot(s) (red marks): worst {worst['pct']:.0f}% of the available grip at t = "
            f"{worst['t']:.1f} s ({worst['x']:.1f}, {worst['y']:.1f}) while {worst['what']}"
            + (" with the trolleys" if worst["with_train"] else "")
            + f"; also while {', '.join(k for k in whats if k != worst['what'])}" * (len(whats) > 1)
            + f". mu = {agv.wheels.friction_coeff}. The real AGV cannot do this: it would spin / slide its wheels"
            + (" - with 'Trolley load calc' off the simulation still assumes the full ramps, so the cycle time "
               "here is optimistic" if not agv.simulation.load_effects else "")
            + ". Lower accel / decel, or more weight on the drive wheels.")

    # ---- energy -----------------------------------------------------------------------
    e = L["energy_wh"]
    usable = dtr.battery_voltage * dtr.battery_capacity_ah * dtr.battery_usable_fraction
    s["energy"] = {"wh_per_cycle": e, "avg_power_w": e * 3600 / T,
                   "battery_usable_wh": usable,
                   "cycles_per_charge": usable / e if e > 0 else math.inf,
                   "runtime_h": (usable / e) * T / 3600 if e > 0 else math.inf}

    # ---- closed-loop tape following ----------------------------------------------------------
    if res.lf is not None:
        lf, cfg = res.lf, agv.line_follow
        pv = lf["pv"]
        drv = driving & np.isfinite(pv)
        rng = cfg.sensor_range_mm
        info = {"range_mm": rng, "aborted": bool(lf["aborted"]),
                "events": [{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in e.items()}
                           for e in lf["events"]],
                "stops": lf["runs"]}
        if drv.any():
            ap = np.abs(pv[drv])
            k = int(np.argmax(np.where(drv, np.abs(np.nan_to_num(pv)), -1)))
            info.update(max_abs_mm=float(ap.max()), rms_mm=float(np.sqrt(np.mean(ap ** 2))),
                        margin_mm=float(rng - ap.max()),
                        near_edge_s=float(np.sum(drv & (np.abs(np.nan_to_num(pv)) > 0.75 * rng)) * agv.simulation.dt),
                        worst={"t": float(t[k]), "x": float(res.x[k]), "y": float(res.y[k]), "pv": float(pv[k])},
                        sat_frac=float(np.mean(lf["sat"][driving])) if driving.any() else 0.0)
        losses = [e for e in lf["events"] if e.get("kind") == "tape lost"]
        info["losses"] = len(losses)
        s["line_follow"] = info
        s["cycle_complete"] = not lf["aborted"]
        if losses and not lf["aborted"]:
            warnings.append(f"Tape lost {len(losses)}x but re-acquired while braking (AGV stops and restarts).")
        if not losses and drv.any() and info["max_abs_mm"] > 0.75 * rng:
            warnings.append(f"Tape error reaches {info['max_abs_mm']:.0f} mm of the {rng:.0f} mm sensor range "
                            f"at ({info['worst']['x']:.1f}, {info['worst']['y']:.1f}) - close to losing the track.")
        for e in lf["events"]:
            if e.get("kind") == "warning":
                warnings.append(e["text"])
            elif e.get("kind") == "note":
                warnings.append("Note: " + e["text"])
        # speed per speed_mode (what the AGV really drove, incl. ramps and speed reduction)
        rows = []
        dtv = np.diff(t, prepend=t[0])
        for code, name in enumerate(lf.get("mode_names", [])):
            m0 = driving & (np.round(lf["sm"]) == code)
            if not m0.any():
                continue
            att = res.attached if (res.attached is not None and agv.hook_type != "none"
                                   and agv.simulation.load_effects) else None
            split = att is not None and att[m0].any() and (~att[m0]).any()
            parts = ([(m0 & att, "loaded"), (m0 & ~att, "empty")] if split else
                     [(m0, ("loaded" if att[m0].all() else "empty") if att is not None else "")])
            for m, load in parts:
                if not m.any():
                    continue
                tt = float(dtv[m].sum())
                dist = float(np.sum(np.abs(res.v[m]) * dtv[m]))
                rows.append({"mode": name, "target": lf.get("mode_speeds", {}).get(name), "load": load,
                             "avg": dist / tt if tt > 0 else 0.0, "max": float(np.max(np.abs(res.v[m]))),
                             "time_s": tt, "distance_m": dist})
        s["speed_by_mode"] = rows
        if rows:
            s["modes_used"] = list(dict.fromkeys(r["mode"] for r in rows))

    # ---- what the load does to speeding up / braking (flat floor) ---------------------------
    if agv.hook_type in ("tow", "lurking") and agv.simulation.load_effects:
        from .drive import load_table
        s["load_table"] = load_table(agv)

    # ---- stability ---------------------------------------------------------------------------
    s["lat_accel_max"] = float(np.max(L["lat_accel"]))
    tr = float(np.max(L["tip_ratio"]))
    s["tip_safety_agv"] = (1.0 / tr) if tr > 1e-9 else math.inf
    if s["tip_safety_agv"] < 1.0:
        warnings.append(f"AGV TIPS OVER: inertial load reaches {tr * 100:.0f}% of the tipping limit "
                        f"(CG height {agv.chassis.cg_height:.2f} m). Lower speed/accel or CG.")
    elif s["tip_safety_agv"] < 2.0:
        warnings.append(f"Low tip-over margin for the AGV: safety factor {s['tip_safety_agv']:.1f} (want >= 2).")
    if "cart_tip_ratio" in L:
        ctr = float(np.max(L["cart_tip_ratio"]))
        s["cart_lat_accel_max"] = float(np.max(L["cart_lat_accel"]))
        s["tip_safety_cart"] = (1.0 / ctr) if ctr > 1e-9 else math.inf
        if s["tip_safety_cart"] < 1.0:
            warnings.append(f"TOWED CART TIPS OVER in curves (lateral accel {s['cart_lat_accel_max']:.2f} m/s2, "
                            f"CG {agv.cart.cg_height:.2f} m, wheel track {agv.cart.wheel_track:.2f} m).")
        elif s["tip_safety_cart"] < 2.0:
            warnings.append(f"Low tip-over margin for the towed cart: safety factor {s['tip_safety_cart']:.1f}.")
    if res.grade is not None and np.any(res.grade != 0):
        s["max_grade_pct"] = float(np.max(np.abs(res.grade)))

    # ---- cart ----------------------------------------------------------------------------
    if res.articulation_deg is not None and np.isfinite(res.articulation_deg).any():
        arts = [float(np.nanmax(np.abs(tr["art"]))) for tr in res.train if "art" in tr
                and np.isfinite(tr["art"]).any()] or [float(np.nanmax(np.abs(res.articulation_deg)))]
        s["articulation_max_deg"] = max(arts)
        s["articulation_joints_deg"] = arts
    if agv.hook_type in ("tow", "lurking"):
        ms = agv.train_masses()
        s["train"] = {"count": len(ms), "masses_kg": ms, "total_kg": sum(ms),
                      "length_m": (len(ms) * (agv.cart.tongue_length + agv.cart.coupling_behind_axle)
                                   if agv.hook_type == "tow" else agv.cart.length)}
    s["cart_events"] = [e.label for e in res.events if e.label.startswith(("Attach cart", "Detach cart"))]

    # ---- static design checks -------------------------------------------------------------
    c, wh, hk, cart = agv.chassis, agv.wheels, agv.hook, agv.cart
    if dtr.layout.lower() == "coaxial":
        need = 2 * (dtr.gearbox_length + dtr.motor_length)
        avail = wh.track - wh.width
        checks.append(f"Coaxial drives need {need * 1000:.0f} mm between wheels, available "
                      f"{avail * 1000:.0f} mm -> {'OK' if need <= avail else 'DOES NOT FIT'}")
        if need > avail:
            warnings.append("Coaxial motor+gearbox do not fit between the drive wheels.")
    if wh.track + wh.width > c.width + 1e-6:
        checks.append("Drive wheels stick out beyond the chassis width.")
    if agv.hook_type == "lurking":
        ok_h = c.height < cart.underside_clearance  # drives under with the lift retracted
        checks.append(f"Lurking: AGV height {c.height * 1000:.0f} mm vs cart underside "
                      f"{cart.underside_clearance * 1000:.0f} mm -> {'OK' if ok_h else 'DOES NOT FIT'}")
        if not ok_h:
            warnings.append("AGV is too tall to drive under the cart.")
    if agv.hook_type == "tow":
        dz = hk.z - cart.hitch_height
        checks.append(f"Tow hitch height {hk.z * 1000:.0f} mm vs cart drawbar {cart.hitch_height * 1000:.0f} mm "
                      f"(diff {dz * 1000:+.0f} mm)")
        if abs(dz) > 0.05:
            warnings.append("Hitch and cart drawbar heights differ by more than 50 mm.")
        if hk.x > agv.x_rear:
            checks.append("Hook sits inside the chassis outline (x > rear bumper) - check it is reachable.")
        ntr = len(agv.train_masses())
        if ntr > 1:
            gap = cart.coupling_behind_axle + cart.tongue_length - cart.length
            checks.append(f"Trolley train: {ntr} trolleys, gap between trolleys on a straight {gap * 100:.0f} cm "
                          f"(coupling {cart.coupling_behind_axle * 100:.0f} cm behind axle + tongue "
                          f"{cart.tongue_length * 100:.0f} cm - length {cart.length * 100:.0f} cm)")
            if gap < 0.05:
                warnings.append(f"The trolleys overlap / touch in the train ({gap * 100:.0f} cm gap): check trolley "
                                f"length, tongue_length (coupling -> fixed axle) and coupling_behind_axle.")
    for sn in agv.sensors:
        inside = agv.x_rear <= sn.x <= agv.x_front and abs(sn.y) <= c.width / 2
        checks.append(f"Sensor '{sn.name}' ({sn.type}) at x={sn.x:+.3f} y={sn.y:+.3f} z={sn.z:.3f} m"
                      f"{'' if inside else '  <- OUTSIDE chassis outline'}")
    s["checks"] = checks

    # ---- map-based checks -------------------------------------------------------------------
    if fmap is not None:
        s["collisions"] = collision_scan(res, fmap, 0.0)
        margin = agv.simulation.safety_margin
        s["clearance_violations"] = collision_scan(res, fmap, margin) if margin > 0 else []
        mask = swept_mask(res, fmap)
        s["swept_area_m2"] = float(mask.sum() * fmap.mpp ** 2)
        s["_swept_mask"] = mask
        if s["collisions"]:
            warnings.append(f"{len(s['collisions'])} COLLISION episode(s) with map obstacles.")
        if s["clearance_violations"]:
            warnings.append(f"{len(s['clearance_violations'])} place(s) closer than the "
                            f"{margin * 100:.0f} cm safety margin.")
    # ---- what the numbers do / do not include --------------------------------------------
    if agv.hook_type in ("tow", "lurking") and res.attached is not None and res.attached.any() \
            and not agv.simulation.load_effects:
        warnings.append("Note: 'Trolley load calc' is off - speed and timing do not depend on the trolley / cart "
                        "weight (it only enters the forces, torque, grip and energy above). Switch it on to let the "
                        "weight limit speeding up and braking.")
    unc = [f"sensors.{sn.name}: {sn.note.replace('TODO', '').strip() or 'placeholder'}"
           for sn in agv.sensors if "TODO" in (getattr(sn, "note", "") or "")]
    unc += list(getattr(agv, "_todo_fields", []) or [])
    if unc:
        s["uncalibrated"] = unc
    tgt = float(getattr(res.route, "target_cycle_s", 0.0) or 0.0)
    if tgt > 0:
        s["target_cycle_s"] = tgt
        s["target_gap_s"] = T - tgt
        if T > tgt + 0.05:
            warnings.append(f"Cycle time {T:.1f} s is {T - tgt:.1f} s ABOVE your target {tgt:.1f} s "
                            f"(driving {kinds['drive']:.0f} s, waiting {kinds['dwell']:.0f} s, turning {kinds['pivot']:.0f} s). "
                            f"Auto route can help; faster speed modes, shorter waits or fewer stops too.")
    s["warnings"] = warnings
    return s


# ----------------------------------------------------------------------------- report text
def format_report(s: dict) -> str:
    L = []
    add = L.append
    add("=" * 72)
    add(f" AGV SIMULATION REPORT   AGV: {s['agv']}   Route: {s['route']}")
    add("=" * 72)
    ct = s["cycle_time_s"]
    if s.get("cycle_complete") is False:
        add(f"CYCLE NOT COMPLETED   : the AGV lost the tape at t = {ct:.1f} s (see Line following)")
    add(f"Cycle time            : {ct:8.1f} s  ({ct / 60:.2f} min)  -> {s['cycles_per_hour']:.1f} cycles/h")
    if s.get("uncalibrated"):
        add(f"Uncalibrated inputs   : {len(s['uncalibrated'])} value(s) still marked TODO (see the end of the report)")
    if s.get("target_cycle_s"):
        g = s["target_gap_s"]
        add(f"  target              : {s['target_cycle_s']:8.1f} s  -> {abs(g):.1f} s {'OVER' if g > 0.05 else 'to spare'}")
    add(f"  driving             : {s['drive_time_s']:8.1f} s")
    add(f"  pivoting            : {s['pivot_time_s']:8.1f} s")
    add(f"  dwelling (stations) : {s['dwell_time_s']:8.1f} s")
    add(f"Path length           : {s['path_length_m']:8.2f} m   (axle travelled {s['distance_axle_m']:.2f} m)")
    add(f"Average driving speed : {s['avg_speed_driving']:8.3f} m/s   max {s['max_speed']:.3f} m/s "
        f"(motor limit {s['max_wheel_speed_limit']:.3f} m/s)")
    add(f"Average incl. stops   : {s.get('avg_speed_overall', 0):8.3f} m/s   (distance / cycle time)")
    lp = s.get("laps") or []
    if lp and lp[0]["closed"]:
        for L_ in lp:
            add(f"  Loop {L_['lap']:<2d} {L_['t0']:7.1f}-{L_['t1']:7.1f} s : {L_['time_s']:6.1f} s  {L_['distance_m']:7.1f} m  "
                f"avg {L_['avg_speed']:.3f} m/s incl. stops, {L_['avg_speed_driving']:.3f} m/s driving, {L_['stops']} stop(s)")
    for r in s.get("speed_by_mode", []):
        tg = f"target {r['target']:.2f}" if r.get("target") is not None else "target  -  "
        ld = f" ({r['load']})" if r.get("load") else ""
        add(f"  speed_mode {r['mode'] + ld:18s}: avg {r['avg']:.3f} m/s  max {r['max']:.3f}  {tg}  "
            f"{r['time_s']:6.1f} s  {r['distance_m']:6.1f} m")
    if s.get("load_table"):
        add("Load (flat floor)     :  load           mass   speed-up  brake   0->top   stop from top  max slope")
        for r in s["load_table"]:
            add(f"  {r['label']:<20s} {r['load_kg']:5.0f} kg {r['mass_kg']:5.0f} kg  {r['accel']:5.2f}    {r['brake']:5.2f}  "
                f"{r['t_to_top']:5.1f} s  {100 * r['brake_dist']:6.1f} cm     {r['max_slope_pct']:4.1f} %  ({r['limit']})")
    r = s["min_turn_radius_axle_m"]
    add(f"Tightest turn (axle)  : {r:8.2f} m" if math.isfinite(r) else "Tightest turn (axle)  :   (no curves)")
    add(f"Pivot swept radius    : {s['pivot_swept_radius_m']:8.2f} m  (clear circle needed to spin in place)")
    add(f"Speed ramps           : {'S-curve, jerk %.2f m/s3' % s['jerk'] if s.get('jerk') else 'trapezoid (no jerk limit)'}"
        f"   speed modes used: {', '.join(s.get('modes_used') or ['default'])}")
    if s.get("train"):
        tr = s["train"]
        add(f"Trolleys              : {tr['count']} x  ({', '.join('%.0f' % m for m in tr['masses_kg'])} kg) = "
            f"{tr['total_kg']:.0f} kg")
    if "articulation_max_deg" in s:
        js = s.get("articulation_joints_deg") or []
        add(f"Max cart articulation : {s['articulation_max_deg']:8.1f} deg"
            + (f"   (per joint: {', '.join('%.0f' % a for a in js)})" if len(js) > 1 else ""))
    tsa = s.get("tip_safety_agv", math.inf)
    add(f"Tip-over safety AGV   : {tsa:8.1f} x   (max lateral accel {s.get('lat_accel_max', 0):.2f} m/s2)"
        if math.isfinite(tsa) else "Tip-over safety AGV   :   (no lateral / braking load)")
    if "tip_safety_cart" in s:
        tsc = s["tip_safety_cart"]
        add(f"Tip-over safety cart  : {tsc:8.1f} x   (max lateral accel {s['cart_lat_accel_max']:.2f} m/s2)"
            if math.isfinite(tsc) else "Tip-over safety cart  :   (no lateral load)")
    if s.get("max_grade_pct"):
        add(f"Steepest slope        : {s['max_grade_pct']:8.1f} %")
    if "swept_area_m2" in s:
        add(f"Swept floor area      : {s['swept_area_m2']:8.1f} m2")

    add("\n-- Segment timeline " + "-" * 52)
    for e in s["events"]:
        extra = f"{e['distance']:.2f} m" if e["kind"] == "drive" else (
            f"{e['angle_deg']:+.0f} deg" if e["kind"] == "pivot" else "")
        md = f"  [{e['mode']}]" if e.get("mode") and e["mode"] != "default" else ""
        avg = f"  avg {e['distance'] / e['duration']:.2f} m/s" if e["kind"] == "drive" and e["duration"] > 0 else ""
        add(f"  {e['t0']:7.1f}-{e['t1']:7.1f} s  {e['kind']:5s} {e['duration']:6.1f} s  {extra:>9s}  {e['label']}{md}{avg}")

    if s["arcs"]:
        add("\n-- Curves " + "-" * 62)
        add("  waypoint          R_path  R_axle  R_inWhl  R_outSwept R_inSwept  width  speed  time  cart_off  tape_err")
        for a in s["arcs"]:
            off = f"{a['trailer_offtrack']:.2f}" if a["trailer_offtrack"] is not None else "  -"
            add(f"  {a['label'][:16]:16s} {a['path_radius']:6.2f}  {a['axle_radius_min']:6.2f}  "
                f"{a['inner_wheel_radius']:7.2f}  {a['outer_swept_radius']:9.2f} {a['inner_swept_radius']:9.2f}  "
                f"{a['swept_width']:5.2f}  {a['speed']:5.2f} {a['time_s']:5.1f}  {off:>7s}  "
                f"{'%5.0f mm' % a['pv_max_mm'] if a.get('pv_max_mm') is not None else '     -'}")
        add("  (m, m/s, s; cart_off = max distance of towed-cart axle from the path = off-tracking)")

    lfi = s.get("line_follow")
    if lfi:
        add("\n-- Line following (closed loop, your PID) " + "-" * 30)
        if "max_abs_mm" in lfi:
            add(f"  Tape error max      : {lfi['max_abs_mm']:7.0f} mm of +-{lfi['range_mm']:.0f} mm sensor range "
                f"(margin {lfi['margin_mm']:.0f} mm) at ({lfi['worst']['x']:.2f}, {lfi['worst']['y']:.2f}) "
                f"t={lfi['worst']['t']:.1f} s")
            add(f"  Tape error RMS      : {lfi['rms_mm']:7.1f} mm     time beyond 75 % of range: {lfi['near_edge_s']:.1f} s")
            add(f"  Command saturated   : {100 * lfi['sat_frac']:7.0f} % of driving time (5 V AO limit or PID clamp)")
        add(f"  Track lost          : {'YES - cycle aborted' if lfi['aborted'] else ('%d x (re-acquired)' % lfi['losses'] if lfi['losses'] else 'no')}")
        for e in lfi["events"]:
            if e.get("kind") in ("tape lost", "reacquired", "no tape at start", "timeout"):
                add(f"    {e['kind']:16s} t={e['t']:7.1f} s at ({e['x']:.2f}, {e['y']:.2f})  [{e.get('run', '')}]")
        for r in lfi["stops"]:
            if r.get("stop_overshoot") is not None:
                add(f"  Stop after tag      : {100 * r['stop_overshoot']:7.1f} cm past the stop tag  [{r['label']}]")

    m = s["motor"]
    add("\n-- Drivetrain (per motor) " + "-" * 46)
    add(f"  Peak motor speed    : {m['peak_rpm']:7.0f} rpm   (max {m['max_rpm_allowed']:.0f})   "
        f"wheel {m['peak_wheel_rpm']:.1f} rpm")
    add(f"  Peak motor torque   : {m['peak_torque_nm']:7.3f} N*m (motor peak {m['peak_torque_allowed_nm']:.3f})")
    add(f"  RMS motor torque    : {m['rms_torque_nm']:7.3f} N*m (rated {m['rated_torque_nm']:.3f})  "
        f"-> {100 * m['rms_torque_nm'] / m['rated_torque_nm']:.0f}% thermal load")
    add(f"  Peak electrical pwr : {m['peak_power_w']:7.0f} W")
    add(f"  Tractive force      : {m['cont_tractive_force_n']:7.0f} N continuous / "
        f"{m['peak_tractive_force_n']:.0f} N peak (both wheels)")
    add(f"  Grip usage (max)    : {100 * s['traction_usage_max']:7.0f} %   (longitudinal + lateral tyre force)")
    if m.get("drawbar_force_peak_n"):
        add(f"  Drawbar pull (peak) : {m['drawbar_force_peak_n']:7.0f} N")
    if m.get("castor_swivel_force_peak_n"):
        add(f"  Castor swivel force : {m['castor_swivel_force_peak_n']:7.0f} N peak per castor (pivot start / direction change)")
    e = s["energy"]
    add(f"  Energy per cycle    : {e['wh_per_cycle']:7.2f} Wh  (avg {e['avg_power_w']:.0f} W incl. aux)")
    add(f"  Battery             : {e['cycles_per_charge']:7.0f} cycles / charge  ~ {e['runtime_h']:.1f} h")

    add("\n-- Design checks " + "-" * 55)
    for c in s["checks"]:
        add("  " + c)
    if "collisions" in s:
        add("\n-- Map clearance " + "-" * 55)
        if not s["collisions"] and not s["clearance_violations"]:
            add("  No collisions, safety margin respected everywhere.")
        for c in s["collisions"][:15]:
            add(f"  COLLISION  {c['part']:10s} t={c['t_start']:7.1f}-{c['t_end']:7.1f} s at ({c['x']:.2f}, {c['y']:.2f})")
        for c in s["clearance_violations"][:15]:
            add(f"  margin     {c['part']:10s} t={c['t_start']:7.1f}-{c['t_end']:7.1f} s near ({c['x']:.2f}, {c['y']:.2f})")
    if s["warnings"]:
        add("\n-- WARNINGS " + "-" * 60)
        for w in s["warnings"]:
            add("  ! " + w)
    if s.get("uncalibrated"):
        add("\n-- UNCALIBRATED INPUTS (marked TODO: placeholders, measure them) " + "-" * 8)
        for u in s["uncalibrated"]:
            add("  ? " + u)
    add("=" * 72)
    return "\n".join(L)


# ----------------------------------------------------------------------------- CSV export
def save_timeseries(res: SimResult, path: str) -> None:
    """Write the time series (one row per time step) to CSV."""
    L = res.loads
    cols = {"t": res.t, "x": res.x, "y": res.y, "heading_deg": np.degrees(res.th), "v": res.v,
            "omega_dps": np.degrees(res.w), "accel": res.a, "mode": res.mode,
            "wheel_rpm_l": L["wheel_rpm_l"], "wheel_rpm_r": L["wheel_rpm_r"],
            "motor_rpm_l": L["motor_rpm_l"], "motor_rpm_r": L["motor_rpm_r"],
            "motor_torque_l": L["motor_torque_l"], "motor_torque_r": L["motor_torque_r"],
            "power_total_w": L["power_total"], "grip_l": L["traction_l"], "grip_r": L["traction_r"],
            "tip_ratio": L["tip_ratio"]}
    if res.grade is not None:
        cols["grade_pct"] = res.grade
    if res.attached is not None:
        cols["cart_attached"] = res.attached.astype(int)
        cols["cart_mass_kg"] = res.cart_mass
    if res.cart_x is not None:
        cols.update({"cart_x": res.cart_x, "cart_y": res.cart_y})
    if res.articulation_deg is not None:
        cols["articulation_deg"] = res.articulation_deg
        cols["drawbar_n"] = L["drawbar_force"]
    if res.lf is not None:
        cols.update({"tape_pv_mm": res.lf["pv"], "tape_lost": res.lf["lost"], "pid_left_rpm": res.lf["cl"],
                     "pid_right_rpm": res.lf["cr"], "target_speed": res.lf["tv"]})
    np.savetxt(path, np.column_stack(list(cols.values())), delimiter=",", header=",".join(cols),
               comments="", fmt="%.5g")
