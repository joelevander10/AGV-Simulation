"""Turning-radius study: drive one corner (default 90 deg) with a range of radii and with a
pivot (turn on the spot), and compare swept envelope, time, loads and trailer behaviour.

    from agv_sim.turning import turning_study
    study = turning_study(agv, radii=[0.8, 1.0, 1.5, 2.0], angle_deg=90, with_cart=True)
    print(study["text"]); study["figure"].savefig("turning.png")
"""
from __future__ import annotations

import copy
import math
from typing import List, Optional

import numpy as np

from .analysis import analyze, body_rect, cart_rect, rect_corners, swept_mask
from .config import AGVConfig, MapConfig, RouteConfig, Waypoint
from .factory_map import FactoryMap
from .path import build_plan
from .simulate import simulate

DEFAULT_RADII = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]


def _reach(agv: AGVConfig) -> float:
    """Rough max distance of anything (body, cart) from the drive axle."""
    r = float(np.hypot(*rect_corners(body_rect(agv)).T).max())
    if agv.hook_type == "tow":
        c = agv.cart
        ntr = len(agv.train_masses())
        r = max(r, abs(agv.hook.x) + ntr * (c.tongue_length + c.coupling_behind_axle) + c.axle_from_rear + c.width)
    elif agv.hook_type == "lurking":
        c = agv.cart
        r = max(r, math.hypot(abs(agv.hook.x) + c.length / 2, abs(agv.hook.y) + c.width / 2))
    return r


def _corner_route(R: float, angle: float, L: float, margin: float):
    """Straight leg of length L to the corner point, one corner of radius R (0 = pivot) turning
    LEFT by `angle`, and a leg of length L after it. Same start / end points for every R."""
    P0 = np.array([0.0, 0.0])
    P1 = np.array([L, 0.0])
    P2 = P1 + L * np.array([math.cos(angle), math.sin(angle)])
    pts = np.array([P0, P1, P2])
    lo = pts.min(axis=0) - margin
    hi = pts.max(axis=0) + margin
    pts = pts - lo
    size = hi - lo
    wps = [Waypoint(x=float(pts[0, 0]), y=float(pts[0, 1]), name="Start", action="stop"),
           Waypoint(x=float(pts[1, 0]), y=float(pts[1, 1]), name="Corner", radius=float(R)),
           Waypoint(x=float(pts[2, 0]), y=float(pts[2, 1]), name="End", action="stop")]
    return wps, pts, size


def _envelope(mask: np.ndarray, fmap: FactoryMap, center, a_start, a_end):
    """Min / max distance of swept pixels from `center` inside the angular sector [a_start, a_end]."""
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return math.nan, math.nan
    x = (cols + 0.5) * fmap.mpp
    y = (fmap.h - rows - 0.5) * fmap.mpp
    dx, dy = x - center[0], y - center[1]
    ang = np.arctan2(dy, dx)
    span = (a_end - a_start) % (2 * math.pi)
    rel = (ang - a_start) % (2 * math.pi)
    sel = rel <= span
    if not sel.any():
        return math.nan, math.nan
    d = np.hypot(dx[sel], dy[sel])
    return float(d.min()), float(d.max())


def _pivot_envelope(res, agv: AGVConfig) -> float:
    """Largest distance of any body / cart corner from the spin point while pivoting."""
    from .simulate import PIVOT
    idx = np.nonzero(res.mode == PIVOT)[0]
    if not len(idx):
        return math.nan
    px, py = res.x[idx[0]], res.y[idx[0]]
    best = 0.0
    body = rect_corners(body_rect(agv))
    cpoly = rect_corners(cart_rect(agv)) if res.cart_x is not None else None
    for i in idx[::2]:
        c, s = math.cos(res.th[i]), math.sin(res.th[i])
        X = res.x[i] + c * body[:, 0] - s * body[:, 1]
        Y = res.y[i] + s * body[:, 0] + c * body[:, 1]
        best = max(best, float(np.hypot(X - px, Y - py).max()))
        for tr in (res.train if cpoly is not None else []):
            if not np.isfinite(tr["x"][i]):
                continue
            c, s = math.cos(tr["th"][i]), math.sin(tr["th"][i])
            X = tr["x"][i] + c * cpoly[:, 0] - s * cpoly[:, 1]
            Y = tr["y"][i] + s * cpoly[:, 0] + c * cpoly[:, 1]
            best = max(best, float(np.hypot(X - px, Y - py).max()))
    return best


def _outward_bulge(mask, fmap, pts, half_band):
    """How far the swept area sticks out to the OUTSIDE of the corner beyond the straight-running
    band (|lateral| <= half_band) of the incoming and outgoing leg."""
    rows, cols = np.nonzero(mask)
    x = (cols + 0.5) * fmap.mpp
    y = (fmap.h - rows - 0.5) * fmap.mpp
    out = 0.0
    for k, (a, b) in enumerate(((pts[0], pts[1]), (pts[2], pts[1]))):
        d = (b - a) / np.linalg.norm(b - a)
        n = np.array([-d[1], d[0]])
        lat = (x - a[0]) * n[0] + (y - a[1]) * n[1]
        # outside of a LEFT turn = right of leg 1 (lat < 0); for leg 2 walked backwards: left (lat > 0)
        side = -lat if k == 0 else lat
        out = max(out, float(side.max()) - half_band)
    return max(out, 0.0)


def turning_study(agv: AGVConfig, radii: Optional[List[float]] = None, angle_deg: float = 90.0,
                  with_cart: bool = True, show_radius: Optional[float] = None, mpp: float = 0.01,
                  make_figure: bool = True) -> dict:
    a = copy.deepcopy(agv)
    if not with_cart:
        a.hook.type = "none"
    angle = math.radians(max(10.0, min(float(angle_deg), 179.0)))
    radii = sorted({round(float(r), 4) for r in (radii or DEFAULT_RADII) if float(r) > 0})
    reach = _reach(a)
    d_track = max(a.tracking_offset(1), 0.0)
    lead = max(3.0, 1.2 * reach + d_track + 1.0)
    margin = reach + 1.0
    half_band = a.chassis.width / 2
    if a.hook_type in ("tow", "lurking"):
        half_band = max(half_band, a.cart.width / 2)
    perf, dtr = a.performance, a.drivetrain

    rows = []
    cases = {}
    L_leg = lead + (max(radii) if radii else 0.0) * math.tan(angle / 2)
    for R in [0.0] + radii:
        wps, pts, size = _corner_route(R, angle, L_leg, margin)
        route = RouteConfig(name=f"turn R={R:g}", map=MapConfig(image="", meters_per_pixel=mpp),
                            loop=False, default_radius=max(R, 0.0), waypoints=wps)
        fmap = FactoryMap("", mpp, 100, blank_size_m=(float(size[0]), float(size[1])))
        row = {"R": R, "status": "ok", "notes": []}
        try:
            plan = build_plan(route, a)
            res = simulate(a, route, plan)
            s = analyze(res, None)
        except Exception as ex:          # e.g. geometry that cannot be driven
            row.update(status="error", notes=[str(ex)])
            rows.append(row)
            continue
        mask = swept_mask(res, fmap)
        T = s["cycle_time_s"] - s["dwell_time_s"]
        row["time_s"] = T
        row["distance_m"] = s["distance_axle_m"]
        m = s["motor"]
        # loads while TURNING only (start / stop ramps on the straights are not the corner's fault)
        from .simulate import PIVOT
        L = res.loads
        turning = (res.mode == PIVOT) | (np.abs(res.w) > 0.02)
        if not turning.any():
            turning = np.ones(len(res.t), bool)
        tq = np.maximum(np.abs(L["motor_torque_l"]), np.abs(L["motor_torque_r"]))[turning]
        row["peak_torque_pct"] = 100 * float(tq.max()) / m["peak_torque_allowed_nm"]
        row["peak_rpm"] = float(np.maximum(np.abs(L["motor_rpm_l"]), np.abs(L["motor_rpm_r"]))[turning].max())
        row["grip_pct"] = 100 * float(np.maximum(L["traction_l"], L["traction_r"])[turning].max())
        row["grip_pct_whole"] = 100 * s["traction_usage_max"]
        row["tip_agv"] = s.get("tip_safety_agv", math.inf)
        row["tip_cart"] = s.get("tip_safety_cart", math.inf)
        row["articulation_deg"] = s.get("articulation_max_deg")
        lfi = s.get("line_follow")
        row["tape_err_mm"] = lfi.get("max_abs_mm") if lfi else None
        row["tape_lost"] = bool(lfi and (lfi["aborted"] or lfi["losses"]))
        if R > 0 and s["arcs"]:
            arc_s = s["arcs"][0]
            arc = plan.runs[0].arcs[0]
            row.update(R_axle=arc_s["axle_radius_min"], R_inner_wheel=arc_s["inner_wheel_radius"],
                       curve_speed=arc_s["speed"], cart_offtrack=arc_s["trailer_offtrack"])
            cx, cy = arc.center
            a0 = -math.pi / 2            # the corner turns left from a leg along +x: arc starts below the centre
            inner, outer = _envelope(mask, fmap, (cx, cy), a0, a0 + angle)
            row.update(env_outer=outer, env_inner=inner, band=outer - inner, cut_in=R - inner)
            # fastest possible cornering at this radius (ignoring the configured curve speed)
            Rax = arc_s["axle_radius_min"]
            v_lat = math.sqrt(perf.max_lateral_accel * Rax) if perf.max_lateral_accel > 0 else math.inf
            v_mot = a.max_wheel_speed / (1 + a.wheels.track / (2 * Rax))
            row["v_max_at_R"] = min(v_lat, v_mot, perf.v_straight)
            if Rax < a.wheels.track / 2:
                row["notes"].append("inner wheel turns backwards")
            if d_track > 0 and R < d_track:
                row["notes"].append(f"R below sensor offset {d_track:.2f} m: axle cuts deep")
            if R < perf.min_turn_radius:
                row["notes"].append(f"below min_turn_radius {perf.min_turn_radius:g} m")
        else:
            outer = _pivot_envelope(res, a)
            row.update(R_axle=0.0, R_inner_wheel=-a.wheels.track / 2, curve_speed=0.0, env_outer=outer,
                       env_inner=0.0, band=outer, cut_in=None, v_max_at_R=0.0,
                       cart_offtrack=None, pivot_circle=s["pivot_swept_radius_m"])
            row["notes"].append(f"stop + spin; body needs a {2 * s['pivot_swept_radius_m']:.2f} m circle")
        row["bulge"] = _outward_bulge(mask, fmap, pts, half_band)
        # hard failures
        fails = []
        if row.get("tape_lost"):
            fails.append("TAPE LOST")
        if row["articulation_deg"] is not None and row["articulation_deg"] > a.cart.max_articulation_deg:
            fails.append("jackknife")
        if row["peak_torque_pct"] > 100:
            fails.append("motor torque")
        if row["peak_rpm"] > dtr.motor_max_rpm * 1.001:
            fails.append("motor rpm")
        if row["grip_pct"] > 100:
            fails.append("wheel slip")
        if min(row["tip_agv"], row["tip_cart"]) < 1.0:
            fails.append("tips over")
        if fails:
            row["status"] = "FAIL: " + ", ".join(fails)
        elif min(row["tip_agv"], row["tip_cart"]) < 2.0 or row["grip_pct"] > 80 or row["peak_torque_pct"] > 80 \
                or (row.get("tape_err_mm") is not None and lfi and row["tape_err_mm"] > 0.75 * lfi["range_mm"]):
            row["status"] = "marginal"
        rows.append(row)
        cases[R] = (res, s, mask, fmap, pts, plan)

    ok_r = [r["R"] for r in rows if r["R"] > 0 and r["status"] in ("ok", "marginal")]
    min_ok = min(ok_r) if ok_r else None
    sel = show_radius if (show_radius is not None and show_radius in cases) else (
        min(cases, key=lambda k: abs(k - show_radius)) if show_radius is not None and cases else
        (min_ok if min_ok is not None else (radii[0] if radii else 0.0)))
    out = {"agv": a.name, "angle_deg": math.degrees(angle), "with_cart": with_cart and a.hook_type != "none",
           "rows": rows, "min_feasible_radius": min_ok, "selected_radius": sel, "lead_m": lead}
    out["text"] = format_turning(out)
    if make_figure:
        out["figure"] = plot_turning(out, cases, a)
    return out


def _f(v, d=2, w=6):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "-".rjust(w)
    return f"{v:{w}.{d}f}"


def format_turning(study: dict) -> str:
    L = []
    L.append("=" * 118)
    L.append(f" TURNING STUDY  {study['agv']}   corner {study['angle_deg']:.0f} deg   "
             f"{'with cart' if study['with_cart'] else 'without cart'}")
    L.append("=" * 118)
    L.append("  R_path  R_axle  R_inWhl  env_out  env_in   band  bulge  cut_in  v_curve  v_max@R  time_s  "
             "artic  cart_off  torque%  grip%  tape_mm  status")
    for r in study["rows"]:
        if r["status"] == "error":
            L.append(f"  {r['R']:6.2f}  error: {'; '.join(r['notes'])}")
            continue
        lab = "pivot" if r["R"] == 0 else f"{r['R']:6.2f}"
        L.append(f"  {lab:>6s}  {_f(r.get('R_axle'))}  {_f(r.get('R_inner_wheel'), 2, 7)}  "
                 f"{_f(r.get('env_outer'), 2, 7)}  {_f(r.get('env_inner'))}  {_f(r.get('band'), 2, 5)}  "
                 f"{_f(r.get('bulge'), 2, 5)}  {_f(r.get('cut_in'))}  {_f(r.get('curve_speed'), 2, 7)}  "
                 f"{_f(r.get('v_max_at_R'), 2, 7)}  {_f(r.get('time_s'), 1, 6)}  {_f(r.get('articulation_deg'), 0, 5)}  "
                 f"{_f(r.get('cart_offtrack'), 2, 8)}  {_f(r.get('peak_torque_pct'), 0, 7)}  {_f(r.get('grip_pct'), 0, 5)}  "
                 f"{_f(r.get('tape_err_mm'), 0, 7)}  "
                 f"{r['status']}{'  (' + '; '.join(r['notes']) + ')' if r['notes'] else ''}")
    L.append("-" * 118)
    L.append("  R_path = drawn/tape radius, R_axle = radius the drive-axle centre really drives, R_inWhl = inner wheel,")
    L.append("  env_out / env_in = outer / inner edge of the swept floor in the corner (from the curve centre; pivot: from")
    L.append("  the spin point), band = swept width, bulge = how far it sticks out beyond the straight-running lane,")
    L.append("  cut_in = how far the swept area cuts inside the drawn line, v_max@R = fastest cornering the lateral-accel")
    L.append("  and motor limits allow at this radius, time = start -> end of the whole manoeuvre (same start/end points).")
    L.append("  torque% / grip% are measured while turning; accelerating and braking on the straights is left out.")
    L.append("  tape_mm = largest tape offset under the MGS1600 with your PID (closed loop, if enabled); TAPE LOST = the")
    L.append("  AGV would leave the tape in this corner at the speed_mode it starts in.")
    if study["min_feasible_radius"] is not None:
        L.append(f"  Smallest radius without failures: {study['min_feasible_radius']:.2f} m")
    L.append("=" * 118)
    return "\n".join(L)


def plot_turning(study: dict, cases: dict, agv: AGVConfig):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon

    from .visual import agv_shapes

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1])
    rows = [r for r in study["rows"] if r["status"] != "error"]

    def draw_case(ax, R, title):
        if R not in cases:
            ax.set_axis_off()
            return
        res, s, mask, fmap, pts, plan = cases[R]
        ov = np.zeros(mask.shape + (4,))
        ov[mask] = (1.0, 0.55, 0.0, 0.35)
        ax.imshow(ov, extent=fmap.extent, origin="upper", interpolation="nearest")
        for run in plan.runs:
            ax.plot(run.x, run.y, "--", color="#1f6feb", lw=1.2, label="drawn path" if run.index == 0 else None)
        ax.plot(res.x, res.y, color="#e03131", lw=1.2, label="drive axle")
        if res.cart_x is not None and agv.hook_type == "tow":
            ax.plot(res.cart_x, res.cart_y, color="#2b8a3e", lw=1.2, label="cart axle")
        shapes = agv_shapes(agv, with_cart=False)
        cpoly = rect_corners(cart_rect(agv)) if res.cart_x is not None else None
        from .visual import _tf
        for i in np.linspace(0, len(res.t) - 1, 14).astype(int):
            for name, (poly, fc) in shapes.items():
                if name == "chassis":
                    ax.add_patch(Polygon(_tf(poly, res.x[i], res.y[i], res.th[i]), closed=True, fc="none",
                                         ec="#495057", lw=0.6, alpha=0.7))
            for tr in (res.train if cpoly is not None else []):
                if np.isfinite(tr["x"][i]):
                    ax.add_patch(Polygon(_tf(cpoly, tr["x"][i], tr["y"][i], tr["th"][i]), closed=True,
                                         fc="none", ec="#2b8a3e", lw=0.6, alpha=0.7))
        row = next((r for r in rows if r["R"] == R), None)
        if R > 0 and plan.runs[0].arcs:
            c = plan.runs[0].arcs[0].center
            for rad, ls, lab in ((row.get("env_outer"), "-", "outer"), (row.get("env_inner"), "-", "inner")):
                if rad and math.isfinite(rad):
                    ax.add_patch(Circle(c, rad, fc="none", ec="#c92a2a", ls=ls, lw=1))
            ax.plot(*c, "k+")
            ax.set_title(f"{title}\nR {R:.2f} m -> axle {row['R_axle']:.2f} m, swept {row['env_inner']:.2f}"
                         f"-{row['env_outer']:.2f} m (band {row['band']:.2f} m)", fontsize=10)
        else:
            from .simulate import PIVOT
            ip = np.nonzero(res.mode == PIVOT)[0]
            cpt = (res.x[ip[0]], res.y[ip[0]]) if len(ip) else pts[1]
            if math.isfinite(row["env_outer"]):
                ax.add_patch(Circle(cpt, row["env_outer"], fc="none", ec="#c92a2a", lw=1))
            ax.set_title(f"{title}\nstop + spin: swept radius {row['env_outer']:.2f} m", fontsize=10)
        # zoom on the corner
        eo = row.get("env_outer")
        span = (eo if (eo is not None and math.isfinite(eo)) else 3.0) + 1.5
        cx, cy = (plan.runs[0].arcs[0].center if R > 0 and plan.runs[0].arcs else pts[1])
        ax.set_xlim(cx - span, cx + span)
        ax.set_ylim(cy - span, cy + span)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")

    sel = study["selected_radius"]
    draw_case(fig.add_subplot(gs[0, 0]), sel, "Selected radius")
    draw_case(fig.add_subplot(gs[0, 1]), 0.0, "Turn on the spot")
    # choose a larger radius for comparison
    bigger = [r["R"] for r in rows if r["R"] > (sel or 0)]
    draw_case(fig.add_subplot(gs[0, 2]), bigger[len(bigger) // 2] if bigger else sel, "Larger radius")

    arc_rows = [r for r in rows if r["R"] > 0]
    piv = next((r for r in rows if r["R"] == 0), None)
    Rs = [r["R"] for r in arc_rows]
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(Rs, [r.get("env_outer") for r in arc_rows], "o-", label="outer edge (from centre)")
    ax.plot(Rs, [r.get("env_inner") for r in arc_rows], "o-", label="inner edge (from centre)")
    ax.plot(Rs, [r.get("band") for r in arc_rows], "s--", label="swept band width")
    ax.plot(Rs, [r.get("bulge") for r in arc_rows], "^:", label="bulge outside lane")
    if piv:
        ax.axhline(piv["env_outer"], color="k", ls=":", lw=1, label="pivot swept radius")
    ax.set_xlabel("drawn radius R (m)")
    ax.set_ylabel("m")
    ax.set_title("Floor space in the corner", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(Rs, [r.get("time_s") for r in arc_rows], "o-", label="corner manoeuvre time")
    if piv:
        ax.axhline(piv["time_s"], color="k", ls=":", lw=1, label=f"stop + pivot ({piv['time_s']:.1f} s)")
    ax.set_xlabel("drawn radius R (m)")
    ax.set_ylabel("s")
    ax2 = ax.twinx()
    ax2.plot(Rs, [r.get("curve_speed") for r in arc_rows], "g^--", label="curve speed used")
    ax2.plot(Rs, [r.get("v_max_at_R") for r in arc_rows], "gv:", label="max speed possible at R")
    ax2.set_ylabel("m/s", color="g")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7)
    ax.set_title("Time and speed", fontsize=10)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(Rs, [r.get("peak_torque_pct") for r in arc_rows], "o-", label="peak motor torque %")
    ax.plot(Rs, [r.get("grip_pct") for r in arc_rows], "s-", label="grip used %")
    if any(r.get("articulation_deg") is not None for r in arc_rows):
        ax.plot(Rs, [r.get("articulation_deg") for r in arc_rows], "^-", label="cart angle (deg)")
        ax.axhline(agv.cart.max_articulation_deg, color="r", ls=":", lw=1, label="cart angle limit")
    ax.axhline(100, color="r", ls="--", lw=1)
    if piv:
        ax.plot([0], [piv.get("peak_torque_pct")], "k*", ms=10, label="pivot: torque %")
        ax.plot([0], [piv.get("grip_pct")], "kX", ms=8, label="pivot: grip %")
    ax.set_xlabel("drawn radius R (m)")
    ax.set_title("Loads", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.suptitle(f"Turning study: {study['agv']}  |  {study['angle_deg']:.0f} deg corner  |  "
                 f"{'with cart' if study['with_cart'] else 'without cart'}  |  smallest OK radius: "
                 f"{study['min_feasible_radius'] if study['min_feasible_radius'] is not None else 'none'} m",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig
