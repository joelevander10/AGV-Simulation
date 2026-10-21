"""Drawing: AGV spec sheet, results dashboard and animated playback."""
from __future__ import annotations

import math
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Polygon, Rectangle

from .analysis import body_rect, cart_rect, lurking_body_rect, rect_corners
from .config import AGVConfig
from .factory_map import FactoryMap
from .simulate import DRIVE, PIVOT, SimResult


# ----------------------------------------------------------------------------- AGV shapes
def _rect(x0, x1, y0, y1):
    return rect_corners((x0, x1, y0, y1))


def agv_shapes(agv: AGVConfig, with_cart: bool = True) -> dict:
    """Polygons in the AGV body frame: name -> (Nx2 array, facecolor)."""
    sh = {"chassis": (rect_corners(body_rect(agv)), "#9ecae1")}
    r, w, b = agv.wheel_radius, agv.wheels.width, agv.wheels.track
    sh["wheel_L"] = (_rect(-r, r, b / 2 - w / 2, b / 2 + w / 2), "#252525")
    sh["wheel_R"] = (_rect(-r, r, -b / 2 - w / 2, -b / 2 + w / 2), "#252525")
    for i, c in enumerate(agv.wheels.castors):
        sh[f"castor_{i}"] = (_rect(c.x - c.diameter / 2, c.x + c.diameter / 2,
                                   c.y - c.width / 2, c.y + c.width / 2), "#636363")
    d = agv.drivetrain
    md, ml, gl = d.motor_diameter, d.motor_length, d.gearbox_length
    for side, sg in (("L", 1), ("R", -1)):
        yin = sg * (b / 2 - w / 2)
        yg = yin - sg * gl
        sh[f"gearbox_{side}"] = (_rect(-md / 2, md / 2, min(yin, yg), max(yin, yg)), "#fdae6b")
        if d.layout.lower() == "coaxial":
            ym = yg - sg * ml
            sh[f"motor_{side}"] = (_rect(-md / 2, md / 2, min(yg, ym), max(yg, ym)), "#e6550d")
        else:
            yc = (yin + yg) / 2
            sh[f"motor_{side}"] = (_rect(md / 2, md / 2 + ml, yc - md / 2, yc + md / 2), "#e6550d")
    if with_cart and agv.hook_type == "lurking":
        sh["cart"] = (rect_corners(lurking_body_rect(agv)), "none")
    return sh


def _tf(pts, x, y, th):
    c, s = math.cos(th), math.sin(th)
    return np.column_stack([x + c * pts[:, 0] - s * pts[:, 1], y + s * pts[:, 0] + c * pts[:, 1]])


# ----------------------------------------------------------------------------- spec sheet
def draw_agv_spec(agv: AGVConfig):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [1.6, 1]})
    fig.suptitle(f"AGV specification: {agv.name}", fontsize=13, fontweight="bold")

    # --- top view
    for name, (poly, fc) in agv_shapes(agv).items():
        ls = "--" if name == "cart" else "-"
        ax1.add_patch(Polygon(poly, closed=True, fc=fc, ec="k", lw=1, ls=ls, alpha=0.9 if fc != "none" else 1))
    if agv.hook_type == "tow":
        hk, c = agv.hook, agv.cart
        cr = cart_rect(agv)
        hx = hk.x
        for k, m in enumerate(agv.train_masses()):
            ax_x = hx - c.tongue_length
            poly = _tf(rect_corners(cr), ax_x, hk.y, 0.0)
            ax1.add_patch(Polygon(poly, closed=True, fc="#e5f5e0", ec="g", lw=1, ls="--"))
            ax1.plot([hx, ax_x], [hk.y, hk.y], "g-", lw=2)
            ax1.plot(ax_x, hk.y, "gs")
            ax1.text(ax_x, hk.y + c.width / 2 + 0.05, f"trolley {k + 1}\n{m:.0f} kg", color="g",
                     ha="center", fontsize=8)
            hx = ax_x - c.coupling_behind_axle
    if agv.hook_type in ("tow", "lurking"):
        ax1.plot(agv.hook.x, agv.hook.y, "o", ms=10, mfc="none", mec="g", mew=2)
        ax1.annotate(f"{agv.hook_type} hook\nx={agv.hook.x:+.3f}", (agv.hook.x, agv.hook.y), (8, -25),
                     textcoords="offset points", color="g", fontsize=8)
    for sn in agv.sensors:
        ax1.plot(sn.x, sn.y, "r^", ms=9)
        ax1.annotate(sn.name, (sn.x, sn.y), (5, 6), textcoords="offset points", color="r", fontsize=8)
    ax1.plot(0, 0, "k+", ms=14, mew=2)
    ax1.annotate("drive-axle centre (0,0)", (0, 0), (5, -14), textcoords="offset points", fontsize=8)
    ax1.annotate("", (agv.x_front + 0.15, 0), (agv.x_front - 0.05, 0), arrowprops=dict(arrowstyle="->", lw=2))
    ax1.text(agv.x_front + 0.17, 0, "FWD", va="center", fontsize=8)
    y_dim = -agv.chassis.width / 2 - 0.12
    ax1.annotate("", (agv.x_rear, y_dim), (agv.x_front, y_dim), arrowprops=dict(arrowstyle="<->"))
    ax1.text((agv.x_rear + agv.x_front) / 2, y_dim - 0.06, f"L = {agv.chassis.length:.3f} m",
             ha="center", fontsize=9)
    ax1.text(agv.x_rear - 0.05, 0, f"W = {agv.chassis.width:.3f} m", rotation=90, ha="right", va="center",
             fontsize=9)
    ax1.text(0.02, agv.wheels.track / 2 + 0.08, f"track {agv.wheels.track:.3f} m", fontsize=8)
    ax1.set_aspect("equal")
    ax1.autoscale()
    ax1.margins(0.08)
    ax1.set_title("Top view (body frame, m)")
    ax1.grid(alpha=0.3)

    # --- side view
    c = agv.chassis
    ax2.axhline(0, color="k", lw=1.5)
    ax2.add_patch(Rectangle((agv.x_rear, c.ground_clearance), c.length, c.height - c.ground_clearance,
                            fc="#9ecae1", ec="k"))
    ax2.add_patch(Circle((0, agv.wheel_radius), agv.wheel_radius, fc="#252525", ec="k"))
    for cs in agv.wheels.castors:
        ax2.add_patch(Circle((cs.x, cs.diameter / 2), cs.diameter / 2, fc="#636363", ec="k"))
    for sn in agv.sensors:
        ax2.plot(sn.x, sn.z, "r^", ms=8)
        ax2.annotate(f"{sn.name}\nz={sn.z * 1000:.0f} mm", (sn.x, sn.z), (4, 6), textcoords="offset points",
                     color="r", fontsize=7)
    if agv.hook_type == "tow":
        ax2.plot(agv.hook.x, agv.hook.z, "go", ms=8)
        ax2.annotate(f"hitch z={agv.hook.z * 1000:.0f} mm", (agv.hook.x, agv.hook.z), (-10, 10),
                     textcoords="offset points", color="g", fontsize=8, ha="right")
    if agv.hook_type == "lurking":
        cc = agv.cart
        ax2.add_patch(Rectangle((agv.hook.x - cc.length / 2, cc.underside_clearance), cc.length,
                                max(cc.height - cc.underside_clearance, 0.05), fc="#e5f5e0", ec="g", ls="--"))
    ax2.text(agv.x_front + 0.03, c.height, f"H = {c.height * 1000:.0f} mm", fontsize=9, va="center")
    ax2.set_aspect("equal")
    ax2.autoscale()
    ax2.margins(0.08)
    ax2.set_title("Side view (x vs height, m)")
    ax2.grid(alpha=0.3)

    d = agv.drivetrain
    info = (f"Motor {d.motor_rated_power_w:.0f} W, rated {d.rated_torque:.3f} N*m @ {d.motor_rated_rpm:.0f} rpm, "
            f"max {d.motor_max_rpm:.0f} rpm | gearbox 1:{d.gear_ratio:g} ({d.layout}) | "
            f"wheel D {agv.wheels.diameter * 1000:.0f} mm -> top speed {agv.max_wheel_speed:.3f} m/s | "
            f"tractive {2 * d.rated_torque * d.gear_ratio * d.gear_efficiency / agv.wheel_radius:.0f} N cont.")
    fig.text(0.5, 0.01, info, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return fig


# ----------------------------------------------------------------------------- report
def _draw_map(ax, fmap: FactoryMap):
    ax.imshow(fmap.rgb, extent=fmap.extent, origin="upper", interpolation="nearest")
    ax.set_xlim(0, fmap.width_m)
    ax.set_ylim(0, fmap.height_m)
    ax.set_aspect("equal")


def _draw_route(ax, res: SimResult, labels=True):
    for run in res.plan.runs:
        ax.plot(run.x, run.y, "--", color="#08519c" if run.direction > 0 else "#ae017e", lw=1.2)
    if labels:
        for i, wp in enumerate(res.route.waypoints):
            mk = {"stop": "s", "pivot": "D"}.get(wp.action, "o")
            ax.plot(wp.x, wp.y, mk, color="k", ms=5)
            ax.annotate(wp.name or f"WP{i}", (wp.x, wp.y), (4, 4), textcoords="offset points", fontsize=7)


def plot_report(res: SimResult, summary: dict, fmap: FactoryMap):
    L = res.loads
    has_lf = res.lf is not None
    nrow = 5 if has_lf else 4
    fig = plt.figure(figsize=(17, 9.5 if not has_lf else 11))
    gs = fig.add_gridspec(nrow, 3, width_ratios=[1.6, 1.6, 1.3])
    axm = fig.add_subplot(gs[:, :2])
    _draw_map(axm, fmap)
    mask = summary.get("_swept_mask")
    if mask is not None:
        ov = np.zeros(mask.shape + (4,))
        ov[mask] = (1.0, 0.55, 0.0, 0.35)
        axm.imshow(ov, extent=fmap.extent, origin="upper", interpolation="nearest")
    _draw_route(axm, res)
    axm.plot(res.x, res.y, color="#e31a1c", lw=1, label="drive-axle path")
    if res.cart_x is not None and res.agv.hook_type == "tow":
        trs = res.train or [{"x": res.cart_x, "y": res.cart_y}]
        axm.plot(trs[-1]["x"], trs[-1]["y"], color="#33a02c", lw=1,
                 label="cart axle path" if len(trs) == 1 else f"last trolley ({len(trs)}) axle path")
    if res.cart_x is not None:
        cpoly = rect_corners(cart_rect(res.agv))
        for k, pk in enumerate(res.parked):
            axm.add_patch(Polygon(_tf(cpoly, pk["x"], pk["y"], pk["th"]), closed=True, fc="none", ec="#2b8a3e",
                                  lw=1.2, ls="--", label="parked cart" if k == 0 else None))
    for c in summary.get("collisions", []):
        axm.plot(c["x"], c["y"], "rx", ms=12, mew=3)
    for c in summary.get("clearance_violations", []):
        axm.plot(c["x"], c["y"], "o", ms=10, mfc="none", mec="m", mew=2)
    lfi = summary.get("line_follow")
    if lfi:
        for e in lfi["events"]:
            if e.get("kind") in ("tape lost", "no tape at start", "timeout"):
                axm.plot(e["x"], e["y"], "X", color="#c92a2a", ms=16, mec="k", zorder=12)
                axm.annotate("TAPE LOST", (e["x"], e["y"]), (8, 8), textcoords="offset points", color="#c92a2a",
                             fontweight="bold", fontsize=9)
        if "worst" in lfi:
            w = lfi["worst"]
            axm.plot(w["x"], w["y"], "o", ms=14, mfc="none", mec="#f08c00", mew=2.5, zorder=11,
                     label=f"largest tape error {abs(w['pv']):.0f} mm")
    axm.plot([], [], "s", color=(1, 0.55, 0, 0.5), label="swept area")
    axm.plot([], [], "rx", label="collision")
    axm.plot([], [], "o", mfc="none", mec="m", label="margin violated")
    axm.legend(loc="upper right", fontsize=8)
    axm.set_title(f"{res.agv.name} on '{res.route.name}'  |  cycle {summary['cycle_time_s']:.1f} s  |  "
                  f"{summary['cycles_per_hour']:.1f} cycles/h")

    t = res.t
    ax1 = fig.add_subplot(gs[0, 2])
    ax1.plot(t, res.v, label="v axle")
    ax1.plot(t, L["wheel_speed_l"], lw=0.7, alpha=0.7, label="left wheel")
    ax1.plot(t, L["wheel_speed_r"], lw=0.7, alpha=0.7, label="right wheel")
    ax1.set_ylabel("m/s")
    ax1.legend(fontsize=7, ncol=3)
    ax1.set_title("Speed", fontsize=9)

    ax2 = fig.add_subplot(gs[1, 2], sharex=ax1)
    ax2.plot(t, L["motor_rpm_l"], label="L")
    ax2.plot(t, L["motor_rpm_r"], label="R")
    mx = res.agv.drivetrain.motor_max_rpm
    ax2.axhline(mx, color="r", ls=":", lw=1)
    ax2.axhline(-mx, color="r", ls=":", lw=1)
    ax2.set_ylabel("rpm")
    ax2.legend(fontsize=7)
    ax2.set_title("Motor speed", fontsize=9)

    ax3 = fig.add_subplot(gs[2, 2], sharex=ax1)
    ax3.plot(t, L["motor_torque_l"], label="L")
    ax3.plot(t, L["motor_torque_r"], label="R")
    d = res.agv.drivetrain
    for val, ls, lab in ((d.rated_torque, "--", "rated"), (d.peak_torque, ":", "peak")):
        ax3.axhline(val, color="r", ls=ls, lw=1, label=lab)
        ax3.axhline(-val, color="r", ls=ls, lw=1)
    ax3.set_ylabel("N*m")
    ax3.legend(fontsize=7, ncol=4)
    ax3.set_title("Motor torque", fontsize=9)

    ax4 = fig.add_subplot(gs[3, 2], sharex=ax1)
    lfi = summary.get("line_follow")
    if res.articulation_deg is not None and res.agv.hook_type == "tow" and np.isfinite(res.articulation_deg).any():
        ax4.plot(t, res.articulation_deg, color="g")
        ax4.set_ylabel("deg")
        ax4.set_title("Cart articulation angle", fontsize=9)
    else:
        ax4.plot(t, np.degrees(res.w), color="purple")
        ax4.set_ylabel("deg/s")
        ax4.set_title("Yaw rate", fontsize=9)
    axes = [ax1, ax2, ax3, ax4]
    if has_lf:
        ax5 = fig.add_subplot(gs[4, 2], sharex=ax1)
        rng = res.agv.line_follow.sensor_range_mm
        ax5.plot(t, res.lf["pv"], color="#1f6feb", lw=0.9)
        ax5.axhline(rng, color="r", ls="--", lw=1)
        ax5.axhline(-rng, color="r", ls="--", lw=1)
        for e in (lfi or {}).get("events", []):
            if e.get("kind") == "tape lost":
                ax5.axvline(e["t"], color="#c92a2a", lw=2)
        ax5.set_ylabel("mm")
        ax5.set_title("Tape position under the MGS1600 (red = sensor edge / track lost)", fontsize=9)
        axes.append(ax5)
        ax4.tick_params(labelbottom=False)
    axes[-1].set_xlabel("time (s)")
    for ax in axes:
        ax.grid(alpha=0.3)
        for e in res.events:
            if e.kind != "drive":
                ax.axvspan(e.t0, e.t1, color="grey" if e.kind == "dwell" else "orange", alpha=0.12, lw=0)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- animation
def animate(res: SimResult, fmap: FactoryMap, speed: float = 4.0, save: Optional[str] = None,
            follow: bool = False, window: float = 8.0):
    agv = res.agv
    fig, ax = plt.subplots(figsize=(13, 8))
    _draw_map(ax, fmap)
    _draw_route(ax, res, labels=True)
    shapes = agv_shapes(agv, with_cart=False)
    patches = {}
    for name, (poly, fc) in shapes.items():
        p = Polygon(poly, closed=True, fc=fc, ec="k", lw=0.8, zorder=5,
                    ls="--" if name == "cart" else "-", alpha=0.95 if fc != "none" else 1)
        ax.add_patch(p)
        patches[name] = (p, poly)
    cart_patch = tongue = None
    has_cart = agv.hook_type in ("tow", "lurking") and res.cart_x is not None
    cpoly = rect_corners(cart_rect(agv)) if has_cart else None
    parked_patches = []
    trains = (res.train or [{"x": res.cart_x, "y": res.cart_y, "th": res.cart_th}]) if has_cart else []
    extra_patches = []
    if has_cart:
        cart_patch = Polygon(cpoly, closed=True, fc="#c7e9c0", ec="g", lw=1,
                             zorder=4 if agv.hook_type == "tow" else 6, alpha=0.9 if agv.hook_type == "tow" else 0.45)
        ax.add_patch(cart_patch)
        (tongue,) = ax.plot([], [], "g-", lw=2, zorder=6)
        for tr in trains[1:]:
            pp = Polygon(cpoly, closed=True, fc="#c7e9c0", ec="g", lw=1, zorder=4)
            ax.add_patch(pp)
            (ln,) = ax.plot([], [], "g-", lw=2, zorder=6)
            extra_patches.append((pp, ln, tr))
        for pk in res.parked:
            pp = Polygon(_tf(cpoly, pk["x"], pk["y"], pk["th"]), closed=True, fc="#e5f5e0", ec="g", lw=1,
                         ls="--", zorder=3, visible=False)
            ax.add_patch(pp)
            parked_patches.append((pp, pk))
    sens_pts = np.array([[s.x, s.y] for s in agv.sensors]) if agv.sensors else np.zeros((0, 2))
    (sens,) = ax.plot([], [], "r^", ms=6, zorder=7)
    (trail,) = ax.plot([], [], color="#e31a1c", lw=1, zorder=3)
    (ctrail,) = ax.plot([], [], color="#33a02c", lw=1, zorder=3)
    info = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top", ha="left", fontsize=9, family="monospace",
                   bbox=dict(fc="white", alpha=0.85), zorder=10)
    L = res.loads
    dt = agv.simulation.dt
    stride = max(1, int(round(speed * 0.04 / dt)))
    frames = list(range(0, len(res.t), stride))
    if frames[-1] != len(res.t) - 1:
        frames.append(len(res.t) - 1)
    mode_name = {0: "DWELL", 1: "DRIVE", 2: "PIVOT"}

    def update(i):
        x, y, th = res.x[i], res.y[i], res.th[i]
        for name, (p, poly) in patches.items():
            p.set_xy(_tf(poly, x, y, th))
        arts = [p for p, _ in patches.values()]
        if cart_patch is not None:
            on = bool(np.isfinite(res.cart_x[i]))
            cart_patch.set_visible(on)
            tongue.set_visible(on and agv.hook_type == "tow")
            if on:
                cart_patch.set_xy(_tf(cpoly, res.cart_x[i], res.cart_y[i], res.cart_th[i]))
                tongue.set_data([res.hitch_x[i], res.cart_x[i]], [res.hitch_y[i], res.cart_y[i]])
            if agv.hook_type == "tow":
                ctrail.set_data(trains[-1]["x"][:i + 1], trains[-1]["y"][:i + 1])
            prev = trains[0] if trains else None
            cpb = agv.cart.coupling_behind_axle
            for pp, ln, tr in extra_patches:
                on_k = bool(np.isfinite(tr["x"][i]))
                pp.set_visible(on_k)
                ln.set_visible(on_k)
                if on_k:
                    pp.set_xy(_tf(cpoly, tr["x"][i], tr["y"][i], tr["th"][i]))
                    kx = prev["x"][i] - cpb * math.cos(prev["th"][i])
                    ky = prev["y"][i] - cpb * math.sin(prev["th"][i])
                    ln.set_data([kx, tr["x"][i]], [ky, tr["y"][i]])
                prev = tr
            arts += [p_ for p_, _, _ in extra_patches] + [l_ for _, l_, _ in extra_patches]
            for pp, pk in parked_patches:
                pp.set_visible(pk["t0"] <= res.t[i] <= pk["t1"])
            arts += [cart_patch, tongue, ctrail] + [pp for pp, _ in parked_patches]
        if len(sens_pts):
            sp = _tf(sens_pts, x, y, th)
            sens.set_data(sp[:, 0], sp[:, 1])
        trail.set_data(res.x[:i + 1], res.y[:i + 1])
        txt = (f"t = {res.t[i]:7.1f} s   {mode_name[int(res.mode[i])]}\n"
               f"v = {res.v[i]:+6.3f} m/s   w = {math.degrees(res.w[i]):+6.1f} deg/s\n"
               f"motor L/R = {L['motor_rpm_l'][i]:+6.0f} / {L['motor_rpm_r'][i]:+6.0f} rpm\n"
               f"torque L/R = {L['motor_torque_l'][i]:+5.2f} / {L['motor_torque_r'][i]:+5.2f} N*m")
        if res.articulation_deg is not None and agv.hook_type == "tow" and np.isfinite(res.articulation_deg[i]):
            txt += f"\ncart angle = {res.articulation_deg[i]:+5.1f} deg"
        info.set_text(txt)
        if follow:
            ax.set_xlim(x - window, x + window)
            ax.set_ylim(y - window * 0.6, y + window * 0.6)
        return arts + [sens, trail, info]

    ax.set_title(f"{agv.name}  -  playback x{speed:g}")
    anim = FuncAnimation(fig, update, frames=frames, interval=40, blit=not follow, repeat=True)
    if save:
        fps = 25
        writer = PillowWriter(fps=fps) if save.lower().endswith(".gif") else FFMpegWriter(fps=fps)
        anim.save(save, writer=writer, dpi=90)
    return fig, anim
