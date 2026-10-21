"""Time-domain simulation of one route cycle.

Kinematics
  * differential drive (unicycle) body, pose = drive-axle centre + heading
  * 'axle' tracking: axle follows the drawn path exactly
  * sensor tracking (tape): the sensor follows the path, the axle is dragged behind it
    (tractrix) -> reproduces the inside cut of a real tape-following AGV
  * the speed profile (accel / decel / jerk / speed limits) is applied to the DRIVE AXLE,
    i.e. to the speed the motor controller commands, not to the sensor point
  * towed cart: single-axle trailer on an (off-axle) hitch, also a tractrix
  * lurking cart: rigidly carried at the hook point
  * carts can be attached / detached at stations (waypoint `cart: attach | detach`)

Loads are computed in loads.py (per drive wheel, incl. castors, drawbar, slopes).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .config import AGVConfig, RouteConfig
from .path import RoutePlan, Run, wrap
from .planner import motion_profile, pivot_profile, speed_limits, time_grid

DWELL, DRIVE, PIVOT = 0, 1, 2
MODE_NAMES = {DWELL: "dwell", DRIVE: "drive", PIVOT: "pivot"}
PIVOT_TOL_TAPE_DEG = 20.0   # with sensor tracking, smaller heading errors are corrected while driving


@dataclass
class Event:
    kind: str
    label: str
    t0: float = 0.0
    t1: float = 0.0
    distance: float = 0.0
    angle_deg: float = 0.0
    run_index: int = -1
    mode: str = ""

    @property
    def duration(self) -> float:
        return self.t1 - self.t0


@dataclass
class SimResult:
    agv: AGVConfig
    route: RouteConfig
    plan: RoutePlan
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    th: np.ndarray
    v: np.ndarray
    w: np.ndarray
    a: np.ndarray
    ref_x: np.ndarray
    ref_y: np.ndarray
    mode: np.ndarray
    run_id: np.ndarray
    ref_s: np.ndarray
    events: List[Event]
    warnings: List[str]
    grade: Optional[np.ndarray] = None       # % slope in the AGV's forward direction
    attached: Optional[np.ndarray] = None    # bool, a cart is on the hook
    cart_mass: Optional[np.ndarray] = None   # kg of the attached cart (0 when none)
    hitch_x: Optional[np.ndarray] = None
    hitch_y: Optional[np.ndarray] = None
    cart_x: Optional[np.ndarray] = None     # tow: cart axle; lurking: cart centre (NaN when detached)
    cart_y: Optional[np.ndarray] = None
    cart_th: Optional[np.ndarray] = None
    articulation_deg: Optional[np.ndarray] = None   # AGV <-> first trolley
    train: List[dict] = field(default_factory=list)  # every trolley: x, y, th (axle), art (deg, joint in front), m (kg share)
    parked: List[dict] = field(default_factory=list)   # carts standing alone: x, y, th, t0, t1
    loads: dict = field(default_factory=dict)
    lf: Optional[dict] = None           # closed-loop tape following: pv, lost, cl, cr, tv, sat + events
    lf_stats: Optional[dict] = None     # PID auto-tune statistics per gain set (collect_stats=True)


# -----------------------------------------------------------------------------
LF_KEYS = ("pv", "lost", "cl", "cr", "tv", "sat", "sm", "vp", "wp", "ap")   # closed-loop channels
_LF_NAN = ("pv", "vp", "wp", "ap")      # channels that are NaN where the closed loop is not driving


class _Timeline:
    KEYS = ("t", "x", "y", "th", "rx", "ry", "mode", "run", "s", "g", "att", "cm") + LF_KEYS

    def __init__(self, dt):
        self.dt = dt
        self.data = {k: [] for k in self.KEYS}
        self.events: List[Event] = []
        self.t_end = 0.0
        self.att = False
        self.cm = 0.0
        self.cart_events = []    # (time, 'attach'|'detach')

    def start(self, x, y, th):
        for k, v in zip(self.KEYS, (0.0, x, y, th, x, y, DWELL, -1, 0.0, 0.0, float(self.att), self.cm,
                                    np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, np.nan, np.nan, np.nan)):
            self.data[k].append(np.array([v], dtype=float))

    @property
    def pose(self):
        return (float(self.data["x"][-1][-1]), float(self.data["y"][-1][-1]),
                float(self.data["th"][-1][-1]))

    def add(self, t_local, x, y, th, rx, ry, mode, run_id, s, event: Event, grade=0.0, lf=None):
        t = np.asarray(t_local, float) + self.t_end
        n = len(t)
        event.t0 = self.t_end
        if n >= 2:
            lf = lf or {}
            extra = tuple(np.asarray(lf[k], float) if k in lf else np.full(n, np.nan if k in _LF_NAN else (-1.0 if k == "sm" else 0.0))
                          for k in LF_KEYS)
            cols = (t, x, y, th, rx, ry, np.full(n, mode), np.full(n, run_id), s,
                    np.broadcast_to(np.asarray(grade, float), (n,)), np.full(n, float(self.att)),
                    np.full(n, self.cm)) + extra
            for k, v in zip(self.KEYS, cols):
                self.data[k].append(np.asarray(v, float)[1:])
            self.t_end = float(t[-1])
        event.t1 = self.t_end
        self.events.append(event)

    def dwell(self, T, label, kind="dwell"):
        if T <= 0:
            return
        tg = time_grid(T, self.dt)
        x, y, th = self.pose
        n = len(tg)
        g_last = float(self.data["g"][-1][-1])        # standing on the slope it stopped on
        self.add(tg, np.full(n, x), np.full(n, y), np.full(n, th), np.full(n, x), np.full(n, y),
                 DWELL, -1, np.zeros(n), Event(kind, label), grade=g_last)

    def pivot(self, dth, perf, label):
        tg, ang = pivot_profile(dth, math.radians(perf.pivot_speed_dps),
                                math.radians(perf.pivot_accel_dps2), self.dt,
                                math.radians(perf.pivot_jerk_dps3))
        x, y, th0 = self.pose
        n = len(tg)
        th = th0 + math.copysign(1.0, dth) * ang
        self.add(tg, np.full(n, x), np.full(n, y), th, np.full(n, x), np.full(n, y),
                 PIVOT, -1, np.zeros(n), Event("pivot", label, angle_deg=math.degrees(dth)))

    def cart_event(self, ev, mass, label, duration):
        """Couple / uncouple a cart at the current spot. The state changes at the START of the
        handling time, so the whole handling time is spent with the new cart state."""
        if not ev:
            return
        dropped = self.cm
        self.att = ev == "attach"
        self.cm = float(mass) if self.att else 0.0
        self.cart_events.append((self.t_end, ev))
        txt = (f"Attach cart ({self.cm:.0f} kg) @ {label}" if self.att else
               f"Detach cart ({dropped:.0f} kg) @ {label}")
        if duration > 0:
            self.dwell(duration, txt, kind="dwell")
        else:
            # zero-time event: still record it so the timeline shows it
            self.events.append(Event("dwell", txt, t0=self.t_end, t1=self.t_end))

    def arrays(self):
        return {k: np.concatenate(v) for k, v in self.data.items()}


def _nearest_start(run: Run, tx, ty, reach):
    idx = np.nonzero(run.s <= reach)[0]
    dd = np.hypot(run.x[idx] - tx, run.y[idx] - ty)
    k = int(np.argmin(dd))
    return int(idx[k]), float(dd[k])


def _follow(run: Run, i0, axle, th0, d, direction, offset=(0.0, 0.0), blend=0.6):
    """Axle pose along the run's reference samples (geometry only, speed independent).

    offset = actual tracking point - path point at start. It is faded out over `blend` metres,
    like the line-following controller pulling a small initial error back onto the tape.
    """
    fade = np.clip(1.0 - (run.s[i0:] - run.s[i0]) / blend, 0.0, 1.0)
    px, py = run.x[i0:] + offset[0] * fade, run.y[i0:] + offset[1] * fade
    n = len(px)
    flip = math.pi if direction < 0 else 0.0
    if d <= 0:
        ax, ay = px.copy(), py.copy()
        th = run.heading[i0:] + flip
    else:
        ax, ay, th = np.empty(n), np.empty(n), np.empty(n)
        axx, axy = axle
        ex, ey = px[0] - axx, py[0] - axy
        nn = math.hypot(ex, ey)
        if nn < 1e-9:
            ex, ey = math.cos(th0 - flip), math.sin(th0 - flip)
        else:
            ex, ey = ex / nn, ey / nn
        axx, axy = px[0] - d * ex, py[0] - d * ey
        prevx, prevy = px[0], py[0]
        for i in range(n):
            X, Y = px[i], py[i]
            proj = (X - prevx) * ex + (Y - prevy) * ey
            axx += proj * ex
            axy += proj * ey
            rx, ry = X - axx, Y - axy
            rn = math.hypot(rx, ry)
            ex, ey = rx / rn, ry / rn
            axx, axy = X - d * ex, Y - d * ey
            ax[i], ay[i] = axx, axy
            th[i] = math.atan2(ey, ex) + flip
            prevx, prevy = X, Y
    th = np.unwrap(th)
    th += 2 * math.pi * round((th0 - th[0]) / (2 * math.pi))
    return ax, ay, th


def _tow_trailer(hx, hy, th0, L, artic0=0.0):
    """Single-axle cart pulled/pushed by the hitch point (hx, hy). Returns axle x, y, heading."""
    n = len(hx)
    psi0 = th0 - artic0
    Ax, Ay = hx[0] - L * math.cos(psi0), hy[0] - L * math.sin(psi0)
    ox, oy, psi = np.empty(n), np.empty(n), np.empty(n)
    for i in range(n):
        if i > 0:
            dX, dY = hx[i] - hx[i - 1], hy[i] - hy[i - 1]
            nsub = max(1, math.ceil(math.hypot(dX, dY) / 0.005))
            for k in range(1, nsub + 1):
                Hpx = hx[i - 1] + dX * (k - 1) / nsub
                Hpy = hy[i - 1] + dY * (k - 1) / nsub
                Hx = hx[i - 1] + dX * k / nsub
                Hy = hy[i - 1] + dY * k / nsub
                ex, ey = Hpx - Ax, Hpy - Ay
                nn = math.hypot(ex, ey)
                ex, ey = ex / nn, ey / nn
                proj = (Hx - Hpx) * ex + (Hy - Hpy) * ey
                Ax += proj * ex
                Ay += proj * ey
                rx, ry = Hx - Ax, Hy - Ay
                rn = math.hypot(rx, ry)
                Ax, Ay = Hx - L * rx / rn, Hy - L * ry / rn
        ox[i], oy[i] = Ax, Ay
        psi[i] = math.atan2(hy[i] - Ay, hx[i] - Ax)
    return ox, oy, np.unwrap(psi)


def _axle_arclength(ax, ay):
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(ax), np.diff(ay)))])
    # strictly increasing for interpolation (the axle never moves backwards along a run)
    return np.maximum.accumulate(s) + 1e-9 * np.arange(len(s))


def _initial_cart_state(agv: AGVConfig, plan: RoutePlan, warnings: List[str]) -> bool:
    evs = []
    for e in [plan.start_cart] + [r.end_cart for r in plan.runs]:
        evs += ["detach", "attach"] if e == "swap" else ([e] if e else [])
    if agv.hook_type == "none":
        if evs:
            warnings.append("Route has cart attach/detach points but the vehicle has no hook - ignored.")
        return False
    if not evs:
        return True
    att = evs[0] != "attach"
    state = att
    for e in evs:
        want = e == "attach"
        if want == state:
            warnings.append(f"Cart '{e}' while the cart is already {'on' if state else 'off'} the hook "
                            f"- check the attach/detach order along the route.")
        state = want
    if plan.loop and state != att:
        warnings.append("The loop ends with a different cart state than it starts with "
                        "(unequal number of attach and detach points).")
    return att


# ----------------------------------------------------------------------------- load
def _load_now(agv, tl):
    """(kg towed, kg carried on a lurking hook) right now."""
    if not tl.att:
        return 0.0, 0.0
    if agv.hook_type == "tow":
        return float(tl.cm), 0.0
    if agv.hook_type == "lurking":
        return 0.0, float(tl.cm)
    return 0.0, 0.0


def _load_ramps(agv, tl, run, i0, direc, notes):
    """Accel / decel along the run, limited by what the drive can do with the current load."""
    from .drive import DriveLimits
    drv = DriveLimits(agv)
    m_tow, m_lurk = _load_now(agv, tl)
    acc = np.array(run.acc[i0:], float)
    dec = np.array(run.dec[i0:], float)
    g = np.asarray(run.grade[i0:], float) * direc
    worst = None
    for gv in np.unique(np.round(g, 2)):
        st = drv.state(m_tow, m_lurk, float(gv))
        sel = np.round(g, 2) == gv
        a_max = max(st["a_max"], 0.005)
        d_max = max(st["d_max"], 0.005)
        if np.any(acc[sel] > a_max + 1e-9) or np.any(dec[sel] > d_max + 1e-9):
            if worst is None or st["a_max"] < worst["a_max"]:
                worst = st
        acc[sel] = np.minimum(acc[sel], a_max)
        dec[sel] = np.minimum(dec[sel], d_max)
    if worst is not None:
        _note_load(notes, run, worst, 1.0, False, agv, asked=float(np.max(run.acc[i0:])))
    return acc, dec


def _note_load(notes, run, st, limited_s, runaway, agv, asked=None):
    if not st:
        return
    key = (round(st["m_tow"] + st["M"]), round(st.get("grade", 0.0), 1))
    what = (f"{st['m_tow']:.0f} kg of trolleys" if st["m_tow"] > 0 else
            (f"{st['M'] - agv.chassis.mass_kg:.0f} kg on the hook" if st["M"] > agv.chassis.mass_kg + 1e-6
             else "no load"))
    slope = f" on {st['grade']:+.1f} % slope" if abs(st.get("grade", 0.0)) > 1e-6 else ""
    if st["a_max"] <= 0:
        txt = (f"CANNOT MOVE{slope} with {what}: rolling the train needs {st['cruise_force']:.0f} N, the drive gives "
               f"only {st['F_drive']:.0f} N ({st['limit']}) - the wheels slip / motors stall.")
        if st["limit"] == "grip":
            mu, f = agv.wheels.friction_coeff, agv.chassis.drive_load_fraction
            need_n = 1.2 * st["cruise_force"] / max(mu, 1e-6)              # +20 % to be able to speed up
            kg_drive = need_n / 9.81
            crr_ok = max(0.0, (st["F_drive"] / 1.2 - agv.simulation.rolling_resistance * st["m_agv"] * 9.81)
                         / max(st["m_tow"] * 9.81, 1e-6))
            txt += (f" With these numbers the real AGV could not pull this either, so check them: the drive wheels "
                    f"carry {f * st['m_agv']:.0f} kg (chassis.mass_kg {st['m_agv']:.0f} x drive_load_fraction {f:.2f}) - "
                    f"they need about {kg_drive:.0f} kg at friction_coeff {mu:.2f}; or the trolleys roll easier "
                    f"(cart.rolling_resistance {agv.cart.rolling_resistance:.3f} -> it would move at <= {crr_ok:.3f}); "
                    f"or fewer / lighter trolleys. Measure the real pull force with a spring scale if you can.")
        notes[key + ("climb",)] = txt
        return
    if runaway:
        notes[key + ("run",)] = (f"With {what}{slope} the brake cannot hold the train - it pushes the AGV on.")
    if limited_s > 0.2 and key + ("acc",) not in notes:
        asked_txt = f" (asked {asked:.2f})" if asked else ""
        notes[key + ("acc",)] = (f"Load: with {what}{slope} the AGV speeds up at most {st['a_max']:.2f} m/s2"
                                 f"{asked_txt} and brakes at {st['brake']:.2f} m/s2 - limited by "
                                 f"{st['limit']} ({st['F_drive']:.0f} N drive force for {st['M']:.0f} kg, "
                                 f"{st['cruise_force']:.0f} N to keep rolling). More weight on the drive "
                                 f"wheels or a higher friction_coeff helps.")
    if st["cruise_force"] > st["rated_force"] * 1.0 + 1e-6:
        notes[key + ("rated",)] = (f"Load: with {what}{slope} cruising needs {st['cruise_force']:.0f} N - more "
                                   f"than the motors' rated {st['rated_force']:.0f} N (they overheat in the long run).")


# -----------------------------------------------------------------------------
def simulate(agv: AGVConfig, route: RouteConfig, plan: RoutePlan,
             cart_articulation0_deg=0.0, collect_stats=False, light=False) -> SimResult:
    """cart_articulation0_deg: towed-cart angle at t=0 (for steady-state looping, see run_simulation)."""
    from .loads import compute_loads

    dt = agv.simulation.dt
    perf = agv.performance
    warnings = list(plan.warnings)
    hk = agv.hook
    tl = _Timeline(dt)
    for dname in (agv.tracking.forward, agv.tracking.reverse):
        if dname and dname.lower() != "axle":
            try:
                sn = agv.sensor(dname)
            except Exception:  # noqa: BLE001 - reported where the sensor is used
                continue
            if abs(sn.y) > 0.02:
                warnings.append(f"Tracking sensor '{dname}' sits {sn.y * 1000:.0f} mm off the centre line; that "
                                f"sideways offset is ignored for path following.")
    tl.att = _initial_cart_state(agv, plan, warnings)
    tl.cm = agv.train_mass(getattr(route, "start_load", "full")) if tl.att else 0.0
    use_events = agv.hook_type != "none"
    tl.start(plan.start_axle[0], plan.start_axle[1], plan.start_heading)

    def do_cart(ev, mass, where):
        if not use_events or not ev:
            return
        m = agv.train_mass("full") if mass is None else float(mass)
        if ev == "swap":                                  # drop the trolleys, hitch the other ones
            tl.cart_event("detach", 0.0, where, hk.detach_time)
            tl.cart_event("attach", m, where, hk.attach_time)
            return
        tl.cart_event(ev, m, where, hk.attach_time if ev == "attach" else hk.detach_time)

    do_cart(plan.start_cart, plan.start_cart_mass, plan.start_name)
    if plan.start_dwell > 0:
        tl.dwell(plan.start_dwell, f"Dwell @ {plan.start_name}")

    lf_on = bool(agv.line_follow.enabled)
    session = None
    lf_info = {"events": [], "runs": [], "aborted": False}
    if lf_on:
        if agv.tracking_offset(1) <= 0:
            warnings.append("Line following (PID) needs a tape sensor as the tracking point - "
                            "set Path following to the MGS1600 sensor. Using the kinematic model.")
            lf_on = False
        else:
            from .linefollow import LineFollowSession
            session = LineFollowSession(agv, route, plan)
            if collect_stats:
                session.stats = {}

    last_stop = plan.start_name
    load_notes = {}
    # after a direction change the new tracking sensor can be (front + rear offset) along the new leg
    reach = agv.tracking_offset(1) + agv.tracking_offset(-1) + 0.5
    for run in plan.runs:
        d, direc = run.track_offset, run.direction
        i0, off = 0, 0.0
        for attempt in range(2):
            x, y, th = tl.pose
            tx = x + d * direc * math.cos(th)
            ty = y + d * direc * math.sin(th)
            i0, off = _nearest_start(run, tx, ty, reach)
            target = run.heading[i0] + (math.pi if direc < 0 else 0.0)
            dth = float(wrap(target - th))
            # axle tracking must be aligned; a tape sensor re-centres small errors by itself
            tol = math.radians(1.0) if d <= 0 else math.radians(PIVOT_TOL_TAPE_DEG)
            if attempt == 0 and abs(dth) > tol:
                tl.pivot(dth, perf, f"Pivot @ {last_stop}")
                continue
            break
        if off > 0.10:
            warnings.append(f"Run '{run.label}': tracking point starts {off:.2f} m off the path "
                            f"(check stop/pivot layout).")
        if i0 >= len(run.s) - 2:
            warnings.append(f"Run '{run.label}' skipped: too short for the tracking offset.")
            continue

        x, y, th = tl.pose
        if lf_on and d > 0:
            nxt = plan.runs[run.index + 1] if run.index + 1 < len(plan.runs) else (
                plan.runs[0] if plan.loop else None)
            m_tow, m_lurk = _load_now(agv, tl)
            r = session.drive_run(run, nxt, (x, y, th), dt, m_tow=m_tow, m_lurk=m_lurk)
            A = r["arrays"]
            tl.add(A["t"], A["x"], A["y"], A["th"], A["rx"], A["ry"], DRIVE, run.index, A["s"],
                   Event("drive", run.label, distance=float(run.s[-1]), run_index=run.index,
                         mode=", ".join(session.mode_names[int(c)] for c in dict.fromkeys(A["sm"].round())
                                        if c >= 0)),
                   grade=np.interp(A["s"], run.s, run.grade) * direc,
                   lf={k: A[k] for k in LF_KEYS})
            t_base = tl.t_end - float(A["t"][-1])
            for ev in r["lost_events"]:
                lf_info["events"].append(dict(ev, t=ev["t"] + t_base, run=run.label))
            lf_info["runs"].append({"label": run.label, "stop_overshoot": r["stop_overshoot"],
                                    "aborted": r["aborted"], "load_limited_s": r["limited_s"],
                                    "load": r.get("load")})
            if agv.simulation.load_effects:
                _note_load(load_notes, run, r.get("load"), r["limited_s"], r.get("runaway"), agv)
            if r["aborted"]:
                lf_info["aborted"] = True
                kinds = [e["kind"] for e in r["lost_events"]]
                if "stalled" in kinds:
                    warnings.append(f"AGV CANNOT MOVE on '{run.label}' at t={tl.t_end:.1f} s: with 'Trolley load calc' "
                                    f"on, the drive force is smaller than what the load needs (see the CANNOT MOVE "
                                    f"note) - cycle NOT completed.")
                else:
                    warnings.append(f"TRACK LOST on '{run.label}' ({', '.join(kinds)}) at t={tl.t_end:.1f} s: "
                                    f"the AGV brakes and waits for an operator - cycle NOT completed.")
                break
            last_stop = run.end_name
            do_cart(run.end_cart, run.end_cart_mass, run.end_name)
            dwell = run.end_dwell + perf.stop_settle_time
            if dwell > 0:
                tl.dwell(dwell, f"Dwell @ {run.end_name}")
            continue
        tx = x + d * direc * math.cos(th)
        ty = y + d * direc * math.sin(th)
        ofs = (tx - run.x[i0], ty - run.y[i0])
        ax, ay, ath = _follow(run, i0, (x, y), th, d, direc, ofs)
        s_ref = run.s[i0:] - run.s[i0]
        # --- speed profile on the drive axle ---------------------------------------
        if d > 0:
            s_ax = _axle_arclength(ax, ay)
            k_ax = np.gradient(ath, s_ax)
            k_ax = np.clip(k_ax, -20.0, 20.0)
        else:
            s_ax = s_ref + 1e-9 * np.arange(len(s_ref))
            k_ax = run.kappa[i0:]
        lim = speed_limits(run.kappa[i0:], k_ax, run.vmax[i0:], run.vcurve[i0:], run.vcap[i0:], agv)
        if agv.simulation.load_effects:
            acc, dec = _load_ramps(agv, tl, run, i0, direc, load_notes)
        else:
            acc, dec = run.acc[i0:], run.dec[i0:]
        tg, sg = motion_profile(s_ax, lim, acc, dec, perf.jerk, dt)
        ip = lambda arr: np.interp(sg, s_ax, arr)   # noqa: E731
        tl.add(tg, ip(ax), ip(ay), ip(ath), ip(run.x[i0:]), ip(run.y[i0:]),
               DRIVE, run.index, ip(run.s[i0:]),
               Event("drive", run.label, distance=float(s_ref[-1]), run_index=run.index, mode=run.modes),
               grade=ip(run.grade[i0:]) * direc)
        last_stop = run.end_name
        do_cart(run.end_cart, run.end_cart_mass, run.end_name)
        dwell = run.end_dwell + perf.stop_settle_time
        if dwell > 0:
            tl.dwell(dwell, f"Dwell @ {run.end_name}")

    if plan.loop and not lf_info["aborted"]:
        x, y, th = tl.pose
        dth = float(wrap(plan.start_heading - th))
        tol = math.radians(1.0) if plan.runs[0].track_offset <= 0 else math.radians(PIVOT_TOL_TAPE_DEG)
        if abs(dth) > tol:
            tl.pivot(dth, perf, f"Pivot @ {plan.start_name} (re-align for next cycle)")

    for txt in load_notes.values():
        warnings.append(txt)
    D = tl.arrays()
    t = D["t"]
    th = np.unwrap(D["th"])
    x, y = D["x"], D["y"]
    vx, vy = np.gradient(x, t), np.gradient(y, t)
    v = vx * np.cos(th) + vy * np.sin(th)
    w = np.gradient(th, t)
    a = np.gradient(v, t)
    # closed-loop stretches: take speed, yaw rate and acceleration straight from the drive plant
    # (differentiating the recorded positions twice amplifies the sensor-noise jitter into the loads)
    cl = np.isfinite(D["vp"])
    if cl.any():
        v = np.where(cl, D["vp"], v)
        w = np.where(cl, D["wp"], w)
        a = np.where(cl & np.isfinite(D["ap"]), D["ap"], a)
    att = D["att"] > 0.5

    res = SimResult(agv=agv, route=route, plan=plan, t=t, x=x, y=y, th=th, v=v, w=w, a=a,
                    ref_x=D["rx"], ref_y=D["ry"], mode=D["mode"].astype(int),
                    run_id=D["run"].astype(int), ref_s=D["s"], events=tl.events, warnings=warnings,
                    grade=D["g"], attached=att, cart_mass=np.where(att, D["cm"], 0.0))
    if session is not None:
        lf_info["events"] += session.events
        lf_info["mode_names"] = list(session.mode_names)
        lf_info["mode_speeds"] = {"HIGH": agv.line_follow.speed_high, "SLOW": agv.line_follow.speed_slow,
                                  "APPROACH": agv.line_follow.speed_approach,
                                  "REVERSE": agv.line_follow.speed_reverse,
                                  **{k: v[0] for k, v in session.extra_modes.items()}}
        res.lf = dict({k: D[k] for k in LF_KEYS}, **lf_info)
        res.lf_stats = session.stats
    if light:                              # PID tuning: no trolley kinematics / loads needed
        return res

    # ---- hook / cart -------------------------------------------------------
    ht = agv.hook_type
    if ht in ("tow", "lurking"):
        c, s = np.cos(th), np.sin(th)
        res.hitch_x = x + c * hk.x - s * hk.y
        res.hitch_y = y + s * hk.x + c * hk.y
        n = len(t)
        masses = agv.train_masses()
        ntr = len(masses) if ht == "tow" else 1
        a0s = list(cart_articulation0_deg) if isinstance(cart_articulation0_deg, (list, tuple)) \
            else [cart_articulation0_deg]
        a0s += [0.0] * (ntr - len(a0s))
        train = [{"x": np.full(n, np.nan), "y": np.full(n, np.nan), "th": np.full(n, np.nan),
                  "m": masses[k] / max(sum(masses), 1e-9)} for k in range(ntr)]
        # contiguous attached intervals
        edges = np.diff(np.concatenate([[0], att.astype(int), [0]]))
        starts, ends = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]
        cp = agv.cart.coupling_behind_axle
        for i0, i1 in zip(starts, ends):
            if ht == "tow":
                hx, hy, hth = res.hitch_x[i0:i1], res.hitch_y[i0:i1], float(th[i0])
                for k in range(ntr):
                    a0 = math.radians(a0s[k]) if i0 == 0 else 0.0
                    ox, oy, psi = _tow_trailer(hx, hy, hth, agv.cart.tongue_length, a0)
                    tr = train[k]
                    tr["x"][i0:i1], tr["y"][i0:i1], tr["th"][i0:i1] = ox, oy, psi
                    # the next trolley hooks on behind this one's axle
                    hx, hy, hth = ox - cp * np.cos(psi), oy - cp * np.sin(psi), float(psi[0])
            else:
                tr = train[0]
                tr["x"][i0:i1], tr["y"][i0:i1], tr["th"][i0:i1] = \
                    res.hitch_x[i0:i1], res.hitch_y[i0:i1], th[i0:i1]
        res.train = train
        res.cart_x, res.cart_y, res.cart_th = train[0]["x"], train[0]["y"], train[0]["th"]
        # carts standing alone (picked up later / dropped off)
        T_end = float(t[-1])
        for i0 in (starts if ht == "lurking" else []):
            if i0 > 0:   # lurking pick-up: the cart waited at this spot since the cycle start
                tr = train[0]
                res.parked.append({"x": float(tr["x"][i0]), "y": float(tr["y"][i0]), "th": float(tr["th"][i0]),
                                   "t0": 0.0, "t1": float(t[i0])})
        for i1 in ends:
            if i1 < n:   # dropped here: stays until the end of the cycle
                for tr in train:
                    res.parked.append({"x": float(tr["x"][i1 - 1]), "y": float(tr["y"][i1 - 1]),
                                       "th": float(tr["th"][i1 - 1]), "t0": float(t[i1]), "t1": T_end})
        if ht == "tow":
            prev = th
            for k, tr in enumerate(train):
                tr["art"] = np.degrees(wrap(prev - tr["th"]))
                prev = tr["th"]
            res.articulation_deg = train[0]["art"]
            if att.any():
                for k, tr in enumerate(train):
                    amax = float(np.nanmax(np.abs(tr["art"])))
                    if amax > agv.cart.max_articulation_deg:
                        joint = "AGV / trolley 1" if k == 0 else f"trolley {k} / trolley {k + 1}"
                        warnings.append(f"Articulation {joint} reaches {amax:.0f} deg (limit "
                                        f"{agv.cart.max_articulation_deg:.0f}) -> jackknife risk.")

    res.loads = compute_loads(res)
    return res
