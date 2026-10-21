"""Closed-loop magnetic-tape following.

The drawn route is the tape. For every drive run the simulator steps a small physical model
at the control rate (100 Hz) and runs the AGV's own controller (kayaba_pid.KayabaController, ported from
modes.py / core/pid.py) at the control period (DT = 10 ms, one MGS1600 frame per cycle):

  MGS1600 model   the sensor is a 160 mm line across the vehicle; the reading is where that
                  line crosses the tape (1 mm integer, optional noise, CAN latency). No
                  crossing within +-sensor_range_mm -> tape_detected = False.
  controller      speed mode + ramp, sensor sanity / slew limit, PID + speed reduction,
                  rpm -> AO volt (0..5 V clamp, 12-bit DAC), brake on tape loss / stop tag
  drive plant     per wheel: AO volt -> rpm via the calibration map (x plant_scale for a
                  mis-calibrated side) -> wheel speed, first-order lag motor_time_constant;
                  brake DO -> decelerate at brake_decel
  kinematics      differential drive about the drive-axle centre
  RFID tags       every route point is a tag at the nearest place on the tape, read by the
                  'rfid' sensor (or the tape sensor if there is none): a point's `mode`
                  switches speed_mode, `speed` caps the speed, and the stop at the end of a
                  run brakes when the tag is read (like stop_agv / sequence_stop) - so the
                  AGV overshoots the tag by its braking distance, as in reality.
"""
from __future__ import annotations

import math
from collections import deque
from typing import List

import numpy as np

from .kayaba_pid import KNOWN_MODES, KayabaController, voltage_to_rpm

PLANT_DT = 0.01
SHARED_TAG_TOL = 0.25      # m: a speed tag this close to the tape of another leg is read there too


class Tape:
    """All runs of the plan concatenated: the physical tape (with direction of travel)."""

    EXT = 1.0      # m of tape assumed beyond a dead end (stop before reversing / end of the route)

    def __init__(self, plan):
        xs, ys, ss, run_of = [], [], [], []
        self.range = {}
        s0 = 0.0
        i0 = 0
        runs = plan.runs

        def straight(x0, y0, hx, hy, L, run_idx):
            n = max(2, int(L / 0.01))
            f = np.arange(1, n + 1) / n * L
            return x0 + hx * f, y0 + hy * f, run_idx

        if runs and not plan.loop:     # tape before the first point
            r = runs[0]
            hx, hy = -math.cos(r.heading[0]), -math.sin(r.heading[0])
            ex, ey, _ = straight(r.x[0], r.y[0], hx, hy, self.EXT, r.index)
            ex, ey = ex[::-1], ey[::-1]
            seg = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(ex), np.diff(ey)))])
            xs.append(ex)
            ys.append(ey)
            ss.append(seg - seg[-1] - float(np.hypot(ex[-1] - r.x[0], ey[-1] - r.y[0])))
            run_of.append(np.full(len(ex), r.index))
            i0 += len(ex)
        for k, run in enumerate(runs):
            n = len(run.x)
            xs.append(run.x)
            ys.append(run.y)
            ss.append(run.s + s0)
            run_of.append(np.full(n, run.index))
            self.range[run.index] = (i0, i0 + n)
            s0 += float(run.s[-1])
            i0 += n
            nxt = runs[k + 1] if k + 1 < len(runs) else (runs[0] if plan.loop else None)
            dead_end = nxt is None or nxt.direction != run.direction or run.end_pivot
            if dead_end:               # tape continues a little past the stop tag
                hx, hy = math.cos(run.heading[-1]), math.sin(run.heading[-1])
                ex, ey, _ = straight(run.x[-1], run.y[-1], hx, hy, self.EXT, run.index)
                xs.append(ex)
                ys.append(ey)
                ss.append(s0 + np.arange(1, len(ex) + 1) * (self.EXT / len(ex)))
                run_of.append(np.full(len(ex), run.index))
                s0 += self.EXT
                i0 += len(ex)
            self.range[run.index] = self.range[run.index] + (i0,)   # (lo, hi, hi incl. extension)
        if plan.loop and runs:       # tape continues into the first run after the last one
            r0 = runs[0]
            xs.append(r0.x)
            ys.append(r0.y)
            ss.append(r0.s + s0)
            run_of.append(np.full(len(r0.x), r0.index))
        self.x = np.concatenate(xs)
        self.y = np.concatenate(ys)
        self.s = np.concatenate(ss)
        self.run_of = np.concatenate(run_of)
        self.n = len(self.x)

    def nearest(self, px, py, lo=0, hi=None):
        hi = self.n if hi is None else hi
        d = np.hypot(self.x[lo:hi] - px, self.y[lo:hi] - py)
        k = int(np.argmin(d))
        return lo + k, float(d[k])


def _crossings(tape: Tape, cx, cy, fx, fy, lo, hi):
    """Where the sensor line (through C, perpendicular to f) crosses the tape between lo..hi."""
    X, Y = tape.x[lo:hi], tape.y[lo:hi]
    f = (X - cx) * fx + (Y - cy) * fy
    sgn = np.sign(f)
    idx = np.nonzero((sgn[:-1] * sgn[1:] <= 0) & ~((f[:-1] == 0) & (f[1:] == 0)))[0]
    out = []
    for i in idx:
        den = f[i] - f[i + 1]
        u = 0.0 if den == 0 else f[i] / den
        px = X[i] + u * (X[i + 1] - X[i])
        py = Y[i] + u * (Y[i + 1] - Y[i])
        lam = (px - cx) * (-fy) + (py - cy) * fx        # + = to the vehicle's LEFT
        s = tape.s[lo + i] + u * (tape.s[lo + i + 1] - tape.s[lo + i])
        out.append((lam, s, lo + i, px, py))
    return out


class _Sensor:
    """MGS1600 on the vehicle (body position sx, sy; facing +1 = forward, -1 = backward)."""

    def __init__(self, tape: Tape, lf, sx, sy, facing, rng):
        self.tape = tape
        self.lf = lf
        self.sx, self.sy, self.facing = sx, sy, facing
        pos_right = str(lf.front_sensor_positive).lower().startswith("r")
        # front unit: +mm = tape to the vehicle's right if pos_right. Rear unit = mirrored.
        self.k = (-1.0 if pos_right else 1.0) * (1.0 if facing > 0 else -1.0)
        self.range_m = lf.sensor_range_mm / 1000.0
        self.idx = None
        self.lam = 0.0
        self.s = 0.0
        self.pt = (math.nan, math.nan)
        self.rng = rng

    def read(self, x, y, th, run_lo, run_hi):
        c, s_ = math.cos(th), math.sin(th)
        cx = x + c * self.sx - s_ * self.sy
        cy = y + s_ * self.sx + c * self.sy
        tape = self.tape
        if self.idx is None:                     # start of the run (the tape can pass here again later)
            self.idx, _ = tape.nearest(cx, cy, run_lo, min(run_hi, run_lo + 300))
            self.s = float(tape.s[self.idx])
            self.pt = (float(tape.x[self.idx]), float(tape.y[self.idx]))
        cands = []
        for win in (30, 150):
            lo = max(0, self.idx - win)
            hi = min(tape.n, self.idx + win)
            cands = [cd for cd in _crossings(tape, cx, cy, c, s_, lo, hi) if abs(cd[0]) <= self.range_m]
            if cands:
                break
        if not cands:
            return {"tape_detected": False, "left_mm": 0, "right_mm": 0}
        # track continuity: prefer crossings on the current run, then closest to last reading
        # (a tape used twice / merging tapes: stay on the pass that continues forward along the run)
        s0 = self.s if self.pt[0] == self.pt[0] else None
        cands.sort(key=lambda cd: (0 if run_lo <= cd[2] < run_hi else 1,
                                   0 if (s0 is None or s0 - 0.3 <= cd[1] <= s0 + 1.0) else 1,
                                   abs(cd[0] - self.lam)))
        lam, s, i, px, py = cands[0]
        self.idx, self.lam, self.s, self.pt = i, lam, s, (px, py)
        mm = self.k * lam * 1000.0
        if self.lf.sensor_noise_mm > 0:
            mm += self.rng.normal(0.0, self.lf.sensor_noise_mm)
        mm = int(round(mm))
        return {"tape_detected": True, "left_mm": mm, "right_mm": mm}


def _passes(tape: Tape, px, py, lo, hi, tol):
    """Every place where the tape between lo..hi passes within tol of (px, py): [(index, dist)]."""
    d = np.hypot(tape.x[lo:hi] - px, tape.y[lo:hi] - py)
    m = d < tol
    out = []
    if not m.any():
        return out
    idx = np.nonzero(m)[0]
    groups = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
    for g in groups:
        k = g[int(np.argmin(d[g]))]
        out.append((lo + int(k), float(d[k])))
    return out


def _tags_for_run(route, plan, tape: Tape, run, next_run):
    """RFID tags met on this run: (s_on_tape, kind, value), plus the reads of speed tags that belong to
    another stretch but lie on this run's tape too ('foreign': [(s, waypoint index)]) - an RFID tag is
    read every time the AGV drives over it, in both directions."""
    tags = []
    foreign = []
    lo, hi, hx = tape.range[run.index]
    hi2 = min(tape.n, hx + len(next_run.x)) if next_run is not None else hx
    wps = route.waypoints
    n = len(wps)
    # waypoints that belong to this run: from start_wp up to end_wp (wrapping for loops)
    ids = []
    k = run.start_wp
    while True:
        ids.append(k)
        if k == run.end_wp and len(ids) > 1:
            break
        k = (k + 1) % n
        if len(ids) > n + 1:
            break
    design = {}
    cursor = lo
    for j, wi in enumerate(ids):
        wp = wps[wi]
        if j == 0:
            i_t = lo
        else:
            # the place on THIS pass: the earliest near-nearest point after the previous tag
            d = np.hypot(tape.x[cursor:hi] - wp.x, tape.y[cursor:hi] - wp.y)
            dmin = float(d.min())
            i_t = cursor + int(np.nonzero(d <= dmin + 0.05)[0][0])
            cursor = i_t
        s_tag = float(tape.s[i_t])
        if j < len(ids) - 1:                         # leg leaving this point starts here
            if wp.mode and wp.mode.lower() != "default":
                tags.append((s_tag, "mode", wp.mode.strip().upper()))
                design.setdefault(wi, []).append(s_tag)
            tags.append((s_tag, "cap", float(wp.speed) if wp.speed else math.inf))
    i_end = cursor
    s_lo, s_end = float(tape.s[lo]), float(tape.s[i_end])
    for wi, wp in enumerate(wps):
        if not wp.mode or wp.mode.lower() == "default":
            continue
        for i_p, _d in _passes(tape, wp.x, wp.y, lo, i_end + 1, SHARED_TAG_TOL):
            s_p = float(tape.s[i_p])
            if not (s_lo + 0.05 < s_p < s_end - 0.05):
                continue
            if any(abs(s_p - sd) < 0.5 for sd in design.get(wi, [])):
                continue                             # that is its own (designed) read
            tags.append((s_p, "mode", wp.mode.strip().upper()))
            foreign.append((s_p, wi))
    tags.append((s_end, "stop", run.end_name))
    tags.sort(key=lambda tg: (tg[0], 0 if tg[1] != "stop" else 1))
    return tags, (lo, hi2), foreign


class LineFollowSession:
    """Keeps the controller state that survives between runs (speed_mode, forward PID)."""

    def __init__(self, agv, route, plan, seed=1):
        self.agv, self.route, self.plan = agv, route, plan
        self.lf = agv.line_follow
        self.tape = Tape(plan)
        self.rng = np.random.default_rng(seed)
        mode0 = (route.default_mode or "").strip().upper()
        self.speed_mode = mode0 if mode0 and mode0 != "DEFAULT" else self.lf.start_mode.upper()
        self.extra_modes = {m.name.strip().upper(): (m.v_straight, m.accel, m.decel)
                            for m in agv.speed_modes if m.name.strip().upper() not in KNOWN_MODES}
        self.fwd = KayabaController(self.lf, agv.wheels.diameter, agv.drivetrain.gear_ratio,
                                    reverse=False, speed_mode=self.speed_mode, extra_modes=self.extra_modes)
        self.mode_names = []
        self.events: List[dict] = []
        self.warned_modes = set()
        from .drive import DriveLimits
        self.drive = DriveLimits(agv)
        self._lat_warned = False
        self.stats = None                  # PID auto-tune: per gain set {'e': [...], ...} when a dict
        # HIGH / SLOW / APPROACH come from line_follow (as on the real AGV), not from the Speed modes table
        link = {"HIGH": "speed_high", "SLOW": "speed_slow", "APPROACH": "speed_approach"}
        lf_ = self.lf
        rpm_top = min(lf_.rpm_per_volt_left * lf_.ao_max_volt + lf_.rpm_volt_offset_left,
                      lf_.rpm_per_volt_right * lf_.ao_max_volt + lf_.rpm_volt_offset_right)
        vmax = min(agv.max_wheel_speed, rpm_top * math.pi * agv.wheels.diameter / 60 / agv.drivetrain.gear_ratio)
        for m in agv.speed_modes:
            key = m.name.strip().upper()
            if key in link:
                v_lf = getattr(self.lf, link[key])
                if abs(m.v_straight - v_lf) > 0.005:
                    self.events.append({"t": 0.0, "kind": "warning", "text":
                                        f"Speed modes table says {key} = {m.v_straight:.2f} m/s, but with line following "
                                        f"on the AGV runs {key} at line_follow.{link[key]} = {v_lf:.2f} m/s "
                                        f"(like the real controller). Change it under Line following."})
        for key, attr in link.items():
            v_lf = getattr(self.lf, attr)
            if v_lf > vmax + 0.005:
                self.events.append({"t": 0.0, "kind": "note", "text":
                                    f"line_follow.{attr} = {v_lf:.2f} m/s is above what the motors can do at the "
                                    f"{self.lf.ao_max_volt:.1f} V output limit ({vmax:.2f} m/s): {key} runs at "
                                    f"{vmax:.2f} m/s."})
        for key, (v, _, _) in self.extra_modes.items():
            if v > vmax + 0.005:
                self.events.append({"t": 0.0, "kind": "note", "text":
                                    f"Speed mode {key} = {v:.2f} m/s is above what the motors can do ({vmax:.2f} m/s)."})
        rf = [s for s in agv.sensors if "rfid" in (s.type or "").lower() or "rfid" in s.name.lower()]
        self.rfid_x = rf[0].x if rf else None

    def _sensor_for(self, direction):
        name = self.agv.tracking.forward if direction > 0 else self.agv.tracking.reverse
        sn = self.agv.sensor(name)
        return sn

    def drive_run(self, run, next_run, pose, dt_out, m_tow=0.0, m_lurk=0.0):
        """Simulate one run closed-loop from `pose` (axle x, y, heading).

        Returns dict of arrays sampled every dt_out plus 'aborted' / 'lost' info."""
        agv, lf, tape = self.agv, self.lf, self.tape
        direction = run.direction
        sn = self._sensor_for(direction)
        sensor = _Sensor(tape, lf, sn.x, sn.y, direction, self.rng)
        reader_back = 0.0 if self.rfid_x is None else direction * (sn.x - self.rfid_x)
        lo, hi, _ = tape.range[run.index]
        tags, (wlo, whi), foreign = _tags_for_run(self.route, self.plan, tape, run, next_run)
        if foreign and direction > 0:
            wps_ = self.route.waypoints
            self.events.append({"t": 0.0, "kind": "note", "text":
                                f"Shared tape on '{run.label}': it also reads "
                                + ", ".join(f"{wps_[wi].mode.strip().upper()} (tag of point {wi + 1})" for _, wi in foreign)
                                + " - RFID tags work in both directions."})
        if direction > 0:
            ctl = self.fwd
            ctl.speed_mode = self.speed_mode
        else:
            ctl = KayabaController(lf, agv.wheels.diameter, agv.drivetrain.gear_ratio, reverse=True,
                                   extra_modes=self.extra_modes)
        ctl.brake_reset()
        ctl.tape_was_lost = False
        ctl.speed_cap = math.inf

        b = agv.wheels.track
        gr, wd = agv.drivetrain.gear_ratio, agv.wheels.diameter
        rpm2v = math.pi * wd / 60.0 / gr
        tau = max(lf.motor_time_constant, 1e-4)
        # first-order motor lag, exact over one plant step (stable for any time constant; explicit
        # Euler overshoots below 10 ms and diverges below 5 ms)
        k_lag = 1.0 - math.exp(-PLANT_DT / tau)
        x, y, th = pose
        vl = vr = 0.0
        mode = "drive"
        tl_ = tr_ = 0.0                     # wheel target speeds from the last AO command
        t = 0.0
        next_ctrl = 0.0
        hist = deque()
        # sensor latency: the controller sees the pose of lat_n plant steps ago. hist holds the current
        # pose and the lat_n before it, so hist[0] is exactly lat_n x PLANT_DT old.
        lat_n = max(0, int(round(lf.sensor_latency_s / PLANT_DT)))
        if abs(lat_n * PLANT_DT - lf.sensor_latency_s) > 1e-6 and not self._lat_warned:
            self._lat_warned = True
            self.events.append({"t": 0.0, "kind": "note", "text":
                                f"sensor_latency_s {lf.sensor_latency_s:.4f} s is rounded to {lat_n * PLANT_DT:.3f} s "
                                f"(the plant runs in {PLANT_DT * 1000:.0f} ms steps)."})
        tag_i = 0
        stop_tag = None
        stopping = False
        lost_since = None
        stall_since = None
        out = {k: [] for k in ("t", "x", "y", "th", "rx", "ry", "s", "pv", "lost", "cl", "cr", "tv", "e",
                                "out", "sat", "sm", "vp", "wp", "ap")}
        v_rec, t_rec = 0.0, 0.0          # plant speed at the previous output sample (for 'ap')

        def mode_code():
            name = "REVERSE" if direction < 0 else ctl.speed_mode
            if name not in self.mode_names:
                self.mode_names.append(name)
            return float(self.mode_names.index(name))
        last_frame = None
        next_out = dt_out
        length = float(tape.s[hi - 1] - tape.s[lo])
        t_max = 60.0 + 6.0 * length / max(min(lf.speed_slow, lf.speed_reverse, lf.speed_approach, 0.05), 0.05)
        result = {"aborted": False, "lost_events": [], "stop_overshoot": None, "limited_s": 0.0,
                  "runaway": False}
        # load: trolleys / lurking cart / slope limit speeding up and braking (drive.py)
        drv = self.drive
        use_load = bool(agv.simulation.load_effects)
        stats = self.stats

        def stats_for(c, dirn):
            key = "REVERSE" if dirn < 0 else ("HIGH" if c.speed_mode == "HIGH" else
                                               "APPROACH" if c.speed_mode == "APPROACH" else "SLOW")
            d = stats.get(key)
            if d is None:
                d = stats[key] = {"e": [], "jit2": 0.0, "last_out": None, "cross": 0, "sign": 0,
                                  "dist": 0.0, "sat": 0.0, "lost": 0}
            return d
        grades = np.asarray(run.grade, float) * direction
        flat = not np.any(np.abs(grades) > 1e-9)
        st_cache = {}

        def load_state(s_now):
            g = 0.0 if flat else round(float(np.interp(s_now, run.s, grades)), 2)
            st = st_cache.get(g)
            if st is None:
                st = st_cache[g] = drv.state(m_tow, m_lurk, g)
            return st
        st = load_state(float(run.s[0]))
        result["load"] = st if use_load else None
        s_start = float(tape.s[lo])
        first = sensor.read(x, y, th, lo, whi)
        if not first["tape_detected"]:
            result["aborted"] = True
            result["lost_events"].append({"t": 0.0, "x": x, "y": y, "kind": "no tape at start"})
        sat = 0.0
        last_frame = first
        for k_, v_ in (("t", 0.0), ("x", x), ("y", y), ("th", th), ("rx", sensor.pt[0]), ("ry", sensor.pt[1]),
                       ("s", float(run.s[0])), ("pv", float(first["left_mm"]) if first["tape_detected"] else math.nan),
                       ("lost", 0.0 if first["tape_detected"] else 1.0), ("cl", 0.0), ("cr", 0.0), ("tv", 0.0),
                       ("e", math.nan), ("out", 0.0), ("sat", 0.0), ("sm", mode_code()),
                       ("vp", 0.0), ("wp", 0.0), ("ap", 0.0)):
            out[k_].append(v_)
        while not result["aborted"]:
            # ---------------- controller (one frame per DT) ----------------
            if t >= next_ctrl - 1e-12:
                next_ctrl += lf.dt
                hist_pose = hist[0] if (lat_n and len(hist) > lat_n) else (x, y, th)
                frame = sensor.read(*hist_pose, lo, whi)
                last_frame = frame
                s_read = sensor.s - reader_back
                # RFID tags passed
                while tag_i < len(tags) and frame["tape_detected"] and s_read >= tags[tag_i][0] - 1e-9:
                    s_tag, kind, val = tags[tag_i]
                    tag_i += 1
                    if kind == "mode" and direction > 0:
                        if val not in KNOWN_MODES and val not in self.warned_modes:
                            self.warned_modes.add(val)
                            if val in self.extra_modes:
                                self.events.append({"t": t, "kind": "note", "text":
                                                    f"speed_mode '{val}' ({self.extra_modes[val][0]:.2f} m/s) is taken "
                                                    f"from the Speed modes table (SLOW gains). The AGV's modes.py does "
                                                    f"not know '{val}' yet - add it there, or the real AGV runs it at "
                                                    f"HIGH speed."})
                            else:
                                self.events.append({"t": t, "kind": "warning", "text":
                                                    f"speed_mode '{val}' is not in the Speed modes table and not handled "
                                                    f"by modes.py: it runs at HIGH speed with SLOW gains"})
                        self.speed_mode = ctl.speed_mode = val
                    elif kind == "cap":
                        ctl.speed_cap = val
                    elif kind == "stop" and s_tag > s_start + 0.02:
                        stop_tag = s_tag
                        stopping = True
                if stopping:
                    mode = "brake"
                    ctl.brake_reset()
                else:
                    cmd, lv, rv = ctl.step(frame, t)
                    if cmd == "brake":
                        mode = "brake"
                        if lost_since is None:
                            lost_since = t
                            if stats is not None:
                                stats_for(ctl, direction)["lost"] += 1
                            result["lost_events"].append({"t": t, "x": x, "y": y, "kind": "tape lost",
                                                          "pv_last": sensor.lam * sensor.k * 1000})
                    elif cmd == "drive":
                        if lost_since is not None:
                            result["lost_events"].append({"t": t, "x": x, "y": y, "kind": "reacquired"})
                            lost_since = None
                        mode = "drive"
                        sgn = 1.0 if direction > 0 else -1.0
                        rl = max(0.0, voltage_to_rpm(lv, lf.rpm_per_volt_left, lf.rpm_volt_offset_left)) \
                            if lv > 0 else 0.0
                        rr = max(0.0, voltage_to_rpm(rv, lf.rpm_per_volt_right, lf.rpm_volt_offset_right)) \
                            if rv > 0 else 0.0
                        tl_ = sgn * rl * lf.plant_scale_left * rpm2v
                        tr_ = sgn * rr * lf.plant_scale_right * rpm2v
                        dbg = ctl.debug
                        sat = 1.0 if (lv >= lf.ao_max_volt - 1e-6 or rv >= lf.ao_max_volt - 1e-6
                                      or abs(dbg["output"]) >= lf.output_clamp_rpm - 1e-6) else 0.0
                        if stats is not None and ctl.current_target_speed > 0.05:
                            S_ = stats_for(ctl, direction)
                            pv_ = float(frame["left_mm"])
                            S_["e"].append(abs(pv_))
                            o_ = float(dbg["output"])
                            if S_["last_out"] is not None:
                                S_["jit2"] += (o_ - S_["last_out"]) ** 2
                            S_["last_out"] = o_
                            if (pv_ > 2.0 and S_["sign"] < 0) or (pv_ < -2.0 and S_["sign"] > 0):
                                S_["cross"] += 1
                            if abs(pv_) > 2.0:
                                S_["sign"] = 1 if pv_ > 0 else -1
                            S_["dist"] += abs(0.5 * (vl + vr)) * lf.dt
                            S_["sat"] += sat
            # ---------------- plant ----------------
            if use_load and not flat:
                st = load_state(sensor.s - float(tape.s[lo]) + float(run.s[0]))
            if not use_load:                     # kinematic drive (speed independent of the load)
                if mode == "brake":
                    dv = lf.brake_decel * PLANT_DT
                    vl = vl - math.copysign(min(dv, abs(vl)), vl)
                    vr = vr - math.copysign(min(dv, abs(vr)), vr)
                else:
                    k = k_lag
                    vl += k * (tl_ - vl)
                    vr += k * (tr_ - vr)
            elif mode == "brake":
                bd = drv.brake_mag(st, 0.5 * (vl + vr))
                if bd > 0:
                    dv = bd * PLANT_DT
                    vl = vl - math.copysign(min(dv, abs(vl)), vl)
                    vr = vr - math.copysign(min(dv, abs(vr)), vr)
                else:                               # the train pushes the braked AGV downhill
                    result["runaway"] = True
                    vl -= math.copysign(bd * PLANT_DT, vl or direction)
                    vr -= math.copysign(bd * PLANT_DT, vr or direction)
            else:
                vl, vr, lim_ = drv.step(st, vl, vr, (tl_ - vl) * k_lag / PLANT_DT, (tr_ - vr) * k_lag / PLANT_DT,
                                        PLANT_DT)
                if lim_:
                    result["limited_s"] += PLANT_DT
            v = 0.5 * (vl + vr)
            w = (vr - vl) / b
            thm = th + 0.5 * w * PLANT_DT
            x += v * math.cos(thm) * PLANT_DT
            y += v * math.sin(thm) * PLANT_DT
            th += w * PLANT_DT
            t += PLANT_DT
            if lat_n:
                hist.append((x, y, th))
                if len(hist) > lat_n + 1:
                    hist.popleft()
            # ---------------- record ----------------
            if t >= next_out - 1e-12:
                next_out += dt_out
                det = bool(last_frame and last_frame["tape_detected"])
                dbg = ctl.debug if mode == "drive" else {}
                out["t"].append(t)
                out["x"].append(x)
                out["y"].append(y)
                out["th"].append(th)
                out["rx"].append(sensor.pt[0])
                out["ry"].append(sensor.pt[1])
                out["s"].append(sensor.s - float(tape.s[lo]) + float(run.s[0]))
                out["pv"].append(float(last_frame["left_mm"]) if det else math.nan)
                out["lost"].append(0.0 if det else 1.0)
                out["cl"].append(dbg.get("left_rpm", 0.0))
                out["cr"].append(dbg.get("right_rpm", 0.0))
                out["tv"].append(ctl.current_target_speed)
                out["e"].append(dbg.get("e", math.nan))
                out["out"].append(dbg.get("output", 0.0))
                out["sat"].append(sat if mode == "drive" else 0.0)
                out["sm"].append(mode_code())
                # the plant's own speed / yaw rate / mean acceleration over the output interval: the loads
                # use these instead of differentiating the recorded positions twice (noise)
                v_now = 0.5 * (vl + vr)
                out["vp"].append(v_now)
                out["wp"].append((vr - vl) / b)
                out["ap"].append((v_now - v_rec) / max(t - t_rec, 1e-9))
                v_rec, t_rec = v_now, t
            # ---------------- end conditions ----------------
            standing = abs(vl) < 1e-3 and abs(vr) < 1e-3
            if stopping and standing:
                result["stop_overshoot"] = (sensor.s - reader_back) - stop_tag
                break
            if lost_since is not None and standing and t - lost_since > lf.lost_abort_s:
                result["aborted"] = True
                break
            # stalled: the drive asks to move but the AGV does not (load calc: not enough grip / torque)
            if mode == "drive" and ctl.current_target_speed > 0.05 and abs(0.5 * (vl + vr)) < 0.005:
                if stall_since is None:
                    stall_since = t
                elif t - stall_since > 5.0:
                    result["aborted"] = True
                    result["lost_events"].append({"t": t, "x": x, "y": y, "kind": "stalled"})
                    break
            else:
                stall_since = None
            if t > t_max:
                result["aborted"] = True
                result["lost_events"].append({"t": t, "x": x, "y": y, "kind": "timeout"})
                break
        for k_ in out:
            out[k_] = np.asarray(out[k_], float)
        result["arrays"] = out
        result["pose"] = (x, y, th)
        result["speed_mode"] = self.speed_mode
        return result
