"""The AGV's own tape-following controller, ported 1:1 from 'AGV Kayaba (2 Sensor)'.

  PIDController       <- core/pid.py        (only change: the loop period comes from the
                                              simulation clock instead of time.perf_counter)
  mps_to_rpm / rpm_to_voltage / voltage_to_rpm <- motion.py
  speed-mode / gain / ramp / sensor-sanity logic <- modes.py auto_mode() and reverse_mode()

Keep this file in step with the real code when the controller changes.
"""
from __future__ import annotations

import math


class PIDController:

    def __init__(self, kp, td, n, ti, ti_deadband, ti_max, dt, output_clamp,
                 v_red_coef, sr_alpha, sr_cap):
        self._kp = kp
        self._td = td
        self._n = n
        self._ti = ti
        self._ti_deadband = ti_deadband
        self._ti_max = ti_max
        self._dt = dt
        self._clamp_hi = +output_clamp
        self._clamp_lo = -output_clamp
        self._v_red_coef = v_red_coef
        self._sr_alpha = sr_alpha
        self._sr_cap = sr_cap
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.last_e = 0.0
        self.filtered_d = 0.0
        self.speed_reduction = 0.0
        self._last_t = None

    def update_gains(self, kp, td, n, v_red_coef, clamp_hi=None, clamp_lo=None):
        self._kp = kp
        self._td = td
        self._n = n
        self._v_red_coef = v_red_coef
        if clamp_hi is not None:
            self._clamp_hi = clamp_hi
        if clamp_lo is not None:
            self._clamp_lo = clamp_lo

    def compute(self, pv, base_rpm, error_sign=-1.0, feedforward=0.0, now=None):
        e = error_sign * pv

        # measured loop period (simulation clock), clamped like the original
        if self._last_t is None or now is None:
            dt = self._dt
        else:
            dt = now - self._last_t
            dt = max(0.2 * self._dt, min(5.0 * self._dt, dt))
        self._last_t = now

        p_term = self._kp * e

        if self._ti is not None:
            if abs(e) < self._ti_deadband:
                self.integral += e * dt
                self.integral = max(-self._ti_max, min(self._ti_max, self.integral))
            i_term = self._kp * (1.0 / self._ti) * self.integral
        else:
            i_term = 0.0

        de = (e - self.last_e) / dt
        alpha = dt / ((self._td / self._n) + dt)
        raw_d = self._kp * self._td * de
        self.filtered_d += alpha * (raw_d - self.filtered_d)

        output = p_term + i_term + self.filtered_d
        output = max(self._clamp_lo, min(self._clamp_hi, output))
        output_total = output + feedforward

        raw_sr = abs(e * self._v_red_coef) + abs(de) * 0.5
        raw_sr = min(raw_sr, base_rpm * self._sr_cap)
        self.speed_reduction += self._sr_alpha * (raw_sr - self.speed_reduction)

        left_rpm = base_rpm - self.speed_reduction + output_total
        right_rpm = base_rpm - self.speed_reduction - output_total
        self.last_e = e
        return left_rpm, right_rpm, {"e": e, "p": p_term, "i": i_term, "d": self.filtered_d,
                                     "output": output, "feedforward": feedforward,
                                     "output_total": output_total,
                                     "speed_reduction": self.speed_reduction}


# ----------------------------------------------------------------------------- motion.py
def mps_to_rpm(v, wheel_diameter, gear_ratio):
    wheel_circumference = 3.14159 * wheel_diameter          # same constant as config.py
    return v * 60 / wheel_circumference * gear_ratio


def rpm_to_voltage(rpm, k, o, v_max=5.0):
    """rpm <= 0 -> 0 V (true stop), else clamp to [0, v_max] V."""
    if rpm <= 0:
        return 0.0
    return max(0.0, min(v_max, (rpm - o) / k))


def voltage_to_rpm(v, k, o):
    return k * v + o


# ----------------------------------------------------------------------------- modes.py logic
KNOWN_MODES = ("HIGH", "SLOW", "APPROACH", "MAX", "MED", "MEDIUM")   # modes modes.py switches gains for


class KayabaController:
    """auto_mode() / reverse_mode() of modes.py, minus I/O: one call per control period."""

    def __init__(self, lf, wheel_diameter, gear_ratio, reverse=False, speed_mode="SLOW", extra_modes=None):
        self.lf = lf
        # speeds of the modes besides HIGH / SLOW / APPROACH: name -> (speed m/s, accel, decel).
        # MAX / MED come from line_follow (speed_max / speed_med, ACCEL_RATE / DECEL_RATE) and have
        # their own gain sets (fallback HIGH / SLOW); other modes come from the Speed modes table and
        # use the SLOW gains, like the else-branch of auto_mode's gain switch.
        self.extra_modes = dict(extra_modes or {})
        self.wd = wheel_diameter
        self.gr = gear_ratio
        self.reverse = reverse
        g = (lf.kp_reverse, lf.td_reverse, lf.n_reverse, lf.v_red_coef_reverse) if reverse else \
            (lf.kp_slow, lf.td_slow, lf.n_slow, lf.v_red_coef_slow)
        self.pid = PIDController(kp=g[0], td=g[1], n=g[2], ti=lf.ti, ti_deadband=lf.ti_deadband,
                                 ti_max=lf.ti_max, dt=lf.dt, output_clamp=lf.output_clamp_rpm,
                                 v_red_coef=g[3], sr_alpha=lf.sr_alpha, sr_cap=lf.sr_cap)
        self.error_sign = lf.error_sign_reverse if reverse else lf.error_sign_forward
        self.current_target_speed = 0.0
        self.speed_mode = speed_mode
        self.last_speed_mode = None
        self.last_valid_pv = None
        self.ff_filtered = 0.0
        self.tape_was_lost = False
        self.speed_cap = math.inf          # waypoint 'speed' cap (reverse: set_reverse_speed override)
        self.debug = {}

    # ---- helpers ---------------------------------------------------------------------------
    def brake_reset(self):
        """What the original does on tape loss / stop / emergency."""
        self.current_target_speed = 0.0
        self.pid.reset()
        self.last_valid_pv = None
        self.ff_filtered = 0.0

    def _target_speed(self):
        lf = self.lf
        if self.reverse:
            ts = lf.speed_reverse
        elif self.speed_mode == "APPROACH":
            ts = lf.speed_approach
        elif self.speed_mode == "SLOW":
            ts = lf.speed_slow
        elif self._xkey() in self.extra_modes:
            ts = self.extra_modes[self._xkey()][0]
        else:                                   # modes.py: anything else runs at HIGH speed
            ts = lf.speed_high
        return min(ts, self.speed_cap)

    def _xkey(self):
        """Key of the current mode in extra_modes (MEDIUM and MED are the same mode)."""
        m = (self.speed_mode or "").upper()
        return "MED" if m == "MEDIUM" else m

    def _switch_gains(self):
        if self.reverse or self.speed_mode == self.last_speed_mode:
            return
        lf = self.lf
        kp, td, n, vr = lf.gains(self.speed_mode)          # HIGH / MAX / MED / SLOW / else SLOW
        if self.speed_mode == "APPROACH":
            self.pid.update_gains(kp, td, n, vr, clamp_hi=lf.output_clamp_into_bar_approach,
                                  clamp_lo=-lf.output_clamp_away_approach)
        else:
            self.pid.update_gains(kp, td, n, vr, clamp_hi=lf.output_clamp_rpm, clamp_lo=-lf.output_clamp_rpm)
        self.last_speed_mode = self.speed_mode

    # ---- one control cycle -----------------------------------------------------------------
    def step(self, frame, now):
        """frame = {'tape_detected': bool, 'left_mm': int, 'right_mm': int}.

        Returns ('brake', None, None) or ('drive', left_volt, right_volt) or ('hold', None, None)
        (hold = frame skipped, previous AO command stays)."""
        lf = self.lf
        target_speed = self._target_speed()
        self._switch_gains()

        if not frame["tape_detected"]:
            self.brake_reset()
            self.tape_was_lost = True
            return "brake", None, None
        if self.tape_was_lost:
            self.tape_was_lost = False          # "TAPE REACQUIRED — resuming" (ramps from 0)

        xm = self.extra_modes.get(self._xkey()) if not self.reverse else None
        if self.current_target_speed < target_speed:
            acc = xm[1] if xm else lf.accel_rate
            self.current_target_speed = min(target_speed, self.current_target_speed + acc * lf.dt)
        elif self.current_target_speed > target_speed:
            rate = lf.approach_decel_rate if (self.speed_mode == "APPROACH" and not self.reverse) \
                else (xm[2] if xm else lf.decel_rate)
            self.current_target_speed = max(target_speed, self.current_target_speed - rate * lf.dt)
        base_rpm = mps_to_rpm(self.current_target_speed, self.wd, self.gr)

        pv = frame["right_mm"] if self.reverse else frame["left_mm"]   # single tape: both equal
        if pv is None or abs(pv) > lf.sensor_max_mm:
            return "hold", None, None
        if self.last_valid_pv is not None:
            step = pv - self.last_valid_pv
            if abs(step) > lf.sensor_max_step_mm:
                pv = self.last_valid_pv + (lf.sensor_max_step_mm if step > 0 else -lf.sensor_max_step_mm)
        self.last_valid_pv = pv

        ff = 0.0
        if not self.reverse:
            ff_target = 0.0
            if lf.ff_enabled and self.speed_mode in (lf.ff_curve_modes or []):
                sign = -1.0 if str(lf.ff_direction).lower() == "left" else 1.0
                ff_target = lf.ff_scale * sign * 0.5 * base_rpm * (lf.ff_track_width / lf.ff_curve_radius)
            a = lf.ff_alpha if abs(ff_target) > abs(self.ff_filtered) else lf.ff_exit_alpha
            self.ff_filtered += a * (ff_target - self.ff_filtered)
            ff = self.ff_filtered

        left_rpm, right_rpm, dbg = self.pid.compute(pv, base_rpm, self.error_sign, feedforward=ff, now=now)
        lv = rpm_to_voltage(left_rpm, lf.rpm_per_volt_left, lf.rpm_volt_offset_left, lf.ao_max_volt)
        rv = rpm_to_voltage(right_rpm, lf.rpm_per_volt_right, lf.rpm_volt_offset_right, lf.ao_max_volt)
        # 12-bit DAC over V_RANGE
        q = lf.ao_range_volt / max(lf.dac_res, 1)
        lv, rv = round(lv / q) * q, round(rv / q) * q
        dbg.update(pv=pv, base_rpm=base_rpm, left_rpm=left_rpm, right_rpm=right_rpm, lv=lv, rv=rv,
                   target=self.current_target_speed)
        self.debug = dbg
        return "drive", lv, rv
