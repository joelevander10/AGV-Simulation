"""Longitudinal drive limits: how fast the AGV can speed up, brake and climb with what it pulls.

Used by the closed-loop plant (linefollow.py) and the kinematic profile (simulate.py), so the speed
with trolleys differs from the speed without:

  mass         M = AGV (+ lurking cart) + every towed trolley (all move at the AGV speed)
  resistance   AGV rolling (simulation.rolling_resistance) + trolley rolling (cart.rolling_resistance)
               + M g sin(slope)
  drive force  min(motor: 2 x peak torque x gear x gear efficiency / wheel radius,
                   grip:  friction_coeff x load on the two drive wheels)
               The drive wheels only carry the AGV (+ a lurking cart): towed trolleys add mass and
               drag but no grip - with light AGVs grip is usually the limit.
  brake        line_follow.brake_decel is the deceleration of the bare AGV on flat floor; its brake
               force (limited by grip) now also has to stop the trolleys, which have no brakes.
"""
from __future__ import annotations

import math

G = 9.81


class DriveLimits:
    def __init__(self, agv):
        self.agv = agv
        d, wh, ch = agv.drivetrain, agv.wheels, agv.chassis
        r = max(wh.diameter / 2, 1e-6)
        self.F_motor = 2.0 * d.peak_torque * d.gear_ratio * d.gear_efficiency / r
        self.F_motor_rated = 2.0 * d.rated_torque * d.gear_ratio * d.gear_efficiency / r
        self.mu = wh.friction_coeff
        self.f_drive = ch.drive_load_fraction
        self.m0 = ch.mass_kg
        self.crr = agv.simulation.rolling_resistance
        self.crr_cart = agv.cart.rolling_resistance
        lf = agv.line_follow
        self.brake_decel = lf.brake_decel if lf is not None else agv.performance.decel
        self.track = max(wh.track, 1e-3)
        self.F_motor_wheel = self.F_motor / 2.0
        self.I_agv = ch.mass_kg * (ch.length ** 2 + ch.width ** 2) / 12.0

    def state(self, m_tow=0.0, m_lurk=0.0, grade_pct=0.0):
        """Forces for a load state. Returns a dict (N, kg, m/s2)."""
        th = math.atan(grade_pct / 100.0)
        m_agv = self.m0 + m_lurk
        M = m_agv + m_tow
        grip = self.mu * self.f_drive * m_agv * G * math.cos(th)
        F_drive = min(self.F_motor, grip)
        F_roll = (self.crr * m_agv + self.crr_cart * m_tow) * G * math.cos(th)
        F_slope = M * G * math.sin(th)                  # + uphill (against the motion forward)
        # brake: bare AGV on flat floor stops at brake_decel (incl. its own rolling resistance)
        F_brake = min(self.brake_decel * m_agv, grip + self.crr * m_agv * G)
        return {"grade": grade_pct, "M": M, "m_agv": m_agv, "m_tow": m_tow, "grip": grip, "F_drive": F_drive,
                "limit": "grip" if grip < self.F_motor else "motor torque",
                "F_roll": F_roll, "F_slope": F_slope, "F_brake": F_brake,
                "a_max": (F_drive - F_roll - F_slope) / M,             # speeding up, forward
                "d_max": (F_drive + F_roll + F_slope) / M,             # slowing down with the motors
                "brake": (F_brake + (F_roll - self.crr * m_agv * G * math.cos(th)) + F_slope) / M,
                "cruise_force": F_roll + F_slope,
                "rated_force": self.F_motor_rated}

    def brake_mag(self, st, v):
        """Deceleration (m/s2, towards standstill) with the brake on; < 0 = the train pushes the AGV
        faster (steep downhill)."""
        F_tr_roll = st["F_roll"] - self.crr * st["m_agv"] * G * math.cos(math.atan(st.get("grade", 0.0) / 100.0))
        sgn = 1.0 if v >= 0 else -1.0
        return (st["F_brake"] + F_tr_roll + st["F_slope"] * sgn) / st["M"]

    def step(self, st, vl, vr, al_dem, ar_dem, dt, direction=1.0):
        """Advance wheel speeds one step. al/ar_dem: accelerations the motor drivers ask for.

        Per drive wheel: longitudinal force = half of (M x common accel + resistance) -/+ the force that
        turns the AGV (yaw inertia). Each wheel is limited by its motor and by its grip, and the grip
        left for driving shrinks with the sideways force of cornering (friction circle). The common
        part is cut first (steering keeps priority), then the steering part if one wheel is still
        over its limit. Returns (vl, vr, limited)."""
        M = st["M"]
        v = 0.5 * (vl + vr)
        w = (vr - vl) / self.track
        a_c = 0.5 * (al_dem + ar_dem)
        a_d = 0.5 * (ar_dem - al_dem)
        # resistance in the direction of travel (smooth sign at standstill); slope sign follows
        # the heading: grade is + uphill for forward driving (already multiplied by direction)
        sgn = max(-1.0, min(1.0, v / 2e-3))
        F_res = st["F_roll"] * sgn + st["F_slope"]
        F_c = M * a_c + F_res                                   # both wheels together
        F_d = self.I_agv * (2.0 * a_d / self.track) / self.track   # per wheel, +right / -left
        grip_w = 0.5 * st["grip"]
        lat_w = 0.5 * st["m_agv"] * abs(v * w)                  # sideways force per drive wheel
        lim_w = min(self.F_motor_wheel, math.sqrt(max(grip_w * grip_w - lat_w * lat_w, 0.0)))
        limited = False
        if abs(F_d) > lim_w:                                    # even turning alone is too much
            F_d = math.copysign(lim_w, F_d)
            limited = True
        room = lim_w - abs(F_d)                                 # what each wheel has left for driving
        if abs(F_c) > 2.0 * room + 1e-9:
            F_c = math.copysign(2.0 * room, F_c)
            limited = True
        a = (F_c - F_res) / M
        if abs(v) < 1e-3 and abs(F_c - st["F_slope"]) <= st["F_roll"] and abs(a_c) < 1e-9:
            a = 0.0
        a_d = F_d * self.track * self.track / (2.0 * self.I_agv) if self.I_agv > 0 else a_d
        vl += (a - a_d) * dt
        vr += (a + a_d) * dt
        return vl, vr, limited


def load_table(agv):
    """Speed-up / brake / climb with 0 .. n trolleys (tow) or with / without the cart (lurking), flat floor."""
    drv = DriveLimits(agv)
    lf = agv.line_follow
    closed = bool(lf is not None and lf.enabled)
    v_top = min(lf.speed_high, agv.max_wheel_speed) if closed else agv.performance.v_straight
    asked = lf.accel_rate if closed else agv.performance.accel
    asked_dec = lf.decel_rate if closed else agv.performance.decel
    ht = agv.hook_type
    states = [("AGV only", 0, 0.0, 0.0)]
    if ht == "tow":
        ms = agv.train_masses()
        for k in range(1, len(ms) + 1):
            states.append((f"{k} trolley{'s' if k > 1 else ''}", k, sum(ms[:k]), 0.0))
    elif ht == "lurking":
        states.append(("with cart", 1, 0.0, agv.cart.mass_kg))
    rows = []
    for label, k, m_tow, m_lurk in states:
        st = drv.state(m_tow, m_lurk, 0.0)
        a = min(asked, st["a_max"]) if st["a_max"] > 0 else 0.0
        b = drv.brake_mag(st, 1.0)
        # steepest slope it can still start on: F_drive = F_roll + M g sin
        x = (st["F_drive"] - st["F_roll"]) / (st["M"] * G)
        slope = math.tan(math.asin(max(-1.0, min(1.0, x)))) * 100 if x > 0 else 0.0
        rows.append({"label": label, "trolleys": k, "mass_kg": st["M"], "load_kg": m_tow + m_lurk,
                     "accel": a, "accel_asked": asked, "limit": st["limit"] if st["a_max"] < asked else "controller ramp",
                     "decel": min(asked_dec, st["d_max"]), "brake": b,
                     "t_to_top": v_top / a if a > 0 else math.inf,
                     "d_to_top": v_top ** 2 / (2 * a) if a > 0 else math.inf,
                     "brake_dist": v_top ** 2 / (2 * b) if b > 0 else math.inf,
                     "max_slope_pct": slope, "v_top": v_top,
                     "drive_force": st["F_drive"], "roll_force": st["F_roll"]})
    return rows
