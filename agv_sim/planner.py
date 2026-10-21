"""Speed profiles along a path and for pivots.

Two profile generators:
  * trapezoid  (performance.jerk = 0): forward/backward pass, instant acceleration steps
  * S-curve    (performance.jerk > 0): jerk-limited time-domain generator with look-ahead,
               like a real motion controller: acceleration ramps up/down at `jerk`, braking
               starts early enough to reach every lower speed limit ahead (curve, speed zone,
               stop) exactly at its start with zero acceleration.
Both respect a per-sample speed limit and per-sample accel/decel (speed modes).
"""
from __future__ import annotations

import math

import numpy as np

from .config import AGVConfig


def time_grid(T: float, dt: float) -> np.ndarray:
    """0, dt, 2dt, ..., T  (last step stretched instead of creating a tiny sliver)."""
    if T <= 1e-12:
        return np.array([0.0])
    n = int(math.floor(T / dt + 1e-9))
    g = np.arange(n + 1) * dt
    rem = T - g[-1]
    if rem > 1e-9:
        if rem < 0.3 * dt and n >= 1:
            g[-1] = T
        else:
            g = np.append(g, T)
    return g


# ----------------------------------------------------------------------------- limits
def speed_limits(kappa_path, kappa_axle, vmax, vcurve, vcap, agv: AGVConfig) -> np.ndarray:
    """Per-sample speed limit of the drive-axle centre.

    kappa_path : curvature of the drawn line (tape) -> 'curve speed' applies where it is curved
    kappa_axle : curvature the drive axle really drives -> lateral acceleration + outer-wheel rpm
    """
    perf = agv.performance
    if np.any(vmax <= 0):
        raise ValueError("v_straight / v_reverse must be > 0")
    lim = np.minimum(vmax, vcap).astype(float)
    curved = np.abs(kappa_path) > 1e-6
    lim[curved] = np.minimum(lim[curved], np.where(vcurve[curved] > 0, vcurve[curved], np.inf))
    k = np.abs(kappa_axle)
    if perf.max_lateral_accel > 0:
        turning = k > 1e-6
        lim[turning] = np.minimum(lim[turning], np.sqrt(perf.max_lateral_accel / k[turning]))
    # the OUTER wheel turns faster than the centre: v_outer = v (1 + |k| b/2) <= wheel max
    lim = np.minimum(lim, agv.max_wheel_speed / (1.0 + k * agv.wheels.track / 2))
    return np.maximum(lim, 1e-3)


# ----------------------------------------------------------------------------- trapezoid
def _trapezoid(s, lim, acc, dec):
    v = lim.astype(float).copy()
    ds = np.diff(s)
    v[0] = 0.0
    for i in range(1, len(v)):
        v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2 * acc[i - 1] * ds[i - 1]))
    v[-1] = 0.0
    for i in range(len(v) - 2, -1, -1):
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * dec[i] * ds[i]))
    vavg = 0.5 * (v[:-1] + v[1:])
    dt = np.where(vavg > 1e-9, ds / np.maximum(vavg, 1e-12), 0.0)
    t = np.concatenate([[0.0], np.cumsum(dt)])
    return v, t


def s_at_times(tg, tt, s, v):
    """Exact s(t) for a profile with constant acceleration between samples (no staircase)."""
    i = np.clip(np.searchsorted(tt, tg, side="right") - 1, 0, len(tt) - 2)
    span = np.maximum(tt[i + 1] - tt[i], 1e-12)
    tau = np.clip(tg - tt[i], 0.0, span)
    acc = (v[i + 1] - v[i]) / span
    out = s[i] + v[i] * tau + 0.5 * acc * tau ** 2
    return np.minimum(out, s[-1])


# ----------------------------------------------------------------------------- S-curve
def brake_distance(v0: float, a0: float, vt: float, D: float, J: float) -> float:
    """Shortest distance to get from speed v0 / acceleration a0 down to speed vt with zero
    acceleration, using max deceleration D and jerk J. 0 if the speed never exceeds vt."""
    d0 = 0.0
    if a0 > 0.0:                      # first bring the acceleration back to zero
        t0 = a0 / J
        d0 = v0 * t0 + 0.5 * a0 * t0 * t0 - J * t0 ** 3 / 6.0
        v0 = v0 + a0 * a0 / (2.0 * J)
        a1 = 0.0
    else:
        a1 = min(-a0, D)
    if v0 - a1 * a1 / (2.0 * J) <= vt + 1e-9:
        return 0.0
    dv = v0 - vt
    th = (dv - (D * D - a1 * a1) / (2.0 * J) - D * D / (2.0 * J)) / D
    if th >= 0.0:
        ap = D
    else:
        ap = math.sqrt(max((2.0 * J * dv + a1 * a1) / 2.0, a1 * a1))
        th = 0.0
    t1 = (ap - a1) / J
    d1 = v0 * t1 - 0.5 * a1 * t1 * t1 - J * t1 ** 3 / 6.0
    v1 = v0 - a1 * t1 - 0.5 * J * t1 * t1
    d2 = v1 * th - 0.5 * ap * th * th
    v2 = v1 - ap * th
    t3 = ap / J
    d3 = v2 * t3 - 0.5 * ap * t3 * t3 + J * t3 ** 3 / 6.0
    return d0 + d1 + d2 + d3


def _segments(s, lim, acc, dec, q=0.002):
    """Compress the per-sample limit into piecewise-constant zones (conservative)."""
    lq = np.floor(lim / q + 1e-6) * q
    lq = np.maximum(lq, q)
    ch = np.nonzero(np.diff(lq) != 0)[0] + 1
    starts = np.concatenate([[0], ch])
    ends = np.concatenate([ch, [len(s)]])
    seg_s = s[starts].astype(float)
    seg_v = np.array([lq[a:b].min() for a, b in zip(starts, ends)])
    seg_a = np.array([acc[a:b].min() for a, b in zip(starts, ends)])
    seg_d = np.array([dec[a:b].min() for a, b in zip(starts, ends)])
    return seg_s, seg_v, seg_a, seg_d


def _scurve(s, lim, acc, dec, J, h):
    """Jerk-limited profile from standstill to standstill. Returns (t, s, v) at internal steps h."""
    S_end = float(s[-1])
    seg_s, seg_v, seg_a, seg_d = _segments(s, lim, acc, dec)
    # the end of the run is a zone with limit 0
    cs = np.append(seg_s, S_end)
    cv = np.append(seg_v, 0.0)
    n_c = len(cs)
    x = v = a = t = 0.0
    T, X, V = [0.0], [0.0], [0.0]
    j = 0
    t_max = 50.0 + 20.0 * S_end / max(float(lim.min()), 0.02)

    def feasible(a1, D):
        v1 = v + a1 * h
        if v1 < 0.0:
            v1 = 0.0
        x1 = x + 0.5 * (v + v1) * h
        jj = int(np.searchsorted(cs, x1, side="right")) - 1
        jj = max(jj, 0)
        look = brake_distance(v1, a1, 0.0, D, J) + 1e-3
        for k in range(jj, n_c):
            dist = cs[k] - x1 if k > jj else 0.0
            if dist > look:
                break
            if cv[k] < v1 + max(a1, 0.0) ** 2 / (2 * J) + 1e-9 or k == jj:
                if brake_distance(v1, a1, cv[k], D, J) > dist + 1e-7:
                    return False
        return True

    while t < t_max:
        rem = S_end - x
        if rem < 5e-4 and (v > 1e-4 and v * v / (2 * rem + 1e-12) <= max(seg_d[-1], 1e-3) or v <= 1e-4):
            # last half millimetre: finish with a gentle uniform deceleration (no speed step)
            if v > 1e-4:
                tau = 2.0 * rem / v
                for f in (0.25, 0.5, 0.75, 1.0):
                    tt = f * tau
                    T.append(t + tt)
                    X.append(x + v * tt - 0.5 * (v / tau) * tt * tt)
                    V.append(v - (v / tau) * tt)
            else:
                T.append(t + h)
                X.append(S_end)
                V.append(0.0)
            X[-1] = S_end
            break
        while j + 1 < len(seg_s) and seg_s[j + 1] <= x:
            j += 1
        A, D = seg_a[j], seg_d[j]
        hi = min(A, a + J * h)
        lo = max(-D, a - J * h)
        if feasible(hi, D):
            a1 = hi
        elif not feasible(lo, D):
            a1 = lo
        else:
            f_ok, f_bad = lo, hi
            if lo < 0.0 < hi and feasible(0.0, D):
                f_ok = 0.0
            for _ in range(6):
                mid = 0.5 * (f_ok + f_bad)
                if feasible(mid, D):
                    f_ok = mid
                else:
                    f_bad = mid
            a1 = f_ok
        v1 = v + a1 * h
        if v1 < 0.0:
            a1 = -v / h
            v1 = 0.0
        x1 = min(x + 0.5 * (v + v1) * h, S_end)
        t += h
        x, v, a = x1, v1, a1
        T.append(t)
        X.append(x)
        V.append(v)
    else:
        raise RuntimeError("S-curve profile did not reach the end of the run (check speeds / jerk).")
    return np.array(T), np.array(X), np.array(V)


# ----------------------------------------------------------------------------- public
def motion_profile(s, lim, acc, dec, jerk: float, dt: float):
    """Standstill-to-standstill motion over arc length s with per-sample limits.

    Returns (tg, sg): time grid (step dt) and the position at each time.
    """
    s = np.asarray(s, float)
    if len(s) < 2 or s[-1] <= 1e-9:
        return np.array([0.0]), np.array([0.0])
    acc = np.broadcast_to(np.asarray(acc, float), s.shape)
    dec = np.broadcast_to(np.asarray(dec, float), s.shape)
    if np.any(acc <= 0) or np.any(dec <= 0):
        raise ValueError("accel / decel must be > 0")
    if jerk and jerk > 0:
        h = min(dt, 0.02) / 2
        T, X, _ = _scurve(s, np.asarray(lim, float), acc, dec, float(jerk), h)
        tg = time_grid(float(T[-1]), dt)
        return tg, np.interp(tg, T, X)
    v, tt = _trapezoid(s, np.asarray(lim, float), acc, dec)
    tg = time_grid(float(tt[-1]), dt)
    return tg, s_at_times(tg, tt, s, v)


def pivot_profile(angle: float, omega: float, alpha: float, dt: float, jerk: float = 0.0):
    """Spin in place by |angle| rad with max rate omega, accel alpha (and optional jerk).

    Returns (t, angle(t)) with angle(t) going 0 -> |angle|."""
    A = abs(angle)
    if A < 1e-9:
        return np.array([0.0]), np.array([0.0])
    if jerk and jerk > 0:
        n = max(20, int(A / 0.002))
        sgrid = np.linspace(0.0, A, n)
        return motion_profile(sgrid, np.full(n, omega), alpha, alpha, jerk, dt)
    if A >= omega ** 2 / alpha:
        ta = omega / alpha
        tc = (A - omega * ta) / omega
        wp = omega
    else:
        ta = math.sqrt(A / alpha)
        tc = 0.0
        wp = alpha * ta
    T = 2 * ta + tc
    t = time_grid(T, dt)
    ang = np.where(t < ta, 0.5 * alpha * t ** 2,
                   np.where(t < ta + tc, 0.5 * alpha * ta ** 2 + wp * (t - ta),
                            A - 0.5 * alpha * (T - t) ** 2))
    return t, np.clip(ang, 0, A)
