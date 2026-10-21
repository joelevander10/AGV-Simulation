"""Drive loads per wheel (quasi-static rigid-body model).

For every time sample the forces on the AGV body are balanced in the body frame
(origin = drive-axle centre, x forward, y left):

  masses      chassis (CG at cg_x from the load split) + carried lurking cart (at the hook)
              -> inertia incl. centripetal and yaw terms  m*(a_G), I*alpha
  castors     rolling resistance along each castor's own heading, and swivel resistance
              (tyre twist torque / castor trail) while a castor is still turning into the
              new travel direction - this is the classic torque spike at the start of a
              pivot or after a direction change
  drive tyres rolling resistance and tyre twist (bore) torque when yawing
  towed cart  drawbar force along the tongue: cart inertia + rolling resistance + slope
  slope       gravity along the floor slope for AGV and carts

  -> longitudinal force of each drive wheel -> wheel torque -> motor torque (gearbox ratio,
     efficiency) -> motor power / energy; grip usage from longitudinal + lateral tyre force;
     tip-over ratio from the support polygon and CG height.
"""
from __future__ import annotations

import math

import numpy as np

from .config import G

_trapz = getattr(np, "trapezoid", None) or np.trapz


def _ssign(u, eps):
    """Smooth sign (avoids force jumps at zero speed)."""
    return np.tanh(np.asarray(u) / eps)


def _castor_headings(agv, v, w, t, loop=False):
    """Kinematic swivel castor: heading psi (body frame) of each castor wheel over time.

    Kingpin velocity in body frame (vcx, vcy); the wheel trails the kingpin by `trail`:
        dpsi/dt = (-vcx sin psi + vcy cos psi) / trail - w
    """
    out = []
    n = len(t)
    first = np.nonzero(np.abs(v) > 1e-3)[0]
    psi_start = 0.0 if (not len(first) or v[first[0]] > 0) else math.pi
    for c in agv.wheels.castors:
        e = c.trail
        vcx = v - w * c.y
        vcy = w * c.x
        if e <= 1e-4:
            psi = np.arctan2(vcy, vcx)
            psi = np.where(np.hypot(vcx, vcy) > 1e-4, psi, np.nan)
            # hold last direction while standing
            idx = np.where(~np.isnan(psi), np.arange(n), 0)
            np.maximum.accumulate(idx, out=idx)
            psi = np.where(np.isnan(psi[idx]), psi_start, psi[idx])
            out.append(np.unwrap(psi))
            continue
        psi = np.empty(n)
        p0 = psi_start
        for _pass in range(2 if loop else 1):   # a looped cycle starts where it ended
            p = p0
            psi[0] = p
            for i in range(1, n):
                h = t[i] - t[i - 1]
                ax, ay, ww = vcx[i - 1], vcy[i - 1], w[i - 1]
                rate = math.hypot(ax, ay) / e + abs(ww)
                m = max(1, int(math.ceil(rate * h / 0.1)))
                hs = h / m
                for _k in range(m):
                    p += hs * ((-ax * math.sin(p) + ay * math.cos(p)) / e - ww)
                psi[i] = p
            p0 = p
        out.append(psi)
    return out


def support_polygon(agv):
    """Convex hull of the wheel contact points (body frame)."""
    b = agv.wheels.track
    pts = [(0.0, b / 2), (0.0, -b / 2)] + [(c.x, c.y) for c in agv.wheels.castors]
    pts = sorted(set(pts))
    if len(pts) < 3:
        return np.array(pts)

    def cross(o, a, b_):
        return (a[0] - o[0]) * (b_[1] - o[1]) - (a[1] - o[1]) * (b_[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def tip_ratio(ax, ay, cg, h, poly, slope=None):
    """Max over the support-polygon edges of (inertial accel . outward normal * h) / (g cos(slope) * d).

    >= 1 means the vehicle tips over that edge. ax, ay = CG acceleration in body frame.
    slope = floor slope along the body x axis (rad, + = nose up): gravity then pulls along the floor
    like an acceleration of g sin(slope) and presses the wheels down with only g cos(slope)."""
    if slope is not None:
        ax = ax + G * np.sin(slope)
        gz = G * np.cos(slope)
    else:
        gz = G
    if len(poly) < 3 or h <= 0:
        return np.zeros_like(ax)
    best = np.zeros_like(ax)
    k = len(poly)
    for i in range(k):
        p, q = poly[i], poly[(i + 1) % k]
        ex, ey = q[0] - p[0], q[1] - p[1]
        L = math.hypot(ex, ey)
        if L < 1e-9:
            continue
        nx, ny = ey / L, -ex / L                      # outward normal for a CCW polygon
        d = (p[0] - cg[0]) * nx + (p[1] - cg[1]) * ny  # CG distance to the edge (inside > 0)
        if d <= 1e-6:
            return np.full_like(ax, np.inf)
        # the vehicle tips over the edge when the inertial force (-m a) pushes outwards
        best = np.maximum(best, (-(ax * nx + ay * ny)) * h / (gz * d))
    return best


def compute_loads(res) -> dict:
    agv = res.agv
    dtr = agv.drivetrain
    wh = agv.wheels
    ch = agv.chassis
    r = agv.wheel_radius
    b = wh.track
    ratio, eta_g, eta_m = dtr.gear_ratio, dtr.gear_efficiency, dtr.motor_efficiency
    t, v, w, a, th = res.t, res.v, res.w, res.a, res.th
    n = len(t)
    crr = agv.simulation.rolling_resistance
    mu = wh.friction_coeff
    scrub = max(wh.scrub_factor, 0.0)
    ht = agv.hook_type
    att = res.attached if res.attached is not None else np.zeros(n, bool)
    mc = res.cart_mass if res.cart_mass is not None else np.zeros(n)
    grade = res.grade if res.grade is not None else np.zeros(n)
    th_b = np.arctan(grade / 100.0)                     # slope in the AGV forward direction
    lurk = (ht == "lurking") & att
    tow = (ht == "tow") & att

    # ---- filtered yaw acceleration (controller / motor loop response) -------------------------
    tau = max(agv.simulation.yaw_response_time, 1e-6)
    wf = np.empty_like(w)
    wf[0] = w[0]
    for i in range(1, n):
        k = min(1.0, (t[i] - t[i - 1]) / tau)
        wf[i] = wf[i - 1] + k * (w[i] - wf[i - 1])
    alpha = np.gradient(wf, t)

    # ---- masses carried by the AGV --------------------------------------------------------------
    m0 = ch.mass_kg
    xg = agv.cg_x
    bodies = [(np.full(n, m0), xg, 0.0, np.full(n, m0 * (ch.length ** 2 + ch.width ** 2) / 12.0))]
    if ht == "lurking":
        mcl = np.where(lurk, mc, 0.0)
        c = agv.cart
        bodies.append((mcl, agv.hook.x, agv.hook.y, mcl * (c.length ** 2 + c.width ** 2) / 12.0))
    m_agv = sum(bd[0] for bd in bodies)
    Fx_in = np.zeros(n)
    Fy_in = np.zeros(n)
    M_in = np.zeros(n)
    for m, bx, by, I in bodies:
        agx = a - alpha * by - w ** 2 * bx
        agy = v * w + alpha * bx - w ** 2 * by
        Fx_in += m * agx
        Fy_in += m * agy
        M_in += I * alpha + bx * m * agy - by * m * agx

    # ---- external forces on the AGV (body frame) and yaw moments about the axle centre -------
    Wn = m_agv * G * np.cos(th_b)
    f = ch.drive_load_fraction
    N_d = f * Wn / 2.0                                  # per drive wheel
    Ex = -m_agv * G * np.sin(th_b)                      # gravity along the slope
    Ey = np.zeros(n)
    Mz = np.zeros(n)
    castors = wh.castors
    swivel_peak = 0.0
    if castors:
        N_c = (1.0 - f) * Wn / len(castors)
        psis = _castor_headings(agv, v, w, t, bool(res.plan.loop))
        for cs, psi in zip(castors, psis):
            ux, uy = np.cos(psi), np.sin(psi)
            nx, ny = -uy, ux
            vcx, vcy = v - w * cs.y, w * cs.x
            vu = vcx * ux + vcy * uy
            vn = vcx * nx + vcy * ny
            Frr = -crr * N_c * _ssign(vu, 2e-3)
            Fx, Fy = Frr * ux, Frr * uy
            if scrub > 0 and cs.trail > 1e-4:
                wc = vn / cs.trail                       # castor wheel yaw rate (world)
                Mb0 = scrub * mu * N_c * cs.width / 4.0  # tyre twist torque at standstill
                red = 1.0 / (1.0 + np.abs(vu) / (np.abs(wc) * cs.width / 2.0 + 1e-9))
                Fsw = -(Mb0 * red / cs.trail) * _ssign(vn, 2e-3)
                Fx, Fy = Fx + Fsw * nx, Fy + Fsw * ny
                swivel_peak = max(swivel_peak, float(np.max(np.abs(Fsw))))
            Ex += Fx
            Ey += Fy
            Mz += cs.x * Fy - cs.y * Fx
    else:
        Ex += -crr * (1.0 - f) * Wn * _ssign(v, 2e-3)
    vl = v - w * b / 2
    vr = v + w * b / 2
    if scrub > 0:
        Mb_d = scrub * mu * N_d * wh.width / 4.0
        for vw in (vl, vr):
            red = 1.0 / (1.0 + np.abs(vw) / (np.abs(w) * wh.width / 2.0 + 1e-9))
            Mz += -Mb_d * red * _ssign(w, 2e-3)

    drawbar = np.zeros(n)
    cart_lat = None
    if ht == "tow" and res.cart_x is not None and tow.any():
        # trolley train: walk from the last trolley to the first; each drawbar carries its own
        # trolley (inertia, rolling resistance, slope) plus the pull of the drawbar behind it
        # projected onto its axis (the fixed axle takes the sideways part).
        edges = np.diff(np.concatenate([[0], tow.astype(int), [0]]))
        ivs = [(i0, i1) for i0, i1 in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]) if i1 - i0 >= 3]
        train = res.train or [{"x": res.cart_x, "y": res.cart_y, "th": res.cart_th, "m": 1.0}]
        kin = []
        # speed of each trolley axle = speed of the point it hangs on, along the trolley (rigid tongue,
        # the fixed axle only rolls forward). The first hangs on the AGV hitch (from the AGV's v, w),
        # every next one on the rear coupling of the one before. One derivative less than
        # differentiating the trolley positions twice -> no amplified jitter in the drawbar force.
        c, s_ = np.cos(th), np.sin(th)
        hx, hy = agv.hook.x, agv.hook.y
        rwx, rwy = hx * c - hy * s_, hx * s_ + hy * c
        pvx, pvy = v * c - w * rwy, v * s_ + w * rwx          # hitch velocity (world)
        cpl = agv.cart.coupling_behind_axle
        for tr in train:
            psi = np.nan_to_num(tr["th"])
            vc, ac, wc_ = np.zeros(n), np.zeros(n), np.zeros(n)
            for i0, i1 in ivs:
                sl = slice(i0, i1)
                tt = t[sl]
                ps = np.unwrap(psi[sl])
                vc[sl] = pvx[sl] * np.cos(ps) + pvy[sl] * np.sin(ps)
                ac[sl] = np.gradient(vc[sl], tt)
                wc_[sl] = np.gradient(ps, tt)
            kin.append((psi, vc, ac, wc_))
            # the next trolley hangs on this one's rear coupling, cpl behind its axle
            cp, sp = np.cos(psi), np.sin(psi)
            pvx, pvy = vc * cp + wc_ * cpl * sp, vc * sp - wc_ * cpl * cp
        F_next = np.zeros(n)
        psi_next = None
        for k in range(len(train) - 1, -1, -1):
            psi, vc, ac, _ = kin[k]
            mk = np.where(tow, mc * train[k]["m"], 0.0)
            Fk = mk * ac + agv.cart.rolling_resistance * mk * G * _ssign(vc, 2e-3) + mk * G * np.sin(th_b)
            if psi_next is not None:
                Fk = Fk + F_next * np.cos(psi_next - psi)
            F_next, psi_next = np.where(tow, Fk, 0.0), psi
        drawbar = F_next
        psi = kin[0][0]
        # force on the AGV at the hitch, pointing from the hitch towards the first trolley axle
        Fwx, Fwy = -drawbar * np.cos(psi), -drawbar * np.sin(psi)
        Fbx = Fwx * np.cos(th) + Fwy * np.sin(th)
        Fby = -Fwx * np.sin(th) + Fwy * np.cos(th)
        Ex += Fbx
        Ey += Fby
        Mz += agv.hook.x * Fby - agv.hook.y * Fbx
        cart_lat = np.max([np.where(tow, np.abs(vc * wc_), 0.0) for (_, vc, _, wc_) in kin], axis=0)

    # ---- drive wheel forces --------------------------------------------------------------------
    Fsum = Fx_in - Ex                                   # F_left + F_right at the tyres
    Fdiff = 2.0 * (M_in - Mz) / b                        # F_right - F_left
    Fy_d = (Fy_in - Ey) / 2.0                            # lateral force per drive tyre
    Fl = Fsum / 2 - Fdiff / 2 + crr * N_d * _ssign(vl, 2e-3)
    Fr = Fsum / 2 + Fdiff / 2 + crr * N_d * _ssign(vr, 2e-3)

    out = {"m_agv_effective": m_agv, "drawbar_force": drawbar, "castor_swivel_force_peak": swivel_peak}
    P_total = np.full_like(t, dtr.aux_power_w)
    for side, F, vw in (("l", Fl, vl), ("r", Fr, vr)):
        Tw = F * r
        motoring = Tw * vw >= 0
        Tm = np.where(motoring, Tw / (ratio * eta_g), Tw * eta_g / ratio)
        om = vw / r * ratio                                  # motor rad/s
        P_mech = Tm * om
        P_el = np.where(P_mech > 0, P_mech / eta_m, 0.0)     # no regen assumed
        P_total = P_total + P_el
        out[f"force_{side}"] = F
        out[f"wheel_speed_{side}"] = vw
        out[f"wheel_rpm_{side}"] = vw / (2 * math.pi * r) * 60
        out[f"motor_rpm_{side}"] = out[f"wheel_rpm_{side}"] * ratio
        out[f"motor_torque_{side}"] = Tm
        out[f"motor_power_{side}"] = P_el
        out[f"traction_{side}"] = np.hypot(F, Fy_d) / np.maximum(mu * N_d, 1e-9)
    out["power_total"] = P_total
    out["energy_wh"] = float(_trapz(P_total, t) / 3600.0)

    # ---- tip-over ------------------------------------------------------------------------------
    poly = support_polygon(agv)
    msum = np.maximum(m_agv, 1e-9)
    cgx = sum(bd[0] * bd[1] for bd in bodies) / msum
    cgy = sum(bd[0] * bd[2] for bd in bodies) / msum
    hz = (m0 * ch.cg_height + (np.where(lurk, mc, 0.0) * (agv.cart.cg_height + agv.hook.lift_stroke)
                               if ht == "lurking" else 0.0)) / msum
    ag_x = a - w ** 2 * cgx - alpha * cgy
    ag_y = v * w + alpha * cgx - w ** 2 * cgy
    tr = np.zeros(n)
    # evaluate with the time-varying CG in a few groups (CG only changes with the cart state)
    keys = np.round(np.column_stack([cgx, cgy, hz]), 4)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    for ui, (gx, gy, gh) in enumerate(uniq):
        sel = inv == ui
        tr[sel] = tip_ratio(ag_x[sel], ag_y[sel], (gx, gy), gh, poly, slope=th_b[sel])
    out["tip_ratio"] = tr
    out["lat_accel"] = np.abs(v * w)
    if cart_lat is not None:
        c = agv.cart
        out["cart_lat_accel"] = cart_lat
        out["cart_tip_ratio"] = cart_lat * c.cg_height / (G * max(c.wheel_track / 2, 1e-6))
    return out
