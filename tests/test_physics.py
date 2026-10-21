"""Physics sanity checks. Run:  python -m pytest tests   (or  python tests/test_physics.py)
Quick run without the slow auto-route / auto-tune tests:  python -m pytest tests -m "not slow"
(or  python tests/test_physics.py --fast)."""
import copy
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agv_sim import run_simulation  # noqa: E402
from agv_sim.config import G, AGVConfig, Castor, MapConfig, RouteConfig, Waypoint, load_agv  # noqa: E402
from agv_sim.path import build_plan  # noqa: E402
from agv_sim.planner import brake_distance, motion_profile  # noqa: E402
from agv_sim.simulate import simulate  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")   # test vehicles / map (not your configs)

try:                                    # pytest marker when pytest is there, a plain flag otherwise
    import pytest
    slow = pytest.mark.slow
except ImportError:                     # pragma: no cover
    def slow(fn):
        fn.slow = True
        return fn


def straight_route(L=20.0, grade=0.0, loop=False):
    return RouteConfig(name="straight", map=MapConfig(), loop=loop, default_radius=1.0,
                       waypoints=[Waypoint(x=2, y=5, action="stop", grade=grade), Waypoint(x=2 + L, y=5, action="stop")])


def simple_agv(**kw):
    a = AGVConfig()
    a.wheels.castors = [Castor(x=1.0, y=0.15, trail=0.0), Castor(x=1.0, y=-0.15, trail=0.0)]
    a.wheels.scrub_factor = 0.0
    for k, v in kw.items():
        obj = a
        *path, last = k.split(".")
        for p in path:
            obj = getattr(obj, p)
        setattr(obj, last, v)
    return a


def test_trapezoid_time_matches_formula():
    a = simple_agv(**{"performance.v_straight": 0.8, "performance.accel": 0.4, "performance.decel": 0.2})
    res = simulate(a, straight_route(), build_plan(straight_route(), a))
    v, A, D, L = 0.8, 0.4, 0.2, 20.0
    T = v / A + v / D + (L - v * v / (2 * A) - v * v / (2 * D)) / v
    assert abs(res.t[-1] - T) < 0.05, (res.t[-1], T)


def test_scurve_time_matches_formula():
    a = simple_agv(**{"performance.v_straight": 0.8, "performance.accel": 0.4, "performance.decel": 0.4,
                      "performance.jerk": 0.8})
    res = simulate(a, straight_route(), build_plan(straight_route(), a))
    v, A, J, L = 0.8, 0.4, 0.8, 20.0
    t_ramp = v / A + A / J               # jerk-limited ramp 0 -> v
    d_ramp = v / 2 * t_ramp
    T = 2 * t_ramp + (L - 2 * d_ramp) / v
    assert abs(res.t[-1] - T) < 0.1, (res.t[-1], T)
    assert np.max(np.abs(res.a)) <= A + 0.02
    jerk = np.gradient(res.a, res.t)
    assert np.percentile(np.abs(jerk), 99) <= J * 1.3


def test_brake_distance_formula():
    # v=1 -> 0 with D=0.5, J=1: T = 1/0.5 + 0.5/1 = 2.5 s, d = 0.5 * 2.5
    assert abs(brake_distance(1.0, 0.0, 0.0, 0.5, 1.0) - 1.25) < 1e-9
    assert brake_distance(0.3, 0.0, 0.5, 0.5, 1.0) == 0.0


def test_profile_respects_speed_zone():
    s = np.linspace(0, 20, 2001)
    lim = np.full_like(s, 0.8)
    lim[900:1200] = 0.3
    for jerk in (0.0, 0.6):
        tg, sg = motion_profile(s, lim, 0.3, 0.3, jerk, 0.01)
        v = np.gradient(sg, tg)
        inside = (sg >= 9.02) & (sg <= 11.97)
        assert v[inside].max() <= 0.3 + 0.01


def test_cruise_torque_is_rolling_resistance():
    a = simple_agv()
    route = straight_route(30)
    res, s = run_simulation(a, route)
    L = res.loads
    cruise = (np.abs(res.a) < 1e-6) & (res.v > 0.5)
    F = L["force_l"][cruise] + L["force_r"][cruise]
    expected = a.simulation.rolling_resistance * a.chassis.mass_kg * G
    assert np.allclose(F, expected, rtol=0.02), (F.mean(), expected)


def test_grade_adds_m_g_sin():
    a = simple_agv()
    r0, r1 = straight_route(30, 0.0), straight_route(30, 5.0)
    L0 = run_simulation(a, r0)[0]
    res1 = run_simulation(a, r1)[0]
    c0 = (np.abs(L0.a) < 1e-6) & (L0.v > 0.5)
    c1 = (np.abs(res1.a) < 1e-6) & (res1.v > 0.5)
    F0 = (L0.loads["force_l"] + L0.loads["force_r"])[c0].mean()
    F1 = (res1.loads["force_l"] + res1.loads["force_r"])[c1].mean()
    th = math.atan(0.05)
    expected = a.chassis.mass_kg * G * (math.sin(th) + a.simulation.rolling_resistance * (math.cos(th) - 1))
    assert abs((F1 - F0) - expected) < 1.0, (F1 - F0, expected)


def test_tow_cart_adds_drawbar():
    a = load_agv(os.path.join(DATA, "tow_tractor_example.yaml"))
    a.wheels.scrub_factor = 0.0
    route = straight_route(30)
    res = run_simulation(a, route)[0]
    cruise = (np.abs(res.a) < 1e-6) & (res.v > 0.5)
    db = res.loads["drawbar_force"][cruise].mean()
    assert abs(db - a.cart.rolling_resistance * a.cart.mass_kg * G) < 1.0


def test_modes_and_cart_events():
    a = load_agv(os.path.join(DATA, "tow_tractor_example.yaml"))
    route = straight_route(20)
    route.waypoints.insert(1, Waypoint(x=12, y=5, name="drop", cart="detach", mode="EXTRA_SLOW"))
    res, s = run_simulation(a, route)
    assert res.attached[0] and not res.attached[-1]
    assert np.nanmax(res.v[res.mode == 1][-50:]) <= 0.15 + 1e-3     # EXTRA_SLOW after the drop
    assert len(res.parked) == 1
    assert any(e.label.startswith("Detach cart") and e.label.endswith("@ drop") for e in res.events)


def test_tape_axle_cuts_inside():
    a = load_agv(os.path.join(DATA, "tow_tractor_example.yaml"))
    a.hook.type = "none"
    route = RouteConfig(name="L", loop=False, default_radius=2.0,
                        waypoints=[Waypoint(x=2, y=2, action="stop"), Waypoint(x=12, y=2),
                                   Waypoint(x=12, y=12, action="stop")])
    res, s = run_simulation(a, route)
    arc = s["arcs"][0]
    assert arc["axle_radius_min"] < arc["path_radius"]


def test_castor_swivel_spike_on_pivot():
    a = load_agv(os.path.join(DATA, "lurking_amr_example.yaml"))
    route = RouteConfig(name="pivot", loop=False, default_radius=0.0,
                        waypoints=[Waypoint(x=2, y=2, action="stop"), Waypoint(x=8, y=2, radius=0),
                                   Waypoint(x=8, y=8, action="stop")])
    with_s = run_simulation(a, route)[0]
    b = copy.deepcopy(a)
    b.wheels.scrub_factor = 0.0
    without = run_simulation(b, route)[0]
    piv = with_s.mode == 2
    tq = lambda r: np.maximum(np.abs(r.loads["motor_torque_l"]), np.abs(r.loads["motor_torque_r"]))  # noqa: E731
    assert tq(with_s)[piv].max() > 1.5 * tq(without)[without.mode == 2].max()


# ----------------------------------------------------------------------------- closed-loop tape following
def _agv2_lf(**sets):
    from agv_sim.config import set_param
    a = load_agv(os.path.join(DATA, "agv2.yaml"))
    a.hook.type = "none"
    a.line_follow.sensor_noise_mm = 0.0
    for k, v in sets.items():
        set_param(a, k, v)
    return a


def _tape_route(radius, mode="SLOW"):
    return RouteConfig(name="tape", loop=False, default_radius=radius, default_mode=mode,
                       waypoints=[Waypoint(x=2, y=2, action="stop"), Waypoint(x=10, y=2),
                                  Waypoint(x=10, y=10, action="stop")])


def test_pid_port_matches_original_formula():
    from agv_sim.kayaba_pid import PIDController
    pid = PIDController(kp=2.0, td=0.1, n=10, ti=20, ti_deadband=25, ti_max=500, dt=0.01, output_clamp=800,
                        v_red_coef=2, sr_alpha=0.15, sr_cap=0.55)
    l, r, d = pid.compute(10.0, 1000.0, error_sign=1.0, now=0.0)
    # P = 20, I = 2/20*0.1 = 0.01, D: de = 1000 mm/s, alpha = 0.01/(0.01+0.01) = 0.5 -> 0.5*2*0.1*1000 = 100
    assert abs(d["p"] - 20) < 1e-9 and abs(d["i"] - 0.01) < 1e-9 and abs(d["d"] - 100) < 1e-9
    assert abs(d["output"] - 120.01) < 1e-9
    assert abs((l - r) - 2 * 120.01) < 1e-9


def test_straight_tape_stays_centred():
    a = _agv2_lf()
    res, s = run_simulation(a, straight_route(12))
    assert not res.lf["aborted"]
    assert s["line_follow"]["max_abs_mm"] <= 2.0


def test_wrong_error_sign_loses_tape():
    a = _agv2_lf(**{"line_follow.error_sign_forward": -1, "line_follow.sensor_noise_mm": 0.5})
    res, s = run_simulation(a, _tape_route(1.5))
    assert res.lf["aborted"]


def test_too_fast_in_curve_loses_tape():
    ok = run_simulation(_agv2_lf(), _tape_route(1.5))[0]
    fast = run_simulation(_agv2_lf(**{"line_follow.speed_slow": 0.5}), _tape_route(1.5))[0]
    assert not ok.lf["aborted"] and fast.lf["aborted"]


def test_stop_overshoot_is_braking_distance():
    a = _agv2_lf(**{"line_follow.start_mode": "HIGH"})
    route = straight_route(15)
    route.default_mode = "HIGH"
    res, s = run_simulation(a, route)
    ov = s["line_follow"]["stops"][0]["stop_overshoot"]
    v = res.lf["tv"].max()
    expected = v * v / (2 * a.line_follow.brake_decel)
    assert abs(ov - expected) < 0.05, (ov, expected)


def test_extra_mode_from_table_is_used():
    from agv_sim.config import SpeedMode
    a = _agv2_lf()
    a.speed_modes.append(SpeedMode(name="MED", v_straight=0.6, v_curve=0.6, v_reverse=0.2, accel=0.3, decel=0.3))
    route = straight_route(25)
    route.default_mode = "MED"
    res, s = run_simulation(a, route)
    med = [r for r in s["speed_by_mode"] if r["mode"] == "MED"][0]
    assert abs(med["max"] - 0.6) < 0.02, med


def test_trolley_train_drawbar_adds_up():
    a = load_agv(os.path.join(DATA, "tow_tractor_example.yaml"))
    a.wheels.scrub_factor = 0.0
    a.cart.count = 3
    a.cart.masses = [100.0, 50.0, 30.0]
    res, s = run_simulation(a, straight_route(40))
    cruise = (np.abs(res.a) < 1e-6) & (res.v > 0.5)
    db = res.loads["drawbar_force"][cruise].mean()
    assert abs(db - a.cart.rolling_resistance * 180.0 * G) < 1.5, db
    assert len(res.train) == 3 and s["train"]["total_kg"] == 180.0


def test_loop_average_speed():
    a = load_agv(os.path.join(DATA, "tow_tractor_example.yaml"))
    route = RouteConfig(name="sq", loop=True, default_radius=1.5,
                        waypoints=[Waypoint(x=3, y=3, action="stop", dwell=5), Waypoint(x=13, y=3),
                                   Waypoint(x=13, y=9), Waypoint(x=3, y=9)])
    res, s = run_simulation(a, route)
    lp = s["laps"]
    assert len(lp) == 1 and lp[0]["closed"]
    assert abs(lp[0]["avg_speed"] - lp[0]["distance_m"] / s["cycle_time_s"]) < 1e-6


@slow
def test_autoroute_plans_clean_route():
    """Plan mode: stations only -> orthogonal lane, no collisions, tape kept, stops preserved."""
    from agv_sim.autoroute import auto_route
    from agv_sim.factory_map import FactoryMap
    here = os.path.join(os.path.dirname(__file__), "..")
    agv = load_agv(os.path.join(DATA, "tow_tractor_example.yaml"))
    agv.hook.type = "none"
    r = RouteConfig(name="t", loop=True,
                    map=MapConfig(image=os.path.join(DATA, "demo_factory.png"), meters_per_pixel=0.02),
                    waypoints=[Waypoint(x=6.0, y=3.0, name="Home", action="stop", dwell=5),
                               Waypoint(x=12.0, y=7.75, name="A", action="stop", dwell=5)])
    out = auto_route(agv, r, FactoryMap.for_route(r), mode="plan", max_iter=4)
    s = out["summary"]
    assert not s["collisions"], s["collisions"]
    names = [w.name for w in out["route"].waypoints if w.name]
    assert names == ["Home", "A"], names
    legs = out["route"].waypoints
    for a, b in zip(legs, legs[1:]):             # orthogonal lanes
        assert abs(a.x - b.x) < 1e-3 or abs(a.y - b.y) < 1e-3, (a, b)


def test_trolleys_slow_the_ramps_and_lengthen_the_stop():
    """Pulling trolleys: grip-limited speed-up, longer roll past the stop tag (trolleys have no brakes)."""
    from agv_sim.drive import DriveLimits
    empty = _agv2_lf(**{"line_follow.start_mode": "HIGH", "line_follow.speed_high": 0.4})
    route = straight_route(15)
    route.default_mode = "HIGH"
    _, s0 = run_simulation(empty, route)
    full = _agv2_lf(**{"line_follow.start_mode": "HIGH", "line_follow.speed_high": 0.4})
    full.hook.type = "tow"
    full.simulation.load_effects = True
    full.cart.count, full.cart.masses = 2, [400.0, 400.0]
    _, s1 = run_simulation(full, route)
    st = DriveLimits(full).state(800.0)
    ov0 = s0["line_follow"]["stops"][0]["stop_overshoot"]
    ov1 = s1["line_follow"]["stops"][0]["stop_overshoot"]
    b = DriveLimits(full).brake_mag(st, 1.0)
    assert b < full.line_follow.brake_decel
    assert abs(ov1 - 0.4 ** 2 / (2 * b)) < 0.05, (ov1, 0.4 ** 2 / (2 * b))
    assert ov1 > 2 * ov0, (ov0, ov1)
    assert s1["cycle_time_s"] > s0["cycle_time_s"] + 1.0
    assert st["a_max"] < full.line_follow.accel_rate
    tbl = s1["load_table"]
    assert [r["trolleys"] for r in tbl] == [0, 1, 2] and tbl[2]["accel"] < tbl[0]["accel"]


def test_swap_full_for_empty_trolleys():
    a = load_agv(os.path.join(DATA, "tow_tractor_example.yaml"))
    a.cart.empty_mass_kg = 50.0
    route = straight_route(20)
    route.waypoints.insert(1, Waypoint(x=12, y=5, name="line", action="stop", cart="swap", cart_load="empty"))
    res, s = run_simulation(a, route)
    full, empty = a.train_mass("full"), len(a.train_masses()) * 50.0
    assert abs(res.cart_mass[0] - full) < 1e-6 and abs(res.cart_mass[-1] - empty) < 1e-6
    assert res.attached[-1]
    labels = [e.label for e in res.events]
    assert f"Detach cart ({full:.0f} kg) @ line" in labels and f"Attach cart ({empty:.0f} kg) @ line" in labels
    assert not [w for w in s["warnings"] if "already" in w]


@slow
def test_pid_autotune_improves_and_stays_in_bounds():
    from agv_sim.pidtune import Tuner
    a = _agv2_lf()
    route = _tape_route(1.5, mode="SLOW")
    out = Tuner(a, route, rounds=2, workers=1).run()
    rw = [r for r in out["rows"] if r["set"] == "SLOW"][0]
    assert rw["score_new"] <= rw["score_old"] + 1e-9
    assert rw["kp_old"] / 3 - 1e-9 <= rw["kp_new"] <= rw["kp_old"] * 3 + 1e-9
    assert not out["after"]["harsh"]["aborted"]


def test_autoroute_trolley_safe_slowdown():
    from agv_sim.autoroute import AutoRoute, _Pt
    from agv_sim.factory_map import FactoryMap
    a = _agv2_lf(**{"line_follow.speed_high": 0.6, "line_follow.approach_decel_rate": 0.4,
                    "line_follow.decel_rate": 0.2})
    a.hook.type = "tow"
    a.cart.count, a.cart.masses, a.cart.empty_mass_kg, a.cart.rolling_resistance = 2, [412.0, 412.0], 60.0, 0.02
    route = straight_route(20)
    ar = AutoRoute(a, route, FactoryMap.for_route(route), mode="keep")
    steps, _ = ar._slowdown("HIGH", "APPROACH", 824.0)          # 0.4 m/s2 would push 168 N
    assert [m for m, _ in steps] == ["SLOW", "APPROACH"], steps  # gentle DECEL_RATE first
    steps0, _ = ar._slowdown("HIGH", "APPROACH", 0.0)
    assert [m for m, _ in steps0] == ["APPROACH"]
    pts = [_Pt(0, 0, Waypoint(x=0, y=0, name="A", action="stop")),
           _Pt(10, 0, Waypoint(x=10, y=0, name="B", action="stop", cart="swap", cart_load="empty")),
           _Pt(20, 0, Waypoint(x=20, y=0, name="C", action="stop", cart="detach"))]
    assert ar._leg_masses(pts) == [824.0, 120.0, 0.0]


def _shared_route():
    return RouteConfig(name="shared", loop=True, default_radius=1.5, default_mode="SLOW", waypoints=[
        Waypoint(x=2, y=2, name="Home", action="stop", dwell=5, mode="HIGH"),
        Waypoint(x=20, y=2), Waypoint(x=22.4, y=2, mode="SLOW"), Waypoint(x=26, y=2), Waypoint(x=26, y=8),
        Waypoint(x=20, y=8), Waypoint(x=20, y=2.0), Waypoint(x=17.2, y=2, mode="HIGH"), Waypoint(x=3.1, y=2, mode="SLOW")])


def test_rfid_tags_are_read_in_both_directions():
    a = _agv2_lf(**{"line_follow.speed_high": 0.6})
    res, s = run_simulation(a, _shared_route())
    assert s["line_follow"]["losses"] == 0
    assert any("Shared tape" in w for w in s["warnings"])
    # the return's SLOW tag at x=3.1 is also read on the way out -> the outbound run is slowed early
    out = (res.mode == 1) & (res.x > 4) & (res.x < 15) & (np.abs(res.y - 2) < 0.2) & (np.cos(res.th) > 0.9)
    assert np.max(np.abs(res.v[out])) < 0.25


@slow
def test_autoroute_repairs_shared_tags_and_aims_at_target():
    from agv_sim.autoroute import AutoRoute, auto_route
    from agv_sim.factory_map import FactoryMap
    a = _agv2_lf(**{"line_follow.speed_high": 0.6})
    r = _shared_route()
    fm = FactoryMap.for_route(r)
    out = auto_route(a, r, fm, mode="keep", max_iter=2)
    o, _ = AutoRoute(a, out["route"], fm, mode="keep")._conflicts(out["route"])
    assert not o                                   # nobody reads a speed-up tag where it must be slow
    fastest = out["summary"]["cycle_time_s"]
    r.target_cycle_s = fastest + 25.0
    out2 = auto_route(a, r, fm, mode="keep", max_iter=2)
    T = out2["summary"]["cycle_time_s"]
    assert fastest - 1.0 <= T <= r.target_cycle_s + 0.1, (fastest, T)


@slow
def test_autoroute_speeds_up_curves_for_a_tight_target():
    from agv_sim.autoroute import auto_route
    from agv_sim.config import SpeedMode
    from agv_sim.factory_map import FactoryMap
    a = _agv2_lf(**{"line_follow.speed_high": 0.6})
    a.speed_modes = [m for m in a.speed_modes if m.name.upper() in ("HIGH", "SLOW", "APPROACH")]
    a.speed_modes.append(SpeedMode(name="CURVE", v_straight=0.35, v_curve=0.35, v_reverse=0.2, accel=0.2, decel=0.25))
    r = _shared_route()
    fm = FactoryMap.for_route(r)
    base = auto_route(a, r, fm, mode="keep", max_iter=2)["summary"]["cycle_time_s"]
    r.target_cycle_s = base - 15.0
    out = auto_route(a, r, fm, mode="keep", max_iter=2)
    assert out["summary"]["cycle_time_s"] < base - 5.0, (base, out["summary"]["cycle_time_s"])
    assert any("CURVE" in (w.mode or "") for w in out["route"].waypoints)
    assert any(l.startswith("Faster curves") or "To reach the target" in l for l in out["log"])


def _lag_measured(lat):
    """Delay between the pose the sensor sees and the real pose, on a straight at constant speed."""
    import agv_sim.linefollow as L
    a = _agv2_lf(**{"line_follow.sensor_latency_s": lat, "line_follow.start_mode": "SLOW",
                    "line_follow.sensor_noise_mm": 0.0})
    got = []
    orig = L._Sensor.read

    def rd(self, x, y, th, lo, hi):
        got.append(x)
        return orig(self, x, y, th, lo, hi)
    L._Sensor.read = rd
    try:
        res, _ = run_simulation(a, straight_route(12))
    finally:
        L._Sensor.read = orig
    got = np.array(got)
    tk = np.arange(len(got)) * a.line_follow.dt
    xa = np.interp(tk, res.t - res.t[0], res.x)
    v = np.interp(tk, res.t - res.t[0], res.v)
    m = (v > 0.15) & (np.abs(np.gradient(v)) < 1e-4)
    return float(np.median((xa[m] - got[m]) / v[m]))


def test_sensor_latency_is_the_configured_delay():
    base = _lag_measured(0.0)                  # measuring offset (sampling alignment)
    assert abs(_lag_measured(0.05) - base - 0.05) < 0.002
    assert abs(_lag_measured(0.01) - base - 0.01) < 0.002


def test_fast_motor_lag_stays_stable():
    a = _agv2_lf(**{"line_follow.motor_time_constant": 0.003, "line_follow.start_mode": "SLOW"})
    res, s = run_simulation(a, straight_route(8))
    assert np.all(np.isfinite(res.v)) and np.max(np.abs(res.v)) < a.line_follow.speed_slow * 1.2 + 0.05
    assert s["line_follow"]["losses"] == 0


def test_tip_ratio_includes_floor_slope():
    from agv_sim.loads import support_polygon
    a = load_agv(os.path.join(DATA, "tow_tractor_example.yaml"))
    a.hook.type = "none"
    r = RouteConfig(name="ramp", loop=False, waypoints=[Waypoint(x=2, y=2, action="stop", grade=10.0),
                                                        Waypoint(x=12, y=2, action="stop", dwell=5)])
    res, s = run_simulation(a, r)
    dw = (res.mode == 0) & (res.t > res.t[-1] - 4)             # standing on the ramp
    poly = support_polygon(a)
    d_rear = a.cg_x - poly[:, 0].min()
    expect = a.chassis.cg_height * 0.10 / d_rear               # h tan(theta) / d
    assert abs(res.loads["tip_ratio"][dw].max() - expect) < 0.01 * expect + 1e-4


def test_friction_circle_limits_each_wheel():
    from agv_sim.drive import DriveLimits
    a = _agv2_lf()
    d = DriveLimits(a)
    st = d.state(0.0, 0.0, 0.0)
    # straight: asking for far more than the grip -> common part is cut to the grip
    vl, vr, lim = d.step(st, 0.5, 0.5, 50.0, 50.0, 0.01)
    assert lim and abs((vl - 0.5) / 0.01 - (st["grip"] - st["F_roll"]) / st["M"]) < 1e-6
    # same request while turning hard: sideways force leaves less grip for driving
    vl2, vr2, _ = d.step(st, 0.3, 0.7, 50.0, 50.0, 0.01)
    assert 0.5 * ((vl2 - 0.3) + (vr2 - 0.7)) < (vl - 0.5)


def test_todo_values_are_reported():
    from agv_sim.config import todo_fields
    txt = "chassis:\n  mass_kg: 120   # TODO weigh it\n  length: 1.0\nwheels:\n  friction_coeff: 0.5  # TODO\n"
    assert todo_fields(txt) == ["chassis.mass_kg: weigh it", "wheels.friction_coeff: placeholder"]
    a = _agv2_lf()
    a.sensors[-1].note = "TODO real position"
    _, s = run_simulation(a, straight_route(5))
    assert any("real position" in u for u in s.get("uncalibrated", []))


def test_wheel_slip_is_located():
    a = load_agv(os.path.join(DATA, "tow_tractor_example.yaml"))
    a.hook.type = "none"
    a.wheels.friction_coeff = 0.02                            # ice: every ramp slips
    _, s = run_simulation(a, straight_route(10))
    assert s.get("slip_spots") and any(w.startswith("WHEEL SLIP") for w in s["warnings"])
    sp = s["slip_spots"][0]
    assert sp["pct"] > 100 and sp["what"] in ("braking", "speeding up", "turning", "driving")


def test_sequence_adds_up_steps_repeats_and_waits():
    import tempfile
    from agv_sim.sequence import SequenceConfig, SequenceStep, load_sequence, run_sequence, save_sequence
    from agv_sim.config import save_route
    seq = SequenceConfig(name="t", target_s=1.0, steps=[
        SequenceStep(route="a.yaml", agv="tow_tractor_example.yaml", wait_after=12.0),
        SequenceStep(route="a.yaml", agv="tow_tractor_example.yaml", repeat=2),
        SequenceStep(route="nope.yaml", agv="tow_tractor_example.yaml")])
    with tempfile.TemporaryDirectory() as d:
        save_route(straight_route(12), os.path.join(d, "a.yaml"))
        p = os.path.join(d, "s.yaml")
        save_sequence(seq, p)
        seq = load_sequence(p)
        r = run_sequence(seq, d, DATA)
    a, b, c = r["steps"]
    assert abs(b["t0"] - (a["cycle_s"] + 12.0)) < 1e-6 and abs(b["t1"] - b["t0"] - 2 * b["cycle_s"]) < 1e-6
    assert abs(r["total_s"] - b["t1"]) < 1e-6 and c.get("error")
    assert r["gap_s"] > 0 and any("ABOVE the target" in w for w in r["warnings"])


if __name__ == "__main__":
    import traceback
    fast = "--fast" in sys.argv
    fails = skipped = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            marks = [getattr(m, "name", "") for m in getattr(fn, "pytestmark", [])]
            if fast and (getattr(fn, "slow", False) or "slow" in marks):
                skipped += 1
                continue
            try:
                fn()
                print("PASS", name)
            except Exception as ex:                        # noqa: BLE001 - keep running the others
                fails += 1
                print("FAIL", name, type(ex).__name__, ex)
                traceback.print_exc(limit=3)
    if skipped:
        print(f"({skipped} slow test(s) skipped)")
    sys.exit(1 if fails else 0)
