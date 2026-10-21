"""Reverse branch tool: REVERSE tag past the junction, back into the branch, forward out again."""
import os

import pytest

from agv_sim import run_simulation
from agv_sim.branch import add_reverse_branch
from agv_sim.combined import route_tags
from agv_sim.config import MapConfig, RouteConfig, Waypoint, load_agv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")   # test vehicles / map (not your configs)
W = Waypoint


def _base():
    # main line x = 10 driven south (30 -> 5), back north on x = 20
    return RouteConfig(name="t", map=MapConfig(meters_per_pixel=0.05), loop=True, default_radius=1.0,
                       default_mode="SLOW", waypoints=[W(x=10, y=30, action="stop", dwell=2, mode="SLOW"),
                                                       W(x=10, y=5), W(x=20, y=5), W(x=20, y=30)])


def test_reverse_branch_points_and_simulation():
    agv = load_agv(os.path.join(DATA, "agv2.yaml"))
    base = _base()
    r, info = add_reverse_branch(base, agv, (10, 15), [(9.2, 16.5), (9.0, 19)], past_m=1.0, wait_end=5,
                                 end_name="Spur")
    seq = [(w.x, w.y, w.action, w.direction) for w in r.waypoints]
    assert seq[1] == (10.0, 14.0, "stop", "reverse")          # REVERSE tag 1 m past the junction
    assert seq[2] == (10.0, 15.0, "pass", "reverse")          # back up into the branch
    assert seq[4] == (9.0, 19.0, "stop", "forward") and r.waypoints[4].name == "Spur"
    assert seq[6] == (10.0, 15.0, "pass", "forward")          # forward out, back on the route
    assert seq[7] == (10, 5, "pass", "forward")
    assert len(base.waypoints) == 4                           # the input route is not changed
    assert [t["kind"] for t in route_tags(r) if t["i"] == 1] == ["reverse", "speed"]
    _, s = run_simulation(agv, r)
    _, s0 = run_simulation(agv, base)
    assert s.get("cycle_complete", True) and s["cycle_time_s"] > s0["cycle_time_s"] + 5


def test_reverse_branch_wrong_way_and_too_short():
    agv = load_agv(os.path.join(DATA, "agv2.yaml"))
    with pytest.raises(ValueError, match="cannot back into"):
        add_reverse_branch(_base(), agv, (20, 15), [(19.2, 16.5), (19, 19)])   # x = 20 is driven north
    with pytest.raises(ValueError, match="Not enough straight"):
        add_reverse_branch(_base(), agv, (10, 5.5), [(9.2, 7), (9, 9)], past_m=1.0)
    with pytest.raises(ValueError, match="on the route"):
        add_reverse_branch(_base(), agv, (15, 15), [(14, 17)])


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["keep", "plan"])
def test_auto_route_keeps_reverse_branch(mode):
    """The auto route must keep a reverse branch exactly as drawn (no sharp corners, no tags inside it)."""
    from agv_sim.autoroute import _branch_blocks, auto_route
    from agv_sim.factory_map import FactoryMap
    agv = load_agv(os.path.join(DATA, "agv2.yaml"))
    base = RouteConfig(name="t", map=MapConfig(meters_per_pixel=0.05), loop=True, default_radius=1.0,
                       default_mode="SLOW", waypoints=[W(x=8, y=17, action="stop", dwell=2, mode="SLOW"),
                                                       W(x=8, y=3), W(x=20, y=3), W(x=20, y=17)])
    r, _ = add_reverse_branch(base, agv, (8, 10), [(7.4, 11.2), (7.2, 13.5)], past_m=1.0)
    (a, b, j), = _branch_blocks(r)
    want = [(w.x, w.y, w.direction, w.radius, w.action, w.mode) for w in r.waypoints[a:b + 1]]
    out = auto_route(agv, r, FactoryMap(blank_size_m=(30.0, 20.0)), mode=mode, max_iter=3)
    nr = out["route"]
    k = next(i for i, w in enumerate(nr.waypoints) if w.name == "REVERSE")
    got = [(w.x, w.y, w.direction, w.radius, w.action, w.mode) for w in nr.waypoints[k:k + len(want)]]
    assert got == want
    assert out["summary"].get("cycle_complete", True)
