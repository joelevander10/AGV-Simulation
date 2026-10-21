"""Combined view: several routes on one map, shared stop areas."""
import os
import shutil

from agv_sim.combined import combine, label_for, vehicle_for
from agv_sim.config import MapConfig, RouteConfig, Waypoint, save_route

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")   # test vehicles / map (not your configs)


def _route(path, pts, stops):
    wps = [Waypoint(x=x, y=y, action="stop" if i in stops else "pass", dwell=2.0 if i in stops else 0.0)
           for i, (x, y) in enumerate(pts)]
    save_route(RouteConfig(name=os.path.basename(path), map=MapConfig(meters_per_pixel=0.05), loop=True,
                           waypoints=wps), path)


def test_combine_groups_shared_stops(tmp_path):
    rd, cd = tmp_path / "routes", tmp_path / "configs"
    rd.mkdir(); cd.mkdir()
    shutil.copy(os.path.join(DATA, "agv2.yaml"), cd / "Line (A).yaml")
    _route(str(rd / "Line (A)_auto.yaml"), [(1, 1), (10, 1), (10, 8), (1, 8)], {0, 2})
    _route(str(rd / "Line (B)_auto.yaml"), [(1.5, 1), (12, 1), (12, 5), (1.5, 5)], {0})
    assert vehicle_for("Line (A)_auto.yaml", str(cd)).endswith("Line (A).yaml")
    assert label_for("Line (A _ C)_auto.yaml") == "A / C"
    C = combine(["Line (A)_auto.yaml", "Line (B)_auto.yaml", "missing.yaml"], str(rd), str(cd), merge=2.0)
    assert [R["label"] for R in C["routes"]] == ["A", "B"]
    assert len(C["warnings"]) == 1 and "missing" in C["warnings"][0]
    assert all(R["runs"] and R["length_m"] > 20 for R in C["routes"])
    # the two start stops 0.5 m apart are one shared area, the far stop of A is its own
    assert len(C["areas"]) == 2
    shared = [a for a in C["areas"] if a["shared"]]
    assert len(shared) == 1 and {s["route"] for s in shared[0]["stops"]} == {"A", "B"}
    C0 = combine(["Line (A)_auto.yaml", "Line (B)_auto.yaml"], str(rd), str(cd), merge=0.1)
    assert len(C0["areas"]) == 3


def test_follow_in_area_replaces_only_the_area():
    import pytest
    from agv_sim.combined import follow_in_area
    W = lambda x, y, **k: Waypoint(x=x, y=y, **k)   # noqa: E731
    # reference: home spur on the right (x 30), attach at the start, detach further down
    ref = RouteConfig(loop=True, waypoints=[W(30, 2, action="stop", dwell=5, cart="attach"), W(30, 10), W(0, 10),
                                            W(0, 12), W(30, 12), W(30, 0, action="stop", dwell=3, cart="detach")])
    # target: old home loop at x 20..24, own line at x 0..5 (outside the area)
    tgt = RouteConfig(loop=True, waypoints=[W(22, 5, name="Home", action="stop", dwell=2), W(20, 5),
                                            W(20, 10), W(5, 10), W(5, 15, action="stop", dwell=4, cart="detach"),
                                            W(0, 15), W(0, 12), W(24, 12), W(24, 6, action="stop", dwell=1, cart="detach")])
    new, info = follow_in_area(ref, tgt, [18, -1, 32, 13], keep_own_waits=True)
    pts = [(w.x, w.y) for w in new.waypoints]
    refpts = [(w.x, w.y) for w in ref.waypoints]
    assert pts[0] == (30, 2) and new.waypoints[0].cart == "attach"             # starts where the reference starts
    assert new.waypoints[0].dwell == 2 and new.waypoints[0].name == "Home"      # ... with its own wait and name
    assert (5, 15) in pts and (0, 15) in pts                                    # own line kept
    assert all((x, y) in refpts for x, y in pts if 18 <= x <= 32 and -1 <= y <= 13)   # area = reference only
    det = [w for w in new.waypoints if w.cart == "detach" and w.x == 30]
    assert len(det) == 1 and det[0].dwell == 1                                  # its own detach wait
    assert info["removed_stops"] and info["added_stops"]
    with pytest.raises(ValueError):
        follow_in_area(ref, tgt, [100, 100, 110, 110])                           # route not in the area


def test_rfid_tags_insert_move_delete():
    import pytest
    from agv_sim.combined import apply_tag_changes, route_tags
    W = lambda x, y, **k: Waypoint(x=x, y=y, **k)   # noqa: E731
    r = RouteConfig(loop=True, default_radius=1.0,
                    waypoints=[W(0, 0, action="stop", dwell=2, mode="SLOW", name="Home"), W(5, 0, mode="HIGH"),
                               W(20, 0), W(20, 10, cart="detach", action="stop"), W(0, 10)])
    tags = [(t["i"], t["kind"], t["value"]) for t in route_tags(r)]
    assert tags == [(0, "home", "HOME"), (0, "speed", "SLOW"), (1, "speed", "HIGH"), (3, "stop", "STOP")]
    r2, info = apply_tag_changes(
        r, dels=[{"i": 1, "kind": "speed"}], sets=[{"i": 0, "kind": "speed", "value": "MED"}],
        adds=[{"x": 10, "y": 0.2, "kind": "speed", "value": "max"},
              {"x": 19.8, "y": 0.0, "kind": "speed", "value": "SLOW"},
              {"x": 12, "y": 10.1, "kind": "fork", "value": "LEFT"},
              {"x": 8, "y": 10, "kind": "stop", "dwell": 4}])
    pts = [(w.x, w.y, w.mode) for w in r2.waypoints]
    assert (5, 0, "HIGH") not in pts and pts[0] == (0, 0, "MED")        # removed tag point, changed start tag
    assert (10.0, 0.0, "MAX") in pts
    slow = [p for p in pts if p[2] == "SLOW"]
    assert len(slow) == 1 and slow[0][0] <= 20 - 1.0 + 1e-6              # moved out of the r = 1 m corner
    assert len(info["notes"]) == 1 and "out of the curve" in info["notes"][0]
    assert [(w.x, w.y) for w in r2.waypoints if w.fork == "left"] == [(12.0, 10.0)]
    assert [(w.x, w.dwell) for w in r2.waypoints if w.action == "stop" and not w.cart and w.x == 8.0] == [(8.0, 4.0)]
    with pytest.raises(ValueError):                                       # trolley stops stay (Route tab)
        apply_tag_changes(r2, dels=[{"i": [k for k, w in enumerate(r2.waypoints) if w.cart][0], "kind": "stop"}])
    # move the homepost to (10, 10): the cycle starts there, the old start becomes a plain point
    r3, _ = apply_tag_changes(r2, home={"x": 15, "y": 10})
    assert (r3.waypoints[0].x, r3.waypoints[0].y, r3.waypoints[0].action) == (15.0, 10.0, "stop")
    assert r3.waypoints[0].dwell == 2 and r3.waypoints[0].name == "Home"
    old = [w for w in r3.waypoints if w.x == 0 and w.y == 0][0]
    assert old.action == "pass" and old.mode == "MED"                     # its speed tag stays
