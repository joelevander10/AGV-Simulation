"""RFID marking plan files."""
import pytest

from agv_sim.rfidplan import load_plan, plan_from_dict, save_plan


def test_plan_round_trip(tmp_path):
    d = {"name": "Line 1", "routes": ["a/b/UBC.yaml"],
         "tags": [{"id": "0100", "command": "high"}, {"id": "0A00", "command": "STOP", "note": "dock"}],
         "marks": [{"id": "0100", "x": 1.23456, "y": 2}, {"id": "0100", "x": 5, "y": 2}, {"id": "0A00", "x": 9, "y": 2}]}
    p = tmp_path / "plan.yaml"
    save_plan(d, str(p))
    back = load_plan(str(p))
    assert back["routes"] == ["UBC.yaml"]
    assert [t["id"] for t in back["tags"]] == ["0100", "0A00"] and back["tags"][0]["command"] == "HIGH"
    assert back["tags"][1]["note"] == "dock"
    assert back["marks"][0] == {"id": "0100", "x": 1.235, "y": 2.0}
    assert sum(m["id"] == "0100" for m in back["marks"]) == 2          # one ID, placed twice


@pytest.mark.parametrize("bad", [
    {"tags": [{"id": "1"}, {"id": "1"}]},                              # same ID twice
    {"tags": [], "marks": [{"id": "9", "x": 0, "y": 0}]},               # mark with an unknown ID
    {"tags": [{"id": "a b"}]},                                         # space in the ID
    {"colour": "red"},                                                 # unknown setting
])
def test_plan_rejects(bad):
    with pytest.raises(ValueError):
        plan_from_dict(bad)


def test_custom_commands(tmp_path):
    d = {"commands": [{"name": "pin up", "color": "#0B7285", "note": "pusher pin up"}],
         "tags": [{"id": "0800", "command": "PIN UP"}], "marks": [{"id": "0800", "x": 1, "y": 1}]}
    p = tmp_path / "c.yaml"
    save_plan(d, str(p))
    back = load_plan(str(p))
    assert back["commands"] == [{"name": "PIN UP", "color": "#0b7285", "note": "pusher pin up"}]
    assert back["tags"][0]["command"] == "PIN UP"
    for bad in ({"name": "STOP", "color": "#000000"}, {"name": "X", "color": "red"}, {"name": ""}):
        with pytest.raises(ValueError):
            plan_from_dict({"commands": [bad]})
