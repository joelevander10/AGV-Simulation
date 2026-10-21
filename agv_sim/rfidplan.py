"""RFID marking plans (page "RFID marks"): where the RFID tags go on the floor, for the people who lay
the tape and tags. Marking only - the routes and the simulation are not changed.

rfid/<name>.yaml:
    name: Kayaba RFID plan
    routes: ["AGV_Kayaba (UBC)_auto.yaml", ...]      # drawn underneath (and their map)
    commands:                                        # your own (custom) commands, besides the standard ones
      - {name: PIN UP, color: "#0b7285", note: "pusher pin up"}
    tags:                                            # the tag IDs you have / use
      - {id: "0100", command: HIGH, note: ""}
      - {id: "0400", command: STOP}
    marks:                                           # every place a tag is stuck on the floor
      - {id: "0100", x: 120.5, y: 61.9, note: ""}
One tag ID can be placed many times (same ID = same command for the AGV)."""
from __future__ import annotations

import os
import re

import yaml

COMMANDS = ["APPROACH", "SLOW", "MED", "HIGH", "MAX", "LEFT", "RIGHT", "STOP", "HOME", "REVERSE"]


def _id(v) -> str:
    s = str(v if v is not None else "").strip()
    if not s:
        raise ValueError("A tag has no ID.")
    if len(s) > 32 or not re.fullmatch(r"[A-Za-z0-9_.\-:]+", s):
        raise ValueError(f"Tag ID '{s}': use letters, digits, - _ . : (max 32).")
    return s


def plan_from_dict(d: dict) -> dict:
    d = dict(d or {})
    bad = set(d) - {"name", "routes", "commands", "tags", "marks"}
    if bad:
        raise ValueError(f"Unknown setting(s) {sorted(bad)}")
    cmds, names = [], set()
    for c in d.get("commands") or []:
        c = dict(c or {})
        nm = str(c.get("name") or "").strip().upper()[:24]
        if not nm:
            raise ValueError("A custom command has no name.")
        if not re.fullmatch(r"[A-Z0-9 _\-+/.#]+", nm):
            raise ValueError(f"Custom command '{nm}': use letters, digits, space and - _ + / . # only.")
        if nm in names:
            raise ValueError(f"Custom command '{nm}' is in the list twice.")
        if nm in COMMANDS:
            raise ValueError(f"'{nm}' is a standard command - give the custom one another name.")
        col = str(c.get("color") or "#495057").strip()
        if not re.fullmatch(r"#[0-9A-Fa-f]{6}", col):
            raise ValueError(f"Custom command '{nm}': colour must look like #1a2b3c.")
        names.add(nm)
        cmds.append({"name": nm, "color": col.lower(), **({"note": str(c["note"])} if c.get("note") else {})})
    tags, seen = [], set()
    for t in d.get("tags") or []:
        t = dict(t or {})
        i = _id(t.get("id"))
        if i in seen:
            raise ValueError(f"Tag ID '{i}' is in the list twice.")
        seen.add(i)
        cmd = str(t.get("command") or "").strip().upper()[:24]
        tags.append({"id": i, "command": cmd, **({"note": str(t["note"])} if t.get("note") else {})})
    marks = []
    for m in d.get("marks") or []:
        m = dict(m or {})
        i = _id(m.get("id"))
        if i not in seen:
            raise ValueError(f"Mark at ({m.get('x')}, {m.get('y')}) uses tag ID '{i}' which is not in the tag list.")
        marks.append({"id": i, "x": round(float(m["x"]), 3), "y": round(float(m["y"]), 3),
                      **({"note": str(m["note"])} if m.get("note") else {})})
    routes = [os.path.basename(str(r)) for r in (d.get("routes") or []) if r]
    out = {"name": str(d.get("name") or "RFID plan"), "routes": routes}
    if cmds:
        out["commands"] = cmds
    out.update({"tags": tags, "marks": marks})
    return out


def load_plan(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return plan_from_dict(yaml.safe_load(f) or {})


def save_plan(plan: dict, path: str) -> None:
    plan = plan_from_dict(plan)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# RFID marking plan - made with the AGV Simulator (page 'RFID marks'). Marking only.\n")
        yaml.safe_dump(plan, f, sort_keys=False, default_flow_style=None, width=120, allow_unicode=True)
