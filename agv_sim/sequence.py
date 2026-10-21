"""Sequences: several (route, vehicle) jobs driven one after another, e.g. one AGV that serves
UBC, then ITG, then OCC, T6 and Piston.

sequences/<name>.yaml:
    name: Morning shift
    target_s: 3600          # optional: target time for the whole sequence
    steps:
      - {route: "AGV_Kayaba (UBC)_auto.yaml", agv: "AGV_Kayaba (UBC).yaml", repeat: 1, wait_after: 0}
      - {route: "AGV_Kayaba (ITG _ T6)_auto.yaml", agv: "AGV_Kayaba (ITG).yaml"}

Every step is simulated with its own route and vehicle (trolleys, speeds, PID ...). `repeat` drives
the same route several times in a row, `wait_after` adds a pause after the step (loading, changing
trolleys, battery ...). The total time is the sum.
"""
from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional

import yaml


@dataclass
class SequenceStep:
    route: str = ""               # file name in routes/
    agv: str = ""                 # file name in configs/
    repeat: int = 1               # drive this route this many times in a row
    wait_after: float = 0.0       # s pause after the step
    note: str = ""


@dataclass
class SequenceConfig:
    name: str = "sequence"
    target_s: float = 0.0         # target for the whole sequence (0 = none)
    steps: List[SequenceStep] = field(default_factory=list)


def sequence_from_dict(d: dict) -> SequenceConfig:
    d = dict(d or {})
    allowed = {"name", "target_s", "steps"}
    bad = set(d) - allowed
    if bad:
        raise ValueError(f"Unknown sequence setting(s): {sorted(bad)}. Valid: {sorted(allowed)}")
    steps = []
    for i, st in enumerate(d.get("steps") or []):
        st = dict(st or {})
        bad = set(st) - {"route", "agv", "repeat", "wait_after", "note"}
        if bad:
            raise ValueError(f"Step {i + 1}: unknown setting(s) {sorted(bad)}")
        steps.append(SequenceStep(route=str(st.get("route") or ""), agv=str(st.get("agv") or ""),
                                  repeat=max(1, int(st.get("repeat") or 1)),
                                  wait_after=max(0.0, float(st.get("wait_after") or 0.0)),
                                  note=str(st.get("note") or "")))
    return SequenceConfig(name=str(d.get("name") or "sequence"), target_s=max(0.0, float(d.get("target_s") or 0.0)),
                          steps=steps)


def load_sequence(path: str) -> SequenceConfig:
    with open(path, "r", encoding="utf-8") as f:
        return sequence_from_dict(yaml.safe_load(f) or {})


def save_sequence(seq: SequenceConfig, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    d = asdict(seq)
    for st in d["steps"]:
        if st["repeat"] == 1:
            del st["repeat"]
        if not st["wait_after"]:
            del st["wait_after"]
        if not st["note"]:
            del st["note"]
    if not d["target_s"]:
        del d["target_s"]
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(d, f, sort_keys=False, default_flow_style=None, width=120, allow_unicode=True)


def run_sequence(seq: SequenceConfig, routes_dir: str, configs_dir: str,
                 on_step: Optional[Callable] = None, progress: Optional[Callable] = None) -> dict:
    """Simulate every step. on_step(i, step, agv, route, res, summary) may add its own data (the web
    app adds the playback). Returns {'steps': [...], 'total_s', 'target_s', 'gap_s', 'warnings'}."""
    from . import run_simulation
    from .config import load_agv, load_route
    out, warnings = [], []
    t = 0.0
    prev_end = None
    for i, st in enumerate(seq.steps):
        if progress:
            progress(i, len(seq.steps), st)
        rp, cp = os.path.join(routes_dir, os.path.basename(st.route)), os.path.join(configs_dir, os.path.basename(st.agv))
        row = {"i": i, "route": st.route, "agv": st.agv, "repeat": st.repeat, "wait_after": st.wait_after,
               "note": st.note, "t0": t}
        try:
            route = load_route(rp)
            agv = load_agv(cp)
            res, s = run_simulation(agv, route)
        except FileNotFoundError as ex:
            row.update(error=f"file not found: {os.path.basename(str(ex.filename or ex))}", t1=t)
            out.append(row)
            warnings.append(f"Step {i + 1}: {row['error']}")
            continue
        except Exception as ex:  # noqa: BLE001 - one broken step should not stop the rest
            row.update(error=str(ex), t1=t)
            out.append(row)
            warnings.append(f"Step {i + 1} ({st.route}): {ex}")
            continue
        lf = s.get("line_follow") or {}
        cyc = float(s["cycle_time_s"])
        dur = cyc * st.repeat + st.wait_after
        row.update(cycle_s=cyc, drive_s=cyc * st.repeat, t1=t + dur, distance_m=float(s.get("distance_axle_m", 0.0)) * st.repeat,
                   avg_speed=s.get("avg_speed_overall"), tape_max_mm=lf.get("max_abs_mm"),
                   tape_lost=bool(lf.get("aborted") or lf.get("losses")), complete=s.get("cycle_complete", True),
                   collisions=len(s.get("collisions", [])), n_warnings=len(s.get("warnings", [])),
                   energy_wh=(s.get("energy") or {}).get("wh_per_cycle", 0.0) * st.repeat,
                   route_target_s=s.get("target_cycle_s"), agv_name=s.get("agv"), route_name=s.get("route"))
        if not row["complete"]:
            warnings.append(f"Step {i + 1} ({st.route} with {st.agv}): the AGV loses the tape - the cycle is not completed.")
        if row["collisions"]:
            warnings.append(f"Step {i + 1} ({st.route}): {row['collisions']} collision(s).")
        # where does the AGV end, and where does the next step start?
        wps = route.waypoints
        start = (float(wps[0].x), float(wps[0].y))
        end = start if route.loop else (float(wps[-1].x), float(wps[-1].y))   # the drawn start / end stations
        if prev_end is not None and math.hypot(start[0] - prev_end[1][0], start[1] - prev_end[1][1]) > 1.0:
            warnings.append(f"Step {i + 1} starts {math.hypot(start[0] - prev_end[1][0], start[1] - prev_end[1][1]):.1f} m "
                            f"from where step {prev_end[0] + 1} ended - the drive in between is not in the sequence "
                            f"(add a route for it if it matters).")
        prev_end = (i, end)
        if on_step:
            on_step(i, st, agv, route, res, s, row)
        out.append(row)
        t += dur
    gap = (t - seq.target_s) if seq.target_s else None
    if seq.target_s and t > seq.target_s + 0.05:
        warnings.append(f"The sequence takes {t:.1f} s - {t - seq.target_s:.1f} s ABOVE the target {seq.target_s:.1f} s.")
    return {"name": seq.name, "steps": out, "total_s": t, "target_s": seq.target_s or None, "gap_s": gap,
            "warnings": warnings}


def format_sequence(r: dict) -> str:
    L = [f"Sequence: {r['name']}",
         f"{'#':>2s}  {'route':34s} {'vehicle':26s} {'x':>2s} {'cycle s':>8s} {'wait':>5s} {'start':>8s} {'end':>8s} "
         f"{'avg m/s':>7s} {'tape mm':>7s}  status"]
    for st in r["steps"]:
        if st.get("error"):
            L.append(f"{st['i'] + 1:2d}  {st['route'][:34]:34s} {st['agv'][:26]:26s}  ERROR {st['error']}")
            continue
        status = "TAPE LOST" if not st["complete"] else (f"{st['collisions']} collision(s)" if st["collisions"] else "ok")
        L.append(f"{st['i'] + 1:2d}  {st['route'][:34]:34s} {st['agv'][:26]:26s} {st['repeat']:2d} {st['cycle_s']:8.1f} "
                 f"{st['wait_after']:5.0f} {st['t0']:8.1f} {st['t1']:8.1f} {st['avg_speed'] or 0:7.3f} "
                 f"{(st['tape_max_mm'] if st['tape_max_mm'] is not None else float('nan')):7.0f}  {status}")
    L.append(f"Total: {r['total_s']:.1f} s ({r['total_s'] / 60:.1f} min)"
             + (f"  target {r['target_s']:.1f} s -> {abs(r['gap_s']):.1f} s {'OVER' if r['gap_s'] > 0.05 else 'to spare'}"
                if r.get("target_s") else ""))
    for w in r["warnings"]:
        L.append("! " + w)
    return "\n".join(L)
