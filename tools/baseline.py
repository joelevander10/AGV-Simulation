"""Record / compare simulation results for every route in routes/ (regression baseline).

    python tools/baseline.py                         # print the table
    python tools/baseline.py --save baseline.json    # record
    python tools/baseline.py --compare baseline.json # compare against a record (changes marked)

Each route is paired with the vehicle whose name is in the route's name: 'AGV_Kayaba (OCC)_auto.yaml' ->
configs/'AGV_Kayaba (OCC).yaml'; 'AGV_Kayaba (ITG _ T6)' -> the first of ITG / T6 that exists; otherwise agv2.yaml.
Override with --pair "route file=config file" (repeatable).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from agv_sim import run_simulation  # noqa: E402
from agv_sim.config import load_agv, load_route  # noqa: E402

KEYS = [("cycle_s", "cycle", "{:.1f}"), ("tape_max_mm", "tape max", "{:.0f}"), ("tape_rms_mm", "tape rms", "{:.1f}"),
        ("overshoot_max_cm", "stop past tag", "{:.1f}"), ("torque_peak_nm", "torque pk", "{:.2f}"),
        ("torque_rms_nm", "torque rms", "{:.2f}"), ("energy_wh", "energy", "{:.2f}"), ("tip_safety", "tip x", "{:.2f}"),
        ("articulation_deg", "artic.", "{:.1f}"), ("collisions", "coll.", "{:d}"), ("grip_pct", "grip %", "{:.0f}"),
        ("push_n", "push N", "{:.0f}")]


def pair_config(route_path, cfg_dir, pairs):
    base = os.path.basename(route_path)
    if base in pairs:
        return os.path.join(cfg_dir, pairs[base])
    m = re.search(r"\(([^)]*)\)", base)
    if m:
        for name in re.split(r"[ _]+", m.group(1)):
            for cand in (f"AGV_Kayaba ({m.group(1)}).yaml", f"AGV_Kayaba ({name}).yaml"):
                p = os.path.join(cfg_dir, cand)
                if os.path.exists(p):
                    return p
    return os.path.join(cfg_dir, "agv2.yaml")


def metrics(route_path, cfg_path):
    agv = load_agv(cfg_path)
    route = load_route(route_path)
    t0 = time.time()
    res, s = run_simulation(agv, route)
    lf = s.get("line_follow") or {}
    ov = [r["stop_overshoot"] for r in lf.get("stops", []) if r.get("stop_overshoot") is not None]
    m = s.get("motor") or {}
    fin = lambda v: None if v is None or (isinstance(v, float) and not math.isfinite(v)) else v  # noqa: E731
    return {"config": os.path.basename(cfg_path), "cycle_s": s["cycle_time_s"],
            "tape_max_mm": lf.get("max_abs_mm"), "tape_rms_mm": lf.get("rms_mm"),
            "overshoot_max_cm": 100 * max(ov) if ov else None,
            "torque_peak_nm": m.get("peak_torque_nm"), "torque_rms_nm": m.get("rms_torque_nm"),
            "energy_wh": (s.get("energy") or {}).get("wh_per_cycle"), "tip_safety": fin(s.get("tip_safety_agv")),
            "articulation_deg": s.get("articulation_max_deg"), "collisions": len(s.get("collisions", [])),
            "grip_pct": 100 * s.get("traction_usage_max", 0.0),
            "push_n": (s.get("trolley_push") or {}).get("max_n"), "run_s": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", default=os.path.join(ROOT, "routes"))
    ap.add_argument("--configs", default=os.path.join(ROOT, "configs"))
    ap.add_argument("--save", default="")
    ap.add_argument("--compare", default="")
    ap.add_argument("--pair", action="append", default=[])
    a = ap.parse_args()
    pairs = dict(p.split("=", 1) for p in a.pair)
    old = json.load(open(a.compare)) if a.compare else {}
    out = {}
    print(f"{'route':42s}" + "".join(f"{lab:>14s}" for _, lab, _ in KEYS))
    for rp in sorted(glob.glob(os.path.join(a.routes, "*.yaml"))):
        name = os.path.basename(rp)
        try:
            r = metrics(rp, pair_config(rp, a.configs, pairs))
        except Exception as ex:  # noqa: BLE001
            print(f"{name:42s} ERROR {ex}")
            continue
        out[name] = r
        line = f"{name[:42]:42s}"
        for k, _, fmt in KEYS:
            v = r.get(k)
            txt = "-" if v is None else fmt.format(v)
            o = (old.get(name) or {}).get(k)
            if o is not None and v is not None and abs(v - o) > max(1e-6, 0.005 * abs(o)):
                txt += "*"
            line += f"{txt:>14s}"
        print(line + f"   [{r['config']}, {r['run_s']:.1f} s]")
        if old.get(name):
            diffs = [f"{k} {old[name][k]} -> {r[k]}" for k, _, _ in KEYS
                     if old[name].get(k) is not None and r.get(k) is not None
                     and abs(r[k] - old[name][k]) > max(1e-6, 0.005 * abs(old[name][k]))]
            for d in diffs:
                print("      changed:", d)
    if a.save:
        with open(a.save, "w") as f:
            json.dump(out, f, indent=1)
        print("saved", a.save)


if __name__ == "__main__":
    main()
