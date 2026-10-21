"""AGV simulator command line.

  python main.py                                                       # EASY MODE: opens the browser app

  python main.py agv   --agv configs/agv2.yaml                         # draw the spec sheet
  python main.py edit  --map maps/factory.png --route routes/my.yaml   # draw a route on the map
  python main.py run   --agv configs/agv2.yaml --route routes/demo_route.yaml --animate
  python main.py sweep --agv configs/agv2.yaml --route routes/demo_route.yaml \
                       --param performance.v_curve --values 0.2 0.3 0.4 0.5
  python main.py turn  --agv configs/agv2.yaml --radii 0.75 1 1.5 2 3 --angle 90       # turning-radius study
  python main.py autoroute --agv configs/agv2.yaml --route routes/my.yaml --mode keep --save routes/my_auto.yaml
  python main.py tune-pid  --agv configs/agv2.yaml --route routes/my.yaml [--apply]   # PID auto-tune
  python main.py sequence --file sequences/shift.yaml                               # several routes + vehicles in a row
  python main.py import-profile --json ".../profiles/agv2_kim.json" --agv configs/agv2.yaml  # sync the PID
Override any AGV value on the fly:  --set performance.accel=0.4 --set cart.mass_kg=250
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys

import matplotlib
import numpy as np

if not os.environ.get("DISPLAY") and sys.platform.startswith("linux") and "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from agv_sim.config import (AGVConfig, MapConfig, RouteConfig, load_agv, load_route,  # noqa: E402
                            save_route, set_param)
from agv_sim.factory_map import FactoryMap  # noqa: E402
from agv_sim import run_simulation  # noqa: E402
from agv_sim.analysis import format_report  # noqa: E402


def _apply_sets(agv, sets):
    for item in sets or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got '{item}'")
        k, v = item.split("=", 1)
        set_param(agv, k.strip(), v.strip())


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _save_timeseries(res, path):
    from agv_sim.analysis import save_timeseries
    save_timeseries(res, path)


# ----------------------------------------------------------------------------- commands
def cmd_agv(args):
    from agv_sim.visual import draw_agv_spec
    agv = load_agv(args.agv)
    _apply_sets(agv, args.set)
    fig = draw_agv_spec(agv)
    if args.out:
        fig.savefig(args.out, dpi=130)
        print(f"Saved {args.out}")
    if not args.no_show:
        plt.show()


def cmd_edit(args):
    from agv_sim.config import load_route as _lr
    from agv_sim.editor import RouteEditor
    if os.path.exists(args.route):
        route = _lr(args.route)
        if args.map:
            route.map.image = os.path.abspath(args.map)
    else:
        if not args.map:
            raise SystemExit("New route: give --map factory.png")
        route = RouteConfig(name=os.path.splitext(os.path.basename(args.route))[0],
                            map=MapConfig(image=os.path.abspath(args.map), meters_per_pixel=args.mpp))
    if args.mpp and os.path.exists(args.route) and args.mpp != 0.02:
        route.map.meters_per_pixel = args.mpp
    agv = load_agv(args.agv) if args.agv else None
    RouteEditor(route, args.route, agv).show()


def cmd_run(args):
    agv = load_agv(args.agv)
    _apply_sets(agv, args.set)
    route = load_route(args.route)
    fmap = FactoryMap.for_route(route)
    res, summary = run_simulation(agv, route, fmap)
    text = format_report(summary)
    print(text)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "report.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, indent=2)
    _save_timeseries(res, os.path.join(args.out, "timeseries.csv"))

    from agv_sim.visual import animate, plot_report
    fig = plot_report(res, summary, fmap)
    fig.savefig(os.path.join(args.out, "report.png"), dpi=110)
    print(f"\nResults written to {os.path.abspath(args.out)}/ (report.txt, summary.json, timeseries.csv, report.png)")

    anim = None
    if args.animate or args.save_anim:
        _, anim = animate(res, fmap, speed=args.speed, save=args.save_anim, follow=args.follow)
        if args.save_anim:
            print(f"Animation saved to {args.save_anim}")
    if not args.no_show:
        plt.show()
    return anim


def cmd_sweep(args):
    base = load_agv(args.agv)
    _apply_sets(base, args.set)
    route = load_route(args.route)
    fmap = FactoryMap.for_route(route)
    rows = []
    hdr = (f"{args.param:>28s} | cycle s | cyc/h | drive s | min R axle | peak Nm | RMS Nm | "
           f"peak rpm | grip % | Wh/cyc | coll | margin")
    print(hdr)
    print("-" * len(hdr))
    for val in args.values:
        agv = copy.deepcopy(base)
        set_param(agv, args.param, val)
        try:
            _, s = run_simulation(agv, route, fmap)
        except Exception as ex:
            print(f"{val:>28s} | error: {ex}")
            continue
        m = s["motor"]
        row = [val, s["cycle_time_s"], s["cycles_per_hour"], s["drive_time_s"], s["min_turn_radius_axle_m"],
               m["peak_torque_nm"], m["rms_torque_nm"], m["peak_rpm"], 100 * s["traction_usage_max"],
               s["energy"]["wh_per_cycle"], len(s.get("collisions", [])), len(s.get("clearance_violations", []))]
        rows.append(row)
        print(f"{val:>28s} | {row[1]:7.1f} | {row[2]:5.1f} | {row[3]:7.1f} | {row[4]:10.2f} | {row[5]:7.2f} | "
              f"{row[6]:6.2f} | {row[7]:8.0f} | {row[8]:6.0f} | {row[9]:6.2f} | {row[10]:4d} | {row[11]:6d}")
    if args.csv:
        with open(args.csv, "w", encoding="utf-8") as f:
            f.write("value,cycle_s,cycles_h,drive_s,min_R_axle,peak_Nm,rms_Nm,peak_rpm,grip_pct,wh,coll,margin\n")
            for r in rows:
                f.write(",".join(str(x) for x in r) + "\n")
        print(f"Saved {args.csv}")


def cmd_turn(args):
    from agv_sim.turning import turning_study
    agv = load_agv(args.agv)
    _apply_sets(agv, args.set)
    st = turning_study(agv, radii=args.radii, angle_deg=args.angle, with_cart=not args.no_cart,
                       show_radius=args.show)
    print(st["text"])
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "turning.txt"), "w", encoding="utf-8") as f:
        f.write(st["text"])
    st["figure"].savefig(os.path.join(args.out, "turning.png"), dpi=110)
    print(f"\nSaved {os.path.join(args.out, 'turning.txt')} and turning.png")
    if not args.no_show:
        plt.show()


def cmd_autoroute(args):
    from agv_sim.autoroute import auto_route
    from agv_sim.factory_map import FactoryMap
    agv = load_agv(args.agv)
    _apply_sets(agv, args.set)
    route = load_route(args.route)
    if args.target:
        route.target_cycle_s = args.target
    out = auto_route(agv, route, FactoryMap.for_route(route), mode=args.mode, style=args.style,
                     margin=args.margin, max_radius=args.max_radius, min_radius=args.min_radius,
                     tape_limit_pct=args.tape_limit, max_iter=args.max_iter, log=print)
    dest = args.save or os.path.splitext(args.route)[0] + " (auto).yaml"
    save_route(out["route"], dest)
    print(f"Saved {dest}")


def cmd_tune_pid(args):
    from agv_sim.pidtune import Tuner, format_result
    agv = load_agv(args.agv)
    _apply_sets(agv, args.set)
    route = load_route(args.route)
    out = Tuner(agv, route, rounds=args.rounds, log=print).run()
    print()
    print(format_result(out))
    if args.apply and out["changes"]:
        from agv_sim.profile_import import update_yaml_text
        with open(args.agv, "r", encoding="utf-8") as f:
            text = f.read()
        with open(args.agv, "w", encoding="utf-8") as f:
            f.write(update_yaml_text(text, out["changes"], "line_follow"))
        load_agv(args.agv)
        print(f"Written to {args.agv} (line_follow).")


def cmd_sequence(args):
    from agv_sim.sequence import format_sequence, load_sequence, run_sequence
    seq = load_sequence(args.file)
    root = os.path.dirname(os.path.abspath(__file__))
    r = run_sequence(seq, args.routes or os.path.join(root, "routes"), args.configs or os.path.join(root, "configs"),
                     progress=lambda i, n, st: print(f"step {i + 1}/{n}: {st.route} with {st.agv} ..."))
    print()
    print(format_sequence(r))


def cmd_import_profile(args):
    from agv_sim.profile_import import extras, load_profile, profile_to_line_follow, update_yaml_text
    data = load_profile(args.json)
    vals = profile_to_line_follow(data)
    vals["enabled"] = True
    with open(args.agv, "r", encoding="utf-8") as f:
        text = f.read()
    text = update_yaml_text(text, vals, "line_follow")
    with open(args.agv, "w", encoding="utf-8") as f:
        f.write(text)
    agv = load_agv(args.agv)            # validates the result
    for k, v in extras(data).items():
        print(f"  (profile also says {k} = {v}; current value in {args.agv}: "
              f"{eval('agv.' + k)})")
    print(f"Imported {len(vals)} controller values from {args.json} into {args.agv} (line_follow).")


def main():
    if len(sys.argv) == 1 or sys.argv[1] in ("app", "gui", "ui"):
        import app
        sys.argv = sys.argv[:1] + sys.argv[2:]
        return app.main()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("agv", help="draw the AGV spec sheet")
    a.add_argument("--agv", required=True)
    a.add_argument("--out", default="")
    a.add_argument("--set", action="append")
    a.add_argument("--no-show", action="store_true")
    a.set_defaults(func=cmd_agv)

    e = sub.add_parser("edit", help="interactive route editor")
    e.add_argument("--route", required=True, help="route YAML to open or create")
    e.add_argument("--map", default="", help="factory image (PNG/JPG)")
    e.add_argument("--mpp", type=float, default=0.02, help="metres per pixel (or calibrate with 'c')")
    e.add_argument("--agv", default="", help="AGV YAML (for an accurate path preview with sensor tracking)")
    e.set_defaults(func=cmd_edit)

    r = sub.add_parser("run", help="simulate one cycle")
    r.add_argument("--agv", required=True)
    r.add_argument("--route", required=True)
    r.add_argument("--out", default="results")
    r.add_argument("--set", action="append", help="override, e.g. performance.accel=0.4")
    r.add_argument("--animate", action="store_true")
    r.add_argument("--speed", type=float, default=4.0, help="animation playback speed factor")
    r.add_argument("--follow", action="store_true", help="camera follows the AGV")
    r.add_argument("--save-anim", default="", help="save animation (.gif or .mp4, mp4 needs ffmpeg)")
    r.add_argument("--no-show", action="store_true")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("sweep", help="vary one parameter and compare")
    s.add_argument("--agv", required=True)
    s.add_argument("--route", required=True)
    s.add_argument("--param", required=True, help="dotted name, e.g. performance.v_curve")
    s.add_argument("--values", nargs="+", required=True)
    s.add_argument("--set", action="append")
    s.add_argument("--csv", default="")
    s.set_defaults(func=cmd_sweep)

    tu = sub.add_parser("turn", help="turning-radius study: arc radii vs pivot, swept envelope, loads")
    tu.add_argument("--agv", required=True)
    tu.add_argument("--radii", type=float, nargs="+", default=None, help="drawn radii to test (m)")
    tu.add_argument("--angle", type=float, default=90.0, help="corner angle (deg)")
    tu.add_argument("--no-cart", action="store_true", help="without the towed / carried cart")
    tu.add_argument("--show", type=float, default=None, help="radius to draw in the envelope panel")
    tu.add_argument("--out", default="results/turning")
    tu.add_argument("--set", action="append")
    tu.add_argument("--no-show", action="store_true")
    tu.set_defaults(func=cmd_turn)

    au = sub.add_parser("autoroute", help="plan a lane through the stops / optimise radii and speed tags")
    au.add_argument("--agv", required=True)
    au.add_argument("--route", required=True)
    au.add_argument("--mode", choices=["plan", "keep"], default="keep",
                    help="plan: new lane through the stops; keep: your corners, better radii + tags")
    au.add_argument("--style", choices=["orthogonal", "any"], default="orthogonal")
    au.add_argument("--margin", type=float, default=None, help="extra clearance to obstacles (m)")
    au.add_argument("--max-radius", type=float, default=2.5)
    au.add_argument("--min-radius", type=float, default=0.75)
    au.add_argument("--tape-limit", type=float, default=70.0, help="max tape error, %% of sensor range")
    au.add_argument("--max-iter", type=int, default=8)
    au.add_argument("--target", type=float, default=0.0, help="target cycle time in s (default: the route's)")
    au.add_argument("--save", default="", help="output route YAML (default: '<route> (auto).yaml')")
    au.add_argument("--set", action="append")
    au.set_defaults(func=cmd_autoroute)

    tp = sub.add_parser("tune-pid", help="auto-tune Kp / Td of the tape PID on a route")
    tp.add_argument("--agv", required=True)
    tp.add_argument("--route", required=True)
    tp.add_argument("--rounds", type=int, default=8)
    tp.add_argument("--apply", action="store_true", help="write the new gains into the vehicle YAML")
    tp.add_argument("--set", action="append")
    tp.set_defaults(func=cmd_tune_pid)

    sq = sub.add_parser("sequence", help="run several routes (each with its own vehicle) one after another")
    sq.add_argument("--file", required=True, help="sequences/<name>.yaml")
    sq.add_argument("--routes", default="", help="routes folder (default: routes/)")
    sq.add_argument("--configs", default="", help="vehicles folder (default: configs/)")
    sq.set_defaults(func=cmd_sequence)

    ip = sub.add_parser("import-profile", help="copy PID / speed / motor-map values from the AGV profile JSON")
    ip.add_argument("--json", required=True, help="profiles/<agv>.json of the AGV code")
    ip.add_argument("--agv", required=True, help="vehicle YAML to update (comments are kept)")
    ip.set_defaults(func=cmd_import_profile)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
