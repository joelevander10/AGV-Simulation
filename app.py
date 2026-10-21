#!/usr/bin/env python3
"""AGV Simulator - easy mode.

    python3 app.py          (or double-click "Start Simulator.command" on a Mac)

Opens the simulator in your web browser: load a floor plan, click the route, edit the
vehicle in a form, press Run. Everything stays on this computer (http://127.0.0.1).
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import shutil
import socket
import sys
import threading
import traceback
import webbrowser
from dataclasses import asdict

import matplotlib

matplotlib.use("Agg")  # the web server draws figures off-screen

import numpy as np  # noqa: E402
import yaml  # noqa: E402

try:
    from flask import Flask, jsonify, request, send_from_directory  # noqa: E402
except ImportError:  # pragma: no cover
    sys.exit("Flask is missing. Run:  python3 -m pip install -r requirements.txt")

from PIL import Image  # noqa: E402
from werkzeug.exceptions import HTTPException  # noqa: E402

from agv_sim import build_plan, format_report, run_simulation  # noqa: E402
from agv_sim.analysis import cart_rect, rect_corners  # noqa: E402
from agv_sim.config import (AGVConfig, MapConfig, RouteConfig, Waypoint, _from_dict,  # noqa: E402
                            load_agv, load_route, save_route)
from agv_sim.factory_map import FactoryMap  # noqa: E402
from agv_sim.visual import agv_shapes, plot_report  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
DIRS = {k: os.path.join(ROOT, k) for k in ("configs", "routes", "maps", "results", "sequences", "rfid")}
WEB = os.path.join(ROOT, "web")
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp")
SIM_LOCK = threading.Lock()   # matplotlib + simulation one at a time

app = Flask(__name__, static_folder=None)


# ----------------------------------------------------------------------------- helpers
class UserError(Exception):
    pass


def _clean_name(name: str, ext: str) -> str:
    name = os.path.basename(str(name or "")).strip()
    name = re.sub(r"[^A-Za-z0-9 _.\-()]+", "_", name)
    if not name or name.startswith("."):
        raise UserError("Please give a file name.")
    if ext and not name.lower().endswith(ext):
        name = os.path.splitext(name)[0] + ext
    return name


def _list(kind, exts):
    d = DIRS[kind]
    os.makedirs(d, exist_ok=True)
    return sorted(f for f in os.listdir(d) if f.lower().endswith(exts) and not f.startswith("."))


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    return obj


def _r(a, nd=3):
    return [None if not math.isfinite(v) else round(float(v), nd) for v in np.asarray(a, dtype=float)]


def _agv_from(data) -> AGVConfig:
    try:
        return _from_dict(AGVConfig, data)
    except Exception as ex:
        raise UserError(f"Vehicle settings: {ex}")


def _route_from(data) -> RouteConfig:
    """Browser route dict (map = file name inside maps/) -> RouteConfig."""
    d = dict(data or {})
    m = dict(d.get("map") or {})
    img = m.get("image") or ""
    if img:
        img = os.path.join(DIRS["maps"], os.path.basename(img))
        if not os.path.exists(img):
            raise UserError(f"Floor plan '{os.path.basename(img)}' not found in the maps folder.")
    m["image"] = img
    d["map"] = m
    wps = []
    for w in d.get("waypoints") or []:
        w = {k: v for k, v in dict(w).items() if not (v is None or v == "" and k in ("radius", "speed"))}
        wps.append(w)
    d["waypoints"] = wps
    try:
        return _from_dict(RouteConfig, d)
    except Exception as ex:
        raise UserError(f"Route: {ex}")


def _route_to_dict(route: RouteConfig) -> dict:
    img = route.map.image
    if img:
        img = os.path.abspath(img)
        if os.path.dirname(img) != os.path.abspath(DIRS["maps"]):
            dst = os.path.join(DIRS["maps"], os.path.basename(img))   # keep all floor plans in maps/
            if os.path.exists(img) and not os.path.exists(dst):
                shutil.copy2(img, dst)
            img = dst
    d = asdict(route)
    d["map"]["image"] = os.path.basename(img) if img else ""
    return d


@app.errorhandler(UserError)
def _user_error(ex):
    return jsonify({"error": str(ex)}), 400


@app.errorhandler(Exception)
def _any_error(ex):
    if isinstance(ex, HTTPException):   # 404 etc. stay what they are
        return ex
    traceback.print_exc()
    return jsonify({"error": f"{type(ex).__name__}: {ex}"}), 500


# ----------------------------------------------------------------------------- pages / files
@app.get("/")
def index():
    return send_from_directory(WEB, "index.html")


@app.get("/maps/<path:fn>")
def map_file(fn):
    return send_from_directory(DIRS["maps"], os.path.basename(fn))


@app.get("/results/<path:fn>")
def result_file(fn):
    return send_from_directory(DIRS["results"], fn, as_attachment=request.args.get("dl") == "1")


@app.get("/api/files")
def files():
    return jsonify({"agvs": _list("configs", (".yaml", ".yml")),
                    "routes": _list("routes", (".yaml", ".yml")),
                    "sequences": _list("sequences", (".yaml", ".yml")),
                    "rfidplans": _list("rfid", (".yaml", ".yml")),
                    "maps": _list("maps", IMG_EXT)})


@app.post("/api/upload_map")
def upload_map():
    f = request.files.get("file")
    if not f or not f.filename:
        raise UserError("No file received.")
    if not f.filename.lower().endswith(IMG_EXT):
        raise UserError("Please use a PNG or JPG image of the floor plan.")
    name = _clean_name(f.filename, "")
    f.save(os.path.join(DIRS["maps"], name))
    return jsonify({"name": name, "maps": _list("maps", IMG_EXT)})


# ----------------------------------------------------------------------------- vehicle
@app.get("/api/agv/<name>")
def get_agv(name):
    path = os.path.join(DIRS["configs"], _clean_name(name, ""))
    try:
        agv = load_agv(path)
    except FileNotFoundError:
        raise UserError(f"Vehicle file '{name}' not found.")
    except Exception as ex:
        raise UserError(f"Could not read '{name}': {ex}")
    return jsonify({"name": os.path.basename(path), "data": asdict(agv)})


@app.post("/api/agv/<name>")
def save_agv(name):
    name = _clean_name(name, ".yaml")
    agv = _agv_from((request.get_json() or {}).get("data"))
    path = os.path.join(DIRS["configs"], name)
    if os.path.exists(path):  # keep the original (with its comments) once
        bak = os.path.join(DIRS["configs"], ".backup")
        os.makedirs(bak, exist_ok=True)
        orig = os.path.join(bak, name)
        if not os.path.exists(orig):
            shutil.copy2(path, orig)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Vehicle '{agv.name}' - saved from the AGV Simulator app.\n"
                "# Body frame: origin = drive-axle centre, +x forward, +y left, +z up (metres).\n")
        yaml.safe_dump(asdict(agv), f, sort_keys=False, default_flow_style=None, width=110)
    return jsonify({"name": name, "agvs": _list("configs", (".yaml", ".yml"))})


# ----------------------------------------------------------------------------- route
@app.get("/api/route/<name>")
def get_route(name):
    path = os.path.join(DIRS["routes"], _clean_name(name, ""))
    try:
        route = load_route(path)
    except FileNotFoundError:
        raise UserError(f"Route file '{name}' not found.")
    except Exception as ex:
        raise UserError(f"Could not read '{name}': {ex}")
    return jsonify({"name": os.path.basename(path), "data": _route_to_dict(route),
                    "maps": _list("maps", IMG_EXT)})


@app.post("/api/route/<name>")
def put_route(name):
    name = _clean_name(name, ".yaml")
    route = _route_from((request.get_json() or {}).get("data"))
    save_route(route, os.path.join(DIRS["routes"], name))
    return jsonify({"name": name, "routes": _list("routes", (".yaml", ".yml"))})


@app.post("/api/preview")
def preview():
    """Smoothed path (what the tracking point follows) for the live preview while editing."""
    body = request.get_json() or {}
    route = _route_from(body.get("route"))
    if len(route.waypoints) < 2:
        return jsonify({"runs": [], "warnings": [], "length": 0})
    agv = _agv_from(body.get("agv"))
    try:
        plan = build_plan(route, agv)
    except Exception as ex:
        return jsonify({"runs": [], "warnings": [f"Cannot build this route yet: {ex}"], "length": 0})
    runs = []
    for run in plan.runs:
        step = max(1, int(round(0.05 / agv.simulation.ds)))
        idx = list(range(0, len(run.x), step)) + [len(run.x) - 1]
        runs.append({"x": _r(run.x[idx]), "y": _r(run.y[idx]), "dir": run.direction})
    return jsonify({"runs": runs, "warnings": list(plan.warnings), "length": round(plan.length, 2)})


@app.post("/api/reverse_branch")
def reverse_branch():
    """Route tab tool: add a reverse branch (REVERSE tag past the junction, back into the branch, stop at
    its end, forward out again) to the route being edited. Returns the new route (not saved yet)."""
    from agv_sim.branch import add_reverse_branch
    body = request.get_json() or {}
    route = _route_from(body.get("route"))
    agv = _agv_from(body.get("agv"))
    o = body.get("options") or {}
    j = body.get("junction") or []
    pts = body.get("branch") or []
    if len(j) != 2:
        raise UserError("Click the junction on the route first.")

    def num(k, d):
        v = o.get(k)
        return d if v in (None, "") else float(v)
    try:
        new, info = add_reverse_branch(route, agv, j, pts, past_m=num("past_m", None), wait_rev=num("wait_rev", 1.0),
                                       wait_end=num("wait_end", 5.0), end_name=str(o.get("end_name") or ""),
                                       end_cart=str(o.get("end_cart") or ""), mode_in=str(o.get("mode_in") or "SLOW"),
                                       radius=num("radius", 1.0), end_mode=str(o.get("end_mode") or ""))
    except ValueError as ex:
        raise UserError(str(ex))
    return jsonify(_json_safe({"route": _route_to_dict(new), "info": info}))


@app.post("/api/combined")
def combined():
    """Several saved routes on one map (tab 'All routes'): their tape and the stop areas."""
    from agv_sim.combined import combine
    body = request.get_json() or {}
    names = [os.path.basename(str(n)) for n in (body.get("routes") or []) if n]
    if not names:
        raise UserError("Tick at least one route.")
    agv = None
    if body.get("agv"):
        try:
            agv = _agv_from(body.get("agv"))
        except UserError:
            agv = None
    merge = max(0.0, float(body.get("merge") or 2.5))
    return jsonify(_json_safe(combine(names, DIRS["routes"], DIRS["configs"], merge=merge, agv_override=agv)))


@app.post("/api/follow")
def follow():
    """Combined tab: make routes drive like a reference route inside a rectangle. preview (default)
    returns the new routes (+ a simulation check); save=true writes them (old files to routes/.backup)."""
    import time
    from agv_sim.combined import follow_in_area, vehicle_for, _decimate
    body = request.get_json() or {}
    ref_name = os.path.basename(str(body.get("reference") or ""))
    targets = [os.path.basename(str(t)) for t in (body.get("targets") or []) if t and os.path.basename(str(t)) != ref_name]
    area = [float(v) for v in (body.get("area") or [])]
    if not ref_name:
        raise UserError("Choose the route to follow.")
    if len(area) != 4 or abs(area[2] - area[0]) < 0.2 or abs(area[3] - area[1]) < 0.2:
        raise UserError("Draw the area first (▭ Draw area, then drag on the map).")
    if not targets:
        raise UserError("Tick at least one route to change.")
    ref_path = os.path.join(DIRS["routes"], ref_name)
    if not os.path.exists(ref_path):
        raise UserError(f"Route '{ref_name}' not found.")
    ref = load_route(ref_path)
    keep = bool(body.get("keep_waits", True))
    check, save = bool(body.get("check", True)), bool(body.get("save"))
    out, backup = [], None
    for t in targets:
        tp = os.path.join(DIRS["routes"], t)
        row = {"file": t}
        try:
            old = load_route(tp)
            new, info = follow_in_area(ref, old, area, keep_own_waits=keep)
            row.update(info)
            vf = vehicle_for(t, DIRS["configs"])
            agv = load_agv(vf) if vf else None
            row["vehicle"] = os.path.basename(vf) if vf else ""
            if agv is not None:
                plan = build_plan(new, agv)
                row["runs"] = []
                for run in plan.runs:
                    x, y = _decimate(run.x, run.y)
                    row["runs"].append({"x": _r(x), "y": _r(y), "dir": int(run.direction)})
                row["length"] = [round(build_plan(old, agv).length, 1), round(plan.length, 1)]
                if check:
                    with SIM_LOCK:
                        res = []
                        for r_ in (old, new):
                            _, s = run_simulation(agv, r_)
                            lf = s.get("line_follow") or {}
                            res.append({"cycle_s": s["cycle_time_s"], "complete": bool(s.get("cycle_complete", True)),
                                        "tape_mm": lf.get("max_abs_mm"),
                                        "lost": next(({"x": e.get("x"), "y": e.get("y")} for e in (lf.get("events") or [])
                                                      if e.get("kind") in ("tape lost", "stalled", "timeout")), None)})
                    row["sim"] = res
                    row["target_s"] = old.target_cycle_s or None
            if save:
                if backup is None:
                    backup = os.path.join(DIRS["routes"], ".backup", time.strftime("%Y-%m-%d %H%M") + " before follow "
                                          + re.sub(r"[^A-Za-z0-9 _.()-]+", "_", os.path.splitext(ref_name)[0]))
                    os.makedirs(backup, exist_ok=True)
                shutil.copy2(tp, os.path.join(backup, t))
                save_route(new, tp)
                row["saved"] = True
        except Exception as ex:  # noqa: BLE001 - report per route
            row["error"] = str(ex)
        out.append(row)
    return jsonify(_json_safe({"reference": ref_name, "area": area, "routes": out,
                               "backup": os.path.relpath(backup, ROOT) if backup else None}))


@app.post("/api/tags")
def save_tags():
    """Combined tab: write the RFID tags (speed modes, LEFT / RIGHT, stops, homepost) dragged / moved /
    deleted on the map into the route files. The old files go to routes/.backup/<date> tags/ first."""
    import time
    from agv_sim.combined import apply_tag_changes, file_sig
    body = request.get_json() or {}
    changes = body.get("changes") or []
    if not changes:
        raise UserError("No tag changes to save.")
    # check every file first - nothing is written when one of them changed meanwhile
    jobs = []
    for ch in changes:
        f = os.path.basename(str(ch.get("file") or ""))
        path = os.path.join(DIRS["routes"], f)
        if not f or not os.path.exists(path):
            raise UserError(f"Route '{f}' not found.")
        if ch.get("sig") and ch["sig"] != file_sig(path):
            raise UserError(f"'{f}' was changed after the map was loaded (Route tab or another window) - "
                            f"press ⟳ Reload and place the tags again.")
        route = load_route(path)
        try:
            route, info = apply_tag_changes(route, ch.get("dels") or [], ch.get("sets") or [], ch.get("adds") or [],
                                            ch.get("home"))
        except ValueError as ex:
            raise UserError(f"{f}: {ex}")
        jobs.append((f, path, route, info))
    backup = os.path.join(DIRS["routes"], ".backup", time.strftime("%Y-%m-%d %H%M%S") + " tags")
    os.makedirs(backup, exist_ok=True)
    out = []
    for f, path, route, info in jobs:
        shutil.copy2(path, os.path.join(backup, f))
        save_route(route, path)
        out.append({"file": f, **info})
    return jsonify({"routes": out, "backup": os.path.relpath(backup, ROOT)})


# ----------------------------------------------------------------------------- simulation
def _png_b64(fig, dpi=100):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    return base64.b64encode(buf.getvalue()).decode()


def _mask_b64(mask):
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[mask] = (255, 160, 60, 110)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def _attach_todo(agv, name):
    """Values marked '# TODO' in the vehicle file on disk (comments do not travel with the form data)."""
    from agv_sim.config import todo_fields
    if not name:
        return
    p = os.path.join(DIRS["configs"], os.path.basename(str(name)))
    try:
        with open(p, "r", encoding="utf-8") as f:
            agv._todo_fields = todo_fields(f.read())
    except OSError:
        pass


@app.post("/api/run")
def run():
    body = request.get_json() or {}
    agv = _agv_from(body.get("agv"))
    _attach_todo(agv, body.get("agv_name"))
    route = _route_from(body.get("route"))
    if len(route.waypoints) < 2:
        raise UserError("Add at least two route points first (click on the map).")
    with SIM_LOCK:
        fmap = FactoryMap.for_route(route)
        try:
            res, summary = run_simulation(agv, route, fmap)
        except ValueError as ex:
            raise UserError(str(ex))
        payload = _payload(agv, route, res, summary, fmap, files=True, chart=True)
    return jsonify(_json_safe(payload))


def _payload(agv, route, res, summary, fmap, files=True, chart=True):
    """Everything the browser needs to show one run (report, chart, playback)."""
    text = format_report(summary)
    mask = summary.get("_swept_mask")
    chart_png = None
    folder = re.sub(r"[^A-Za-z0-9_\-]+", "_", route.name or "route").strip("_") or "route"
    if chart or files:
        fig = plot_report(res, summary, fmap)
        chart_png = _png_b64(fig, dpi=100)
        import matplotlib.pyplot as plt
        plt.close(fig)
    if files:
        # write the usual result files as well
        out = os.path.join(DIRS["results"], folder)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "report.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        with open(os.path.join(out, "report.png"), "wb") as f:
            f.write(base64.b64decode(chart_png))
        _save_csv(res, os.path.join(out, "timeseries.csv"))

    # trajectory for the in-browser playback (20 frames per second of simulated time)
    step = max(1, int(round(0.05 / agv.simulation.dt)))
    sl = slice(None, None, step)
    traj = {"t": _r(res.t[sl]), "x": _r(res.x[sl]), "y": _r(res.y[sl]), "th": _r(res.th[sl], 4),
            "v": _r(res.v[sl]), "mode": [int(m) for m in res.mode[sl]]}
    L = res.loads
    traj["tq"] = _r(np.maximum(np.abs(L["motor_torque_l"][sl]), np.abs(L["motor_torque_r"][sl])))
    traj["grip"] = _r(np.maximum(L["traction_l"][sl], L["traction_r"][sl]))
    if res.lf is not None:
        traj["pv"] = _r(res.lf["pv"][sl], 0)
        traj["tv"] = _r(res.lf["tv"][sl], 2)
    if res.cart_x is not None:
        traj.update({"cx": _r(res.cart_x[sl]), "cy": _r(res.cart_y[sl]), "cth": _r(res.cart_th[sl], 4)})
    if res.hitch_x is not None:
        traj.update({"hx": _r(res.hitch_x[sl]), "hy": _r(res.hitch_y[sl])})
    if len(res.train) > 1:
        traj["tr"] = [{"x": _r(tr["x"][sl]), "y": _r(tr["y"][sl]), "th": _r(tr["th"][sl], 4)} for tr in res.train[1:]]
        traj["coupling"] = agv.cart.coupling_behind_axle
    if res.articulation_deg is not None:
        traj["art"] = _r(res.articulation_deg[sl], 1)

    shapes = [{"name": k, "pts": _r(v[0].ravel()), "fill": v[1]} for k, v in agv_shapes(agv, with_cart=False).items()]
    cart_poly = None
    if agv.hook_type in ("tow", "lurking") and res.cart_x is not None:
        cart_poly = _r(rect_corners(cart_rect(agv)).ravel())
    sensors = [{"name": s.name, "x": s.x, "y": s.y} for s in agv.sensors]
    return {
        "summary": summary, "report": text, "chart_png": chart_png if chart else None,
        "swept_png": _mask_b64(mask) if mask is not None else None,
        "traj": traj, "shapes": shapes, "cart_poly": cart_poly, "cart_type": agv.hook_type,
        "parked": res.parked, "sensors": sensors,
        "files": {k: f"{folder}/{k}" for k in ("report.txt", "report.png", "timeseries.csv")} if files else {},
        "rated_torque": agv.drivetrain.rated_torque, 
    }


# ----------------------------------------------------------------------------- sequences
@app.get("/api/sequence/<name>")
def get_sequence(name):
    from agv_sim.sequence import load_sequence
    path = os.path.join(DIRS["sequences"], _clean_name(name, ""))
    try:
        seq = load_sequence(path)
    except FileNotFoundError:
        raise UserError(f"Sequence file '{name}' not found.")
    except Exception as ex:
        raise UserError(f"Could not read '{name}': {ex}")
    return jsonify({"name": os.path.basename(path), "data": asdict(seq)})


@app.post("/api/sequence/<name>")
def put_sequence(name):
    from agv_sim.sequence import save_sequence, sequence_from_dict
    name = _clean_name(name, ".yaml")
    try:
        seq = sequence_from_dict((request.get_json() or {}).get("data"))
    except Exception as ex:
        raise UserError(f"Sequence: {ex}")
    save_sequence(seq, os.path.join(DIRS["sequences"], name))
    return jsonify({"name": name, "sequences": _list("sequences", (".yaml", ".yml"))})


@app.get("/api/rfidplan/<name>")
def get_rfidplan(name):
    from agv_sim.rfidplan import load_plan
    path = os.path.join(DIRS["rfid"], _clean_name(name, ""))
    try:
        plan = load_plan(path)
    except FileNotFoundError:
        raise UserError(f"RFID plan '{name}' not found.")
    except Exception as ex:
        raise UserError(f"Could not read '{name}': {ex}")
    return jsonify({"name": os.path.basename(path), "data": plan})


@app.post("/api/rfidplan/<name>")
def put_rfidplan(name):
    import time
    from agv_sim.rfidplan import plan_from_dict, save_plan
    name = _clean_name(name, ".yaml")
    try:
        plan = plan_from_dict((request.get_json() or {}).get("data"))
    except Exception as ex:
        raise UserError(f"RFID plan: {ex}")
    path = os.path.join(DIRS["rfid"], name)
    if os.path.exists(path):                       # keep the previous version
        bak = os.path.join(DIRS["rfid"], ".backup")
        os.makedirs(bak, exist_ok=True)
        shutil.copy2(path, os.path.join(bak, time.strftime("%Y-%m-%d %H%M%S ") + name))
    save_plan(plan, path)
    return jsonify({"name": name, "rfidplans": _list("rfid", (".yaml", ".yml"))})


@app.post("/api/run_sequence")
def run_sequence_api():
    """Simulate every step of a sequence; returns the table, the total and the playback of each step."""
    from agv_sim.sequence import format_sequence, run_sequence, sequence_from_dict
    try:
        seq = sequence_from_dict((request.get_json() or {}).get("sequence"))
    except Exception as ex:
        raise UserError(f"Sequence: {ex}")
    if not seq.steps:
        raise UserError("Add at least one step (route + vehicle).")
    payloads = {}

    def on_step(i, st, agv, route, res, s, row):
        fmap = FactoryMap.for_route(route)
        p = _payload(agv, route, res, s, fmap, files=False, chart=False)
        p["route"] = _route_to_dict(route)
        p["agv_name"] = st.agv
        payloads[i] = p
    with SIM_LOCK:
        out = run_sequence(seq, DIRS["routes"], DIRS["configs"], on_step=on_step)
    for row in out["steps"]:
        row["play"] = payloads.get(row["i"])
    out["report"] = format_sequence(out)
    return jsonify(_json_safe(out))


@app.post("/api/autoroute")
def autoroute_api():
    """Plan / optimize the route (see agv_sim/autoroute.py)."""
    from agv_sim.autoroute import auto_route
    body = request.get_json() or {}
    agv = _agv_from(body.get("agv"))
    route = _route_from(body.get("route"))
    if len(route.waypoints) < 2:
        raise UserError("Place at least a start point and one stop first.")
    o = body.get("options") or {}

    def num(k, d):
        try:
            return float(o.get(k, d)) if o.get(k, "") != "" else d
        except (TypeError, ValueError):
            raise UserError(f"Auto route: '{k}' must be a number.")
    kw = dict(mode="keep" if o.get("mode") == "keep" else "plan",
              style="any" if o.get("style") == "any" else "orthogonal",
              margin=num("margin", agv.simulation.safety_margin), max_radius=num("max_radius", 2.5),
              min_radius=num("min_radius", 0.75), tape_limit_pct=num("tape_limit", 70),
              fast_modes=o.get("fast_modes") or None, max_iter=int(num("max_iter", 6)),
              trolley_safe=bool(o.get("trolley_safe", True)),
              push_allow=(num("push_allow", 0.0) if o.get("push_allow", "") not in ("", None) else None))
    with SIM_LOCK:
        fmap = FactoryMap.for_route(route)
        before = None
        if o.get("compare", True) and o.get("mode") == "keep":
            try:
                _, sb = run_simulation(agv, route, fmap)
                before = _brief(sb)
            except Exception as ex:          # the old route may not even run
                before = {"error": str(ex)}
        try:
            out = auto_route(agv, route, fmap, **kw)
        except ValueError as ex:
            raise UserError(str(ex))
    return jsonify(_json_safe({"route": _route_to_dict(out["route"]), "log": out["log"],
                               "after": _brief(out["summary"]), "before": before}))


def _brief(s):
    lf = s.get("line_follow") or {}
    return {"cycle_time_s": s["cycle_time_s"], "avg_speed": s.get("avg_speed_overall"),
            "path_length_m": s["path_length_m"], "collisions": len(s.get("collisions", [])),
            "margin_spots": len(s.get("clearance_violations", [])), "tape_max_mm": lf.get("max_abs_mm"),
            "tape_lost": bool(lf.get("aborted") or lf.get("losses")), "complete": s.get("cycle_complete", True),
            "push_max_n": (s.get("trolley_push") or {}).get("max_n"),
            "push_over_s": (s.get("trolley_push") or {}).get("over_s")}


@app.post("/api/import_profile")
def import_profile():
    """AGV profile JSON (profiles/<agv>.json of the AGV code) -> line_follow values."""
    from agv_sim.profile_import import extras, profile_to_line_follow
    f = request.files.get("file")
    if not f:
        raise UserError("No file received.")
    try:
        data = json.loads(f.read().decode("utf-8"))
    except Exception as ex:
        raise UserError(f"Not a valid JSON profile: {ex}")
    vals = profile_to_line_follow(data)
    if not vals:
        raise UserError("No pid_tuning / speeds sections found in this JSON.")
    return jsonify({"line_follow": vals, "extras": extras(data), "agv_id": data.get("_agv_id", "")})


# ---------------------------------------------------------------- PID auto-tune (background job)
TUNE_JOBS = {}


@app.post("/api/tune_pid")
def tune_pid_start():
    """Start tuning Kp / Td of the tape PID on the route in the app (runs in the background)."""
    from agv_sim.pidtune import Tuner
    body = request.get_json() or {}
    agv = _agv_from(body.get("agv"))
    route = _route_from(body.get("route"))
    if len(route.waypoints) < 2:
        raise UserError("Draw a route first - the PID is tuned on it.")
    if not agv.line_follow.enabled:
        raise UserError("Switch on 'Simulate the real tape-following PID' in Vehicle → Line following first.")
    rounds = int((body.get("options") or {}).get("rounds", 8))
    jid = f"t{len(TUNE_JOBS) + 1}"
    job = {"log": [], "done": False, "error": None, "result": None, "round": 0, "rounds": rounds, "stop": False}
    TUNE_JOBS[jid] = job

    def work():
        try:
            tn = Tuner(agv, route, rounds=rounds, log=job["log"].append,
                       progress=lambda r, n: job.update(round=r, rounds=n), stop=lambda: job["stop"])
            job["result"] = tn.run()
        except Exception as ex:          # noqa: BLE001 - shown to the user
            job["error"] = str(ex)
            traceback.print_exc()
        job["done"] = True
    threading.Thread(target=work, daemon=True).start()
    return jsonify({"job": jid})


@app.get("/api/tune_pid/<jid>")
def tune_pid_status(jid):
    job = TUNE_JOBS.get(jid)
    if not job:
        raise UserError("Unknown tuning job (the simulator was restarted?).")
    return jsonify(_json_safe({k: v for k, v in job.items() if k != "stop"}))


@app.post("/api/tune_pid/<jid>/stop")
def tune_pid_stop(jid):
    if jid in TUNE_JOBS:
        TUNE_JOBS[jid]["stop"] = True
    return jsonify({"ok": True})


@app.post("/api/turning")
def turning():
    """Turning-radius study for the vehicle in the form."""
    from agv_sim.turning import turning_study
    body = request.get_json() or {}
    agv = _agv_from(body.get("agv"))
    try:
        radii = [float(r) for r in (body.get("radii") or []) if float(r) > 0] or None
        angle = float(body.get("angle") or 90)
        show = body.get("show")
        show = float(show) if show not in (None, "") else None
    except (TypeError, ValueError):
        raise UserError("Radii, angle and radius to show must be numbers.")
    if radii and len(radii) > 20:
        raise UserError("Please test at most 20 radii at once.")
    with SIM_LOCK:
        try:
            st = turning_study(agv, radii=radii, angle_deg=angle, with_cart=bool(body.get("with_cart", True)),
                               show_radius=show)
        except ValueError as ex:
            raise UserError(str(ex))
        fig = st.pop("figure")
        png = _png_b64(fig, dpi=95)
        import matplotlib.pyplot as plt
        plt.close(fig)
        out = os.path.join(DIRS["results"], "turning")
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "turning.txt"), "w", encoding="utf-8") as f:
            f.write(st["text"])
        with open(os.path.join(out, "turning.png"), "wb") as f:
            f.write(base64.b64decode(png))
    st["png"] = png
    st["files"] = {"turning.txt": "turning/turning.txt", "turning.png": "turning/turning.png"}
    return jsonify(_json_safe(st))


def _save_csv(res, path):
    from agv_sim.analysis import save_timeseries
    save_timeseries(res, path)


# ----------------------------------------------------------------------------- start
def _free_port(preferred=8765):
    for port in (preferred, 0):
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", port))
            p = s.getsockname()[1]
            s.close()
            return p
        except OSError:
            s.close()
    return preferred


def main():
    if not _list("maps", IMG_EXT):  # first start: make the demo floor plan
        demo = os.path.join(ROOT, "examples", "make_demo_map.py")
        if os.path.exists(demo):
            os.system(f'"{sys.executable}" "{demo}"')
    port = int(os.environ.get("AGV_SIM_PORT", 0)) or _free_port()
    url = f"http://127.0.0.1:{port}/"
    print("\n  AGV Simulator is running at", url)
    print("  (your browser opens automatically - keep this window open, press Ctrl+C to stop)\n")
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)      # no request spam in the terminal
    import flask.cli
    flask.cli.show_server_banner = lambda *a, **k: None        # no "development server" warning
    if "--no-browser" not in sys.argv:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
