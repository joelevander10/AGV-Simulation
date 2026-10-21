"""Draw several routes together on one map: only the tape and the stop areas.

    python tools/combined_map.py                       # every routes/*_auto.yaml
    python tools/combined_map.py "AGV_Kayaba (UBC)_auto.yaml" "AGV_Kayaba (OCC)_auto.yaml"
    python tools/combined_map.py --map --full          # with the floor plan, whole map
    python tools/combined_map.py --colors              # one colour per route instead of one tape colour
    python tools/combined_map.py --no-stops            # only the route lines

Tape: all routes in one colour, so a shared tape is one line. Stop areas: yellow boxes, numbered;
stops of different routes at the same place (closer than --merge m) are one area, the list says
which route stops there and what it does. The same view is in the web app (tab "All routes").
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agv_sim.combined import area_text, combine
from agv_sim.config import MapConfig
from agv_sim.factory_map import FactoryMap

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("routes", nargs="*", help="route files (in routes/); default: every *_auto.yaml")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "combined_routes.png"))
    ap.add_argument("--map", action="store_true", help="draw the floor plan under the track")
    ap.add_argument("--full", action="store_true", help="whole map (default: crop to the routes)")
    ap.add_argument("--colors", action="store_true", help="one colour per route")
    ap.add_argument("--no-stops", action="store_true", help="only the route lines, no stop areas")
    ap.add_argument("--merge", type=float, default=2.5, help="stops closer than this (m) are one stop area")
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args(argv)
    files = [os.path.basename(p) for p in a.routes] or \
        [os.path.basename(p) for p in sorted(glob.glob(os.path.join(ROOT, "routes", "*_auto.yaml")))]
    if not files:
        sys.exit("no routes found")
    C = combine(files, os.path.join(ROOT, "routes"), os.path.join(ROOT, "configs"), merge=a.merge)
    for w in C["warnings"]:
        print("!", w)
    if not C["routes"]:
        sys.exit("nothing to draw")
    m = C["map"]
    fmap = FactoryMap.from_config(MapConfig(image=os.path.join(ROOT, "maps", m["image"]) if m["image"] else "",
                                            meters_per_pixel=m["meters_per_pixel"]))
    xs = np.concatenate([np.asarray(r["x"]) for R in C["routes"] for r in R["runs"]])
    ys = np.concatenate([np.asarray(r["y"]) for R in C["routes"] for r in R["runs"]])
    if a.full:
        x0, x1, y0, y1 = 0, fmap.width_m, 0, fmap.height_m
    else:
        mg = 6.0
        x0, x1 = max(0, xs.min() - mg), min(fmap.width_m, xs.max() + mg)
        y0, y1 = max(0, ys.min() - mg), min(fmap.height_m, ys.max() + mg)
    w = 18.0
    fig, ax = plt.subplots(figsize=(w, max(4.0, w * (y1 - y0) / (x1 - x0)) + 1.2))
    if a.map:
        ax.imshow(fmap.rgb, extent=fmap.extent, origin="upper", interpolation="bilinear", alpha=0.35)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.grid(True, color="#eeeeee", lw=0.6, zorder=0)

    for k, R in enumerate(C["routes"]):
        c = COLORS[k % len(COLORS)] if a.colors else "#222222"
        for j, r in enumerate(R["runs"]):
            ax.plot(r["x"], r["y"], "-", color=c, lw=3.0 if not a.colors else 2.2, solid_capstyle="round", zorder=2,
                    alpha=0.9, label=(f"{R['label']} ({R['length_m']:.0f} m)" if a.colors and j == 0 else None))
    if not a.colors:
        ax.plot([], [], "-", color="#222222", lw=3.0, label="magnetic tape (all routes)")
    if a.no_stops:
        C["areas"] = []
    for ar in C["areas"]:
        ax.add_patch(plt.Polygon(ar["poly"], closed=True, fc="#ffcc00", ec="#b8860b", lw=1.5, alpha=0.85, zorder=4))
        ax.text(ar["x"], ar["y"], str(ar["n"]), ha="center", va="center", fontsize=8, weight="bold", zorder=5)
        vert = abs(math.sin(ar["heading"])) > 0.7
        ax.annotate(ar["name"], (ar["x"], ar["y"]), (14, 0) if vert else (0, 12), textcoords="offset points",
                    ha="left" if vert else "center", va="center" if vert else "bottom", fontsize=9,
                    weight="bold", color="#8a6500", zorder=6)
    if C["areas"]:
        ax.plot([], [], "s", color="#ffcc00", mec="#b8860b", ms=10, label="stop area")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    if C["areas"]:
        ax.text(0.01, 0.02, "Stop areas\n" + "\n".join(area_text(ar) for ar in C["areas"]), transform=ax.transAxes,
                fontsize=8.5, va="bottom", ha="left", family="monospace", zorder=7,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#b8860b", alpha=0.95))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(("Routes: " if a.no_stops else "Track and stop areas: ") + ", ".join(R["label"] for R in C["routes"]), fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=a.dpi)
    print("saved", a.out)
    for R in C["routes"]:
        print(f"  {R['label']:12s} {R['length_m']:7.1f} m  {R['stops']} stops  (vehicle {R['vehicle']})")
    for ar in C["areas"]:
        print(f"  {area_text(ar)}   @ ({ar['x']:.1f}, {ar['y']:.1f}) m")


if __name__ == "__main__":
    main()
