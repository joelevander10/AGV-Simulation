"""Interactive route editor: load a factory PNG, calibrate the scale, click waypoints, save YAML.

Mouse
  left click (empty spot)   add waypoint at the end
  shift + left click        insert waypoint into the nearest leg
  left drag on a waypoint   move it
  right click               delete nearest waypoint
Keys (act on the waypoint nearest the mouse cursor)
  s   cycle action   pass -> stop -> pivot
  r   toggle direction of the leg LEAVING the waypoint (forward / reverse)
  + / -   corner radius +/- 0.25 m        0  radius 0 (sharp corner = pivot)
  b   reset radius to route default
  d   dwell +5 s       D  dwell = 0
  l   toggle closed loop
  p   toggle smoothed-path preview
  c   calibrate scale: click two points, type the real distance in the box, press Enter
  u   undo last added waypoint
  w   write (save) the route YAML
  h   print this help
Names, speed caps and exact dwell times can be refined in the YAML afterwards.
"""
from __future__ import annotations

import copy
import math
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import TextBox

from .config import AGVConfig, RouteConfig, Waypoint, save_route
from .factory_map import FactoryMap
from .path import build_plan

ACTIONS = ["pass", "stop", "pivot"]
STYLE = {"pass": ("o", "#1f77b4"), "stop": ("s", "#d62728"), "pivot": ("D", "#ff7f0e")}


def _free_keys():
    """Release keys that matplotlib binds by default so the editor can use them."""
    rc = plt.rcParams
    rc["keymap.save"] = ["ctrl+s"]
    rc["keymap.yscale"] = []
    rc["keymap.xscale"] = []
    rc["keymap.pan"] = []
    rc["keymap.back"] = ["left", "backspace"]
    rc["keymap.home"] = ["home"]
    rc["keymap.grid"] = []
    rc["keymap.grid_minor"] = []


class RouteEditor:
    PICK_PX = 12

    def __init__(self, route: RouteConfig, route_path: str, agv: Optional[AGVConfig] = None):
        _free_keys()
        self.route = route
        self.path = route_path
        self.agv = agv or AGVConfig()
        self.fmap = FactoryMap.for_route(route)
        self.preview = True
        self.drag: Optional[int] = None
        self.sel: Optional[int] = None
        self.calib = []
        self.calibrating = False
        self.artists = []
        self.calib_artists = []

        self.fig, self.ax = plt.subplots(figsize=(14, 8.5))
        self.fig.subplots_adjust(bottom=0.1, left=0.05, right=0.98, top=0.93)
        self.img = self.ax.imshow(self.fmap.rgb, extent=self.fmap.extent, origin="upper",
                                  interpolation="nearest")
        self.ax.set_aspect("equal")
        box_ax = self.fig.add_axes([0.25, 0.02, 0.15, 0.04])
        self.box = TextBox(box_ax, "calibration distance (m) ", initial="")
        self.box.on_submit(self._on_calib_submit)
        self.status = self.fig.text(0.45, 0.035, "", fontsize=9)
        c = self.fig.canvas
        c.mpl_connect("button_press_event", self._on_press)
        c.mpl_connect("motion_notify_event", self._on_motion)
        c.mpl_connect("button_release_event", self._on_release)
        c.mpl_connect("key_press_event", self._on_key)
        self.info("Click to add waypoints. Press h for help, c to calibrate scale, w to save.")
        self.redraw()

    # ------------------------------------------------------------------ helpers
    def info(self, msg):
        self.status.set_text(msg)
        print(msg)
        self.fig.canvas.draw_idle()

    def _toolbar_busy(self):
        tb = getattr(self.fig.canvas, "toolbar", None)
        return tb is not None and getattr(tb, "mode", "") not in ("", None)

    def _nearest(self, event, limit_px=None):
        wps = self.route.waypoints
        if not wps or event.x is None:
            return None
        pts = self.ax.transData.transform([(w.x, w.y) for w in wps])
        d = np.hypot(pts[:, 0] - event.x, pts[:, 1] - event.y)
        i = int(np.argmin(d))
        if limit_px is not None and d[i] > limit_px:
            return None
        return i

    def _target(self, event):
        i = self._nearest(event, limit_px=60)
        return i if i is not None else self.sel

    # ------------------------------------------------------------------ mouse
    def _on_press(self, event):
        if event.inaxes is not self.ax or self._toolbar_busy() or event.xdata is None:
            return
        if self.calibrating and event.button == 1:
            self.calib.append((event.xdata, event.ydata))
            self.calib_artists += self.ax.plot(event.xdata, event.ydata, "m+", ms=15, mew=2)
            if len(self.calib) == 2:
                (x0, y0), (x1, y1) = self.calib
                self.calib_artists += self.ax.plot([x0, x1], [y0, y1], "m-", lw=2)
                self.info("Now type the real distance between the two points in the box and press Enter.")
            self.fig.canvas.draw_idle()
            return
        if event.button == 3:
            i = self._nearest(event, self.PICK_PX * 2)
            if i is not None:
                self.route.waypoints.pop(i)
                self.sel = None
                self.redraw()
            return
        if event.button != 1:
            return
        i = self._nearest(event, self.PICK_PX)
        if i is not None:
            self.drag = i
            self.sel = i
            return
        wp = Waypoint(x=float(event.xdata), y=float(event.ydata))
        if event.key == "shift" and len(self.route.waypoints) >= 2:
            k = self._nearest_leg(event.xdata, event.ydata)
            self.route.waypoints.insert(k + 1, wp)
            self.sel = k + 1
        else:
            self.route.waypoints.append(wp)
            self.sel = len(self.route.waypoints) - 1
        self.redraw()

    def _nearest_leg(self, x, y):
        wps = self.route.waypoints
        n = len(wps)
        legs = n if self.route.loop else n - 1
        best, bk = math.inf, 0
        for k in range(legs):
            a = np.array([wps[k].x, wps[k].y])
            b = np.array([wps[(k + 1) % n].x, wps[(k + 1) % n].y])
            ab = b - a
            t = np.clip(np.dot([x, y] - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
            d = np.hypot(*(a + t * ab - [x, y]))
            if d < best:
                best, bk = d, k
        return bk

    def _on_motion(self, event):
        if self.drag is None or event.inaxes is not self.ax or event.xdata is None:
            return
        wp = self.route.waypoints[self.drag]
        wp.x, wp.y = float(event.xdata), float(event.ydata)
        self.redraw(preview=False)

    def _on_release(self, event):
        if self.drag is not None:
            self.drag = None
            self.redraw()

    # ------------------------------------------------------------------ keys
    def _on_key(self, event):
        k = event.key
        if k == "h":
            print(__doc__)
            self.info("Help printed to the console.")
            return
        if k == "c":
            self.calibrating = True
            self.calib = []
            for a in self.calib_artists:
                a.remove()
            self.calib_artists = []
            self.info("CALIBRATE: click two points with a known real distance (e.g. a column pitch).")
            return
        if k == "l":
            self.route.loop = not self.route.loop
        elif k == "p":
            self.preview = not self.preview
        elif k == "w":
            save_route(self.route, self.path)
            self.info(f"Saved {self.path}  ({len(self.route.waypoints)} waypoints)")
            return
        elif k == "u":
            if self.route.waypoints:
                self.route.waypoints.pop()
        else:
            i = self._target(event)
            if i is None:
                return
            self.sel = i
            wp = self.route.waypoints[i]
            r = self.route.default_radius if wp.radius is None else wp.radius
            if k == "s":
                wp.action = ACTIONS[(ACTIONS.index(wp.action) + 1) % 3] if wp.action in ACTIONS else "stop"
            elif k == "r":
                wp.direction = "reverse" if wp.direction == "forward" else "forward"
            elif k in ("+", "="):
                wp.radius = round(r + 0.25, 3)
            elif k == "-":
                wp.radius = round(max(0.0, r - 0.25), 3)
            elif k == "0":
                wp.radius = 0.0
            elif k == "b":
                wp.radius = None
            elif k == "d":
                wp.dwell = wp.dwell + 5.0
                if wp.action == "pass":
                    wp.action = "stop"
            elif k == "D":
                wp.dwell = 0.0
            else:
                return
        self.redraw()

    def _on_calib_submit(self, text):
        if len(self.calib) != 2:
            self.info("Press c and click two points first.")
            return
        try:
            real = float(text)
        except ValueError:
            self.info("Type a number (metres).")
            return
        (x0, y0), (x1, y1) = self.calib
        old = self.fmap.mpp
        px = math.hypot(x1 - x0, y1 - y0) / old
        new = real / px
        f = new / old
        for wp in self.route.waypoints:
            wp.x *= f
            wp.y *= f
        self.fmap.set_scale(new)
        self.route.map.meters_per_pixel = new
        self.img.set_extent(self.fmap.extent)
        self.ax.set_xlim(0, self.fmap.width_m)
        self.ax.set_ylim(0, self.fmap.height_m)
        self.calibrating = False
        self.calib = []
        for a in self.calib_artists:
            a.remove()
        self.calib_artists = []
        self.info(f"Scale set: {new:.5f} m/pixel  (map {self.fmap.width_m:.1f} x {self.fmap.height_m:.1f} m)")
        self.redraw()

    # ------------------------------------------------------------------ drawing
    def redraw(self, preview=None):
        for a in self.artists:
            a.remove()
        self.artists = []
        ax = self.ax
        wps = self.route.waypoints
        n = len(wps)
        if n:
            xs = [w.x for w in wps] + ([wps[0].x] if self.route.loop and n > 1 else [])
            ys = [w.y for w in wps] + ([wps[0].y] if self.route.loop and n > 1 else [])
            self.artists += ax.plot(xs, ys, "-", color="grey", lw=0.8, alpha=0.7)
            for i, w in enumerate(wps):
                if w.direction == "reverse" and (i + 1 < n or self.route.loop) and n > 1:
                    nx = wps[(i + 1) % n]
                    self.artists += ax.plot([w.x, nx.x], [w.y, nx.y], "-", color="m", lw=2, alpha=0.6)
        show_prev = self.preview if preview is None else (preview and self.preview)
        err = ""
        if show_prev and n >= 2:
            try:
                plan = build_plan(self.route, self.agv)
                for run in plan.runs:
                    self.artists += ax.plot(run.x, run.y, "-", color="#08519c" if run.direction > 0 else "m",
                                            lw=1.8)
                    for arc in run.arcs:
                        self.artists.append(ax.text(arc.center[0], arc.center[1], f"R{arc.radius:.2f}",
                                                    fontsize=7, color="#08519c", ha="center"))
                if plan.warnings:
                    err = " | " + plan.warnings[0]
            except Exception as ex:  # keep editing even with an invalid route
                err = f" | preview error: {ex}"
        for i, w in enumerate(wps):
            mk, col = STYLE.get(w.action, STYLE["pass"])
            ms = 11 if i == self.sel else 8
            self.artists += ax.plot(w.x, w.y, mk, color=col, ms=ms, mec="k", zorder=5)
            lab = f"{i}{' ' + w.name if w.name else ''}"
            if w.action != "pass":
                lab += f" [{w.action}{f' {w.dwell:g}s' if w.dwell else ''}]"
            if w.radius is not None:
                lab += f" r={w.radius:g}"
            self.artists.append(ax.annotate(lab, (w.x, w.y), (6, 6), textcoords="offset points", fontsize=8,
                                            color="k", bbox=dict(fc="white", alpha=0.6, lw=0), zorder=6))
        ax.set_title(f"Route editor - {self.route.name}  |  {n} waypoints  |  loop={self.route.loop}  |  "
                     f"{self.fmap.mpp * 1000:.1f} mm/px  |  default R={self.route.default_radius:g} m{err}",
                     fontsize=9)
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()
