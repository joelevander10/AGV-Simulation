"""PID auto-tune for the tape-following controller (kayaba_pid.py) on a route.

Every candidate set of gains is driven over the whole route with the closed-loop simulation
(MGS1600 model, your controller, motor lag) under slightly HARSHER conditions than configured
(more sensor noise, more latency, slower motors, a mis-calibrated wheel), so the chosen values
keep a safety margin on the real AGV.

Gain sets (like modes.py switches them):
  HIGH      kp, td, n                  (speed_mode HIGH)
  SLOW      kp_slow, td_slow, n_slow   (SLOW and every mode modes.py does not know, e.g. MED)
  APPROACH  kp_approach, td_approach, n_approach
  REVERSE   kp_reverse, td_reverse, n_reverse
Only Kp and Td are changed (N and Ti stay). Each set is judged only on the stretches it drives.

Cost of a set ("balanced", in mm):  RMS tape error + 0.25 x 99th percentile
    + 8 x sign changes of the error per metre (wobble)  + 5 x steering jitter / jitter at the start
    + 30 x share of time saturated  + 2 x mm above 60 % of the sensor range  + 500 per tape loss

Search: pattern search in log(Kp), log(Td); all sets move at the same time (each set only
affects its own stretches), candidates run in parallel processes. Kp and Td stay within 1/3 .. 3 x
your current values (the simulation is not the real AGV - move in steps). Finally a gain-margin
test: with 2 x the new Kp the AGV must still hold the tape without wobbling, else Kp is lowered.
"""
from __future__ import annotations

import copy
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

SETS = {"HIGH": ("kp", "td", "n"), "SLOW": ("kp_slow", "td_slow", "n_slow"),
        "APPROACH": ("kp_approach", "td_approach", "n_approach"),
        "REVERSE": ("kp_reverse", "td_reverse", "n_reverse")}
JSON_KEYS = {"kp": "KP", "td": "TD", "n": "N", "kp_slow": "KP_SLOW", "td_slow": "TD_SLOW", "n_slow": "N_SLOW",
             "kp_approach": "KP_APPROACH", "td_approach": "TD_APPROACH", "n_approach": "N_APPROACH",
             "kp_reverse": "KP_REVERSE", "td_reverse": "TD_REVERSE", "n_reverse": "N_REVERSE"}
KP_RANGE = (0.1, 20.0)
TD_RANGE = (0.005, 2.0)
MIN_DIST = 1.0          # m a set must drive to be tuned


def harsh(agv):
    """Copy of the AGV with slightly worse conditions than configured (safety margin)."""
    a = copy.deepcopy(agv)
    lf = a.line_follow
    lf.sensor_noise_mm = max(lf.sensor_noise_mm, 1.0)
    lf.sensor_latency_s = lf.sensor_latency_s + 0.01
    lf.motor_time_constant = lf.motor_time_constant * 1.3
    lf.plant_scale_left = lf.plant_scale_left * 0.97
    return a


def _gains(agv):
    lf = agv.line_follow
    return {s: (getattr(lf, f[0]), getattr(lf, f[1])) for s, f in SETS.items()}


def _set_gains(agv, g):
    lf = agv.line_follow
    for s, (kp, td) in g.items():
        setattr(lf, SETS[s][0], float(kp))
        setattr(lf, SETS[s][1], float(td))


def _metrics(st, rng):
    e = np.asarray(st["e"], float)
    n = len(e)
    if n == 0:
        return None
    dist = max(st["dist"], 1e-6)
    return {"rms": float(np.sqrt(np.mean(e ** 2))), "p99": float(np.percentile(e, 99)), "max": float(e.max()),
            "cross_per_m": st["cross"] / dist, "jitter": math.sqrt(st["jit2"] / max(n - 1, 1)),
            "sat": st["sat"] / n, "lost": int(st["lost"]), "dist": float(st["dist"]), "range": rng}


def evaluate(agv, route, gains=None, conditions="harsh"):
    """Drive the route once; returns {'sets': {set: metrics}, 'aborted', 'cycle_s', 'tape_max'}."""
    from .path import build_plan
    from .simulate import simulate
    a = harsh(agv) if conditions == "harsh" else copy.deepcopy(agv)
    if gains:
        _set_gains(a, gains)
    plan = build_plan(route, a)
    res = simulate(a, route, plan, collect_stats=True, light=True)
    rng = a.line_follow.sensor_range_mm
    sets = {k: m for k, st in (res.lf_stats or {}).items() if (m := _metrics(st, rng)) is not None}
    return {"sets": sets, "aborted": bool(res.lf and res.lf.get("aborted")), "cycle_s": float(res.t[-1]),
            "tape_max": max([m["max"] for m in sets.values()] or [0.0])}


def _job(args):
    agv, route, gains, cond = args
    try:
        return evaluate(agv, route, gains, cond)
    except Exception as ex:                       # a candidate that crashes = very bad
        return {"sets": {}, "aborted": True, "error": str(ex), "cycle_s": math.inf, "tape_max": math.inf}


def cost(m, jit0):
    if m is None:
        return 1e6
    J = m["rms"] + 0.25 * m["p99"] + 8.0 * m["cross_per_m"] + 5.0 * m["jitter"] / max(jit0, 1e-6)
    J += 30.0 * m["sat"] + 2.0 * max(0.0, m["max"] - 0.6 * m["range"]) + 500.0 * m["lost"]
    return J


class Tuner:
    def __init__(self, agv, route, rounds=10, workers=None, log=None, progress=None, stop=None):
        self.agv, self.route = agv, route
        self.rounds = int(rounds)
        self.workers = workers or max(1, min(4, (os.cpu_count() or 2) - 1))
        self._log, self._progress, self._stop = log, progress, stop
        self.evals = 0

    def log(self, msg):
        if self._log:
            self._log(msg)

    def _run(self, gain_list, cond="harsh"):
        jobs = [(self.agv, self.route, g, cond) for g in gain_list]
        out = None
        if self.workers > 1 and len(jobs) > 1:
            try:
                with ProcessPoolExecutor(max_workers=min(self.workers, len(jobs))) as ex:
                    out = list(ex.map(_job, jobs))
            except Exception as ex_:             # no worker processes here: run one by one
                self.log(f"  (parallel runs not available: {ex_} - running one by one)")
                self.workers = 1
        if out is None:
            out = [_job(j) for j in jobs]
        self.evals += len(jobs)
        return out

    def _costs(self, r):
        return {s: cost(r["sets"].get(s), self.jit0.get(s, 1.0)) + (1000.0 if r["aborted"] else 0.0)
                for s in self.active}

    def run(self):
        lf = self.agv.line_follow
        if not lf.enabled:
            raise ValueError("Line following (the PID simulation) is off - switch it on in Vehicle → Line following.")
        g0 = _gains(self.agv)
        self.log("Driving the route with your current gains (harsher than configured: more noise, "
                 "latency, slower motors, one wheel 3 % off) ...")
        base_h, base_n = self._run([g0], cond="harsh")[0], evaluate(self.agv, self.route, g0, "nominal")
        self.evals += 1
        start, start_r = g0, base_h
        if base_h["aborted"]:
            # the current gains lose the tape: look for a starting point that holds it (softer Kp, more
            # damping) - the search needs complete drives to judge every gain set
            self.log("  the current gains LOSE THE TAPE under the harsher conditions - looking for a starting point ...")
            seeds = [{s: (g0[s][0] * fk, g0[s][1] * ft) for s in SETS} for fk in (0.8, 0.6, 0.45) for ft in (1.5, 2.5, 3.0)]
            runs = self._run(seeds)
            ok = [(r["tape_max"], g, r) for g, r in zip(seeds, runs) if not r["aborted"]]
            if ok:
                ok.sort(key=lambda o: o[0])
                _, start, start_r = ok[0]
                self.log("  start from " + ", ".join(f"{s} Kp {start[s][0]:.2f} Td {start[s][1]:.3f}"
                                                        for s in SETS if s in start_r["sets"]))
        used = {s: m for s, m in start_r["sets"].items() if m["dist"] >= MIN_DIST}
        if not used:
            raise ValueError("The route never drives far enough in any speed mode to tune the PID.")
        self.active = [s for s in SETS if s in used]
        self.jit0 = {s: max(used[s]["jitter"], 1e-3) for s in self.active}
        names = {"SLOW": "SLOW (also MED and other modes)"}
        self.log("Gain sets used on this route: " + ", ".join(
            f"{names.get(s, s)} {used[s]['dist']:.0f} m" for s in self.active))
        if base_h["aborted"]:
            self.log("  with the current gains the AGV LOSES THE TAPE under the harsher conditions.")
        x = {s: [math.log(start[s][0]), math.log(start[s][1])] for s in self.active}
        step = {s: math.log(1.6) for s in self.active}
        J0 = self._costs(base_h)
        J = self._costs(start_r)
        best_r = start_r
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        lo = {s: (math.log(max(KP_RANGE[0], g0[s][0] / 3)), math.log(max(TD_RANGE[0], g0[s][1] / 3))) for s in self.active}
        hi = {s: (math.log(min(KP_RANGE[1], g0[s][0] * 3)), math.log(min(TD_RANGE[1], g0[s][1] * 3))) for s in self.active}

        def to_g(xx):
            g = dict(g0)
            for s, (lk, lt) in xx.items():
                g[s] = (float(math.exp(min(max(lk, lo[s][0]), hi[s][0]))),
                        float(math.exp(min(max(lt, lo[s][1]), hi[s][1]))))
            return g
        for rnd in range(1, self.rounds + 1):
            if self._stop and self._stop():
                self.log("Stopped.")
                break
            live = [s for s in self.active if step[s] > math.log(1.08)]
            if not live:
                break
            if self._progress:
                self._progress(rnd, self.rounds)
            cands = []
            for d in dirs:
                xx = {s: list(v) for s, v in x.items()}
                for s in live:
                    xx[s][0] = min(max(xx[s][0] + d[0] * step[s], lo[s][0]), hi[s][0])
                    xx[s][1] = min(max(xx[s][1] + d[1] * step[s], lo[s][1]), hi[s][1])
                cands.append(xx)
            outs = self._run([to_g(c) for c in cands])
            moved = {}
            for s in live:
                cs = [self._costs(o)[s] for o in outs]
                k = int(np.argmin(cs))
                if cs[k] < J[s] - 0.02:
                    moved[s] = cands[k][s]
                else:
                    step[s] *= 0.5
            if moved:
                xn = {s: list(v) for s, v in x.items()}
                xn.update(moved)
                r = self._run([to_g(xn)])[0]
                Jn = self._costs(r)
                ok = [s for s in moved if Jn[s] < J[s] - 0.02]
                bad = [s for s in moved if s not in ok]
                if bad:                                  # interaction between the sets: keep only the good moves
                    for s in bad:
                        step[s] *= 0.5
                    xn = {s: list(v) for s, v in x.items()}
                    for s in ok:
                        xn[s] = moved[s]
                    if ok:
                        r = self._run([to_g(xn)])[0]
                        Jn = self._costs(r)
                if ok and sum(Jn.values()) < sum(J.values()):
                    x, J, best_r = xn, Jn, r
                    for s in ok:
                        step[s] *= 1.2                   # good direction: be a bit bolder
            gnow = to_g(x)
            self.log(f"Round {rnd}: " + "  ".join(
                f"{s} Kp {gnow[s][0]:.2f} Td {gnow[s][1]:.3f} (score {J[s]:.1f}, start {J0[s]:.1f})"
                for s in self.active))
        g1 = to_g(x)
        # ---- gain margin: 2 x Kp must still hold the tape calmly (the real AGV has dynamics the
        # simulation does not know) - otherwise lower Kp
        wob0 = {s: start_r["sets"][s]["cross_per_m"] for s in self.active}
        for k in range(6):
            test = {s: (g1[s][0] * 2.0, g1[s][1]) for s in self.active}
            r2 = self._run([test])[0]
            weak = [s for s in self.active
                    if r2["aborted"] or s not in r2["sets"] or r2["sets"][s]["lost"]
                    or r2["sets"][s]["cross_per_m"] > max(1.0, 1.5 * wob0[s])]
            weak = [s for s in weak if g1[s][0] != g0[s][0] or k == 0]
            if not weak:
                self.log("Gain margin OK: with 2 x the new Kp the AGV still holds the tape without wobbling.")
                break
            for s in weak:
                g1[s] = (max(g0[s][0], g1[s][0] * 0.8) if g1[s][0] > g0[s][0] else g1[s][0] * 0.8, g1[s][1])
            self.log("Gain margin too small for " + ", ".join(weak) + " -> Kp lowered to "
                     + ", ".join(f"{g1[s][0]:.2f}" for s in weak))
        best_r = self._run([g1])[0]
        J = self._costs(best_r)
        after_n = evaluate(self.agv, self.route, g1, "nominal")
        self.evals += 1
        rows = []
        for s in self.active:
            f = SETS[s]
            rows.append({"set": s, "fields": list(f[:2]), "json": [JSON_KEYS[f[0]], JSON_KEYS[f[1]]],
                         "kp_old": g0[s][0], "td_old": g0[s][1], "kp_new": g1[s][0], "td_new": g1[s][1],
                         "n": getattr(lf, f[2]),
                         "before": base_n["sets"].get(s), "after": after_n["sets"].get(s),
                         "before_harsh": base_h["sets"].get(s), "after_harsh": best_r["sets"].get(s),
                         "score_old": J0[s], "score_new": J[s]})
        changes = {}
        for rw in rows:
            for fld, v_old, v_new in ((rw["fields"][0], rw["kp_old"], rw["kp_new"]),
                                      (rw["fields"][1], rw["td_old"], rw["td_new"])):
                if abs(v_new - v_old) > 1e-9:
                    changes[fld] = round(v_new, 3)
        self.log(f"Done after {self.evals} test drives." + ("" if changes else " Your current gains are already the best found."))
        return {"rows": rows, "changes": changes,
                "json": {JSON_KEYS[k]: v for k, v in changes.items()},
                "before": {"nominal": _brief(base_n), "harsh": _brief(base_h)},
                "after": {"nominal": _brief(after_n), "harsh": _brief(best_r)},
                "evals": self.evals}


def _brief(r):
    return {"aborted": r["aborted"], "tape_max": r["tape_max"], "cycle_s": r["cycle_s"]}


def format_result(out):
    L = []
    for rw in out["rows"]:
        b, a = rw["before"] or {}, rw["after"] or {}
        L.append(f"{rw['set']:9s} Kp {rw['kp_old']:.3f} -> {rw['kp_new']:.3f}   Td {rw['td_old']:.3f} -> {rw['td_new']:.3f}"
                 f"   (N {rw['n']:g})")
        if b and a:
            L.append(f"          tape RMS {b['rms']:.1f} -> {a['rms']:.1f} mm, 99% {b['p99']:.0f} -> {a['p99']:.0f} mm, "
                     f"max {b['max']:.0f} -> {a['max']:.0f} mm, wobble {b['cross_per_m']:.2f} -> {a['cross_per_m']:.2f} /m")
    bh, ah = out["before"]["harsh"], out["after"]["harsh"]
    L.append(f"Harsher conditions: tape max {bh['tape_max']:.0f} -> {ah['tape_max']:.0f} mm"
             + ("  (current gains LOSE the tape)" if bh["aborted"] else "")
             + ("  (new gains LOSE the tape!)" if ah["aborted"] else ""))
    if out["json"]:
        L.append("For agv2_kim.json (pid_tuning): " + ", ".join(f'"{k}": {v}' for k, v in out["json"].items()))
    return "\n".join(L)
